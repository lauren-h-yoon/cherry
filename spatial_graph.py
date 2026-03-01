#!/usr/bin/env python3
"""
spatial_graph.py - Z-ordered spatial graph visualization

Converts spatial_graph.json to NetworkX graph with z-order visualization.

Usage:
    # Basic visualization
    python spatial_graph.py --input spatial_graph.json --viz

    # Specific visualization types
    python spatial_graph.py --input spatial_graph.json --viz_type hierarchical

    # Export to different formats
    python spatial_graph.py --input spatial_graph.json --export graphml
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Install with: pip install networkx")


@dataclass
class SpatialNode:
    """Node in spatial graph representing a z-ordered entity."""
    id: str
    name: str
    category: str
    bbox: List[float]
    bbox_center: List[float]
    z_order: int
    relative_depth: float
    depth_stats: Dict


class SpatialSceneGraph:
    """
    Z-ordered spatial scene graph.

    Supports:
    - Graph construction from JSON
    - Multiple visualization styles (z-order based)
    - Export to various formats
    """

    def __init__(self):
        self.nodes: Dict[str, SpatialNode] = {}
        self.graph: Optional[nx.DiGraph] = None
        self.metadata: Dict = {}
        self.image_path: str = ""
        self.image_size: List[int] = []

    @classmethod
    def from_json(cls, json_path: str) -> "SpatialSceneGraph":
        """Load z-ordered spatial graph from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        graph = cls()
        graph.image_path = data.get("image_path", "")
        graph.image_size = data.get("image_size", [])
        graph.metadata = data.get("metadata", {})

        # Load nodes
        for node_data in data.get("nodes", []):
            node = SpatialNode(
                id=node_data["id"],
                name=node_data["name"],
                category=node_data["category"],
                bbox=node_data["bbox"],
                bbox_center=node_data["bbox_center"],
                z_order=node_data["z_order"],
                relative_depth=node_data["relative_depth"],
                depth_stats=node_data["depth_stats"]
            )
            graph.nodes[node.id] = node

        # Build NetworkX graph
        graph._build_nx_graph()

        return graph

    def _build_nx_graph(self):
        """Build NetworkX graph with z-ordered nodes."""
        if not HAS_NETWORKX:
            return

        self.graph = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            self.graph.add_node(
                node_id,
                name=node.name,
                category=node.category,
                bbox=node.bbox,
                bbox_center=node.bbox_center,
                z_order=node.z_order,
                relative_depth=node.relative_depth,
                depth_stats=node.depth_stats
            )

        # Add z-order edges (connecting consecutive z-order levels)
        z_ordered = self.get_z_ordered_entities()
        for i in range(len(z_ordered) - 1):
            self.graph.add_edge(
                z_ordered[i].id,
                z_ordered[i + 1].id,
                z_step=1
            )

    def get_z_ordered_entities(self) -> List[SpatialNode]:
        """Get entities sorted by z-order (front to back)."""
        return sorted(self.nodes.values(), key=lambda n: n.z_order)

    def get_entities_closer_than(self, entity_id: str) -> List[SpatialNode]:
        """Get entities closer to camera than the given entity."""
        entity = self.nodes.get(entity_id)
        if not entity:
            return []
        return [n for n in self.nodes.values() if n.z_order < entity.z_order]

    def get_entities_farther_than(self, entity_id: str) -> List[SpatialNode]:
        """Get entities farther from camera than the given entity."""
        entity = self.nodes.get(entity_id)
        if not entity:
            return []
        return [n for n in self.nodes.values() if n.z_order > entity.z_order]

    def get_depth_range(self) -> Tuple[float, float]:
        """Get min and max relative depth values."""
        if not self.nodes:
            return (0.0, 0.0)
        depths = [n.relative_depth for n in self.nodes.values()]
        return (min(depths), max(depths))

    def export_graphml(self, output_path: str):
        """Export to GraphML format."""
        if not HAS_NETWORKX:
            raise ImportError("networkx required for GraphML export")
        nx.write_graphml(self.graph, output_path)

    def export_gexf(self, output_path: str):
        """Export to GEXF format (for Gephi)."""
        if not HAS_NETWORKX:
            raise ImportError("networkx required for GEXF export")
        nx.write_gexf(self.graph, output_path)

    def export_json(self, output_path: str):
        """Export to JSON format."""
        z_ordered = self.get_z_ordered_entities()
        data = {
            "image_path": self.image_path,
            "image_size": self.image_size,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "category": n.category,
                    "bbox": n.bbox,
                    "bbox_center": n.bbox_center,
                    "z_order": n.z_order,
                    "relative_depth": n.relative_depth,
                    "depth_stats": n.depth_stats
                }
                for n in z_ordered
            ],
            "z_order_sequence": [n.id for n in z_ordered],
            "metadata": self.metadata
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def visualize(
        self,
        output_path: Optional[str] = None,
        viz_type: str = "hierarchical",
        figsize: Tuple[int, int] = (14, 10),
        show_legend: bool = True,
        node_size_by_depth: bool = True,
        title: Optional[str] = None
    ):
        """
        Visualize the z-ordered spatial graph.

        Args:
            output_path: Path to save visualization (shows if None)
            viz_type: Layout type - "hierarchical", "depth_layers", "circular", "spring"
            figsize: Figure size
            show_legend: Show category legend
            node_size_by_depth: Scale node size by depth (closer = larger)
            title: Plot title
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if not HAS_NETWORKX:
            raise ImportError("networkx required for visualization")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Compute layout
        if viz_type == "hierarchical":
            pos = self._hierarchical_layout()
        elif viz_type == "depth_layers":
            pos = self._depth_layer_layout()
        elif viz_type == "circular":
            pos = nx.circular_layout(self.graph)
        elif viz_type == "spring":
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        else:
            pos = self._hierarchical_layout()

        # Node colors by z-order (gradient from green=close to red=far)
        z_ordered = self.get_z_ordered_entities()
        n_entities = len(z_ordered)
        depth_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_entities))
        z_color_map = {node.id: depth_colors[i] for i, node in enumerate(z_ordered)}

        node_colors = [z_color_map[nid] for nid in self.graph.nodes()]

        # Node sizes by depth (closer = larger)
        if node_size_by_depth:
            node_sizes = [
                800 + 1200 * (1 - self.nodes[nid].relative_depth)
                for nid in self.graph.nodes()
            ]
        else:
            node_sizes = 1000

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=1.5
        )

        # Draw z-order edges (connecting consecutive depth levels)
        edge_list = list(self.graph.edges())
        if edge_list:
            nx.draw_networkx_edges(
                self.graph, pos, ax=ax,
                edgelist=edge_list,
                edge_color='#888888',
                alpha=0.4,
                arrows=True,
                arrowsize=12,
                style='dashed'
            )

        # Node labels with z-order and depth
        labels = {
            nid: f"{self.nodes[nid].name}\nz={self.nodes[nid].z_order} | d={self.nodes[nid].relative_depth:.2f}"
            for nid in self.graph.nodes()
        }
        nx.draw_networkx_labels(
            self.graph, pos, ax=ax,
            labels=labels,
            font_size=9,
            font_weight='bold'
        )

        # Legend
        if show_legend:
            # Category legend
            categories = list(set(n.category for n in self.nodes.values()))
            cat_colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
            category_patches = [
                mpatches.Patch(color=cat_colors[i], label=cat)
                for i, cat in enumerate(categories)
            ]

            # Depth legend
            depth_patches = [
                mpatches.Patch(color=plt.cm.RdYlGn_r(0.1), label='Close (z=0)'),
                mpatches.Patch(color=plt.cm.RdYlGn_r(0.9), label='Far (z=max)')
            ]

            ax.legend(
                handles=depth_patches + category_patches,
                loc='upper right',
                title='Depth & Categories',
                fontsize=8
            )

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(
                f"Z-Ordered Scene Graph\n{len(self.nodes)} entities (z=0 closest, z={n_entities-1} farthest)",
                fontsize=14, fontweight='bold'
            )

        ax.axis('off')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()

    def _hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on z-order."""
        pos = {}
        z_orders = sorted(set(n.z_order for n in self.nodes.values()))
        z_to_level = {z: i for i, z in enumerate(z_orders)}

        # Group nodes by z-order
        level_nodes = defaultdict(list)
        for nid, node in self.nodes.items():
            level_nodes[node.z_order].append(nid)

        # Position nodes
        for z_order, node_ids in level_nodes.items():
            level = z_to_level[z_order]
            y = 1 - level / max(len(z_orders) - 1, 1)

            for i, nid in enumerate(node_ids):
                x = (i + 1) / (len(node_ids) + 1)
                pos[nid] = (x, y)

        return pos

    def _depth_layer_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create layout with depth as y-axis, x from bbox center."""
        pos = {}

        # Normalize x positions from bbox centers
        x_coords = [n.bbox_center[0] for n in self.nodes.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        x_range = max_x - min_x if max_x > min_x else 1

        for nid, node in self.nodes.items():
            x = (node.bbox_center[0] - min_x) / x_range
            y = 1 - node.relative_depth  # Closer objects at top
            pos[nid] = (x, y)

        return pos

    def visualize_with_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 10)
    ):
        """
        Visualize z-ordered graph overlaid on original image.

        Args:
            image_path: Path to original image
            output_path: Path to save visualization
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Load and show image
        img = Image.open(image_path)
        axes[0].imshow(img)

        # Draw bounding boxes with z-order labels (color coded by depth)
        z_ordered = self.get_z_ordered_entities()
        n_entities = len(z_ordered)
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_entities))

        for node, color in zip(z_ordered, colors):
            x1, y1, x2, y2 = node.bbox
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(
                x1, y1 - 10,
                f"z={node.z_order}: {node.name} (d={node.relative_depth:.2f})",
                fontsize=9, color='white',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9)
            )

        axes[0].set_title(f"Z-Ordered Entities ({n_entities} objects)\nGreen=Close, Red=Far")
        axes[0].axis('off')

        # For the second subplot, create a depth-layer graph view
        if HAS_NETWORKX:
            pos = self._depth_layer_layout()

            # Node colors by depth
            z_color_map = {node.id: colors[i] for i, node in enumerate(z_ordered)}
            node_colors = [z_color_map[nid] for nid in self.graph.nodes()]

            # Node sizes by depth
            node_sizes = [
                600 + 800 * (1 - self.nodes[nid].relative_depth)
                for nid in self.graph.nodes()
            ]

            nx.draw_networkx_nodes(
                self.graph, pos, ax=axes[1],
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )

            # Draw z-order edges
            edge_list = list(self.graph.edges())
            if edge_list:
                nx.draw_networkx_edges(
                    self.graph, pos, ax=axes[1],
                    edgelist=edge_list,
                    edge_color='#888888',
                    alpha=0.4,
                    arrows=True,
                    arrowsize=10,
                    style='dashed'
                )

            labels = {nid: f"{self.nodes[nid].name}\nz={self.nodes[nid].z_order}"
                      for nid in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, ax=axes[1], labels=labels, font_size=8)

        axes[1].set_title("Z-Order Graph (Depth Layers)")
        axes[1].set_xlabel("← Left | Right →")
        axes[1].set_ylabel("← Far | Close →")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()

    def print_summary(self):
        """Print z-order summary."""
        print("=" * 60)
        print("Z-Ordered Scene Graph Summary")
        print("=" * 60)
        print(f"Image: {self.image_path}")
        print(f"Entities: {len(self.nodes)}")
        print()

        print("Z-Order (front to back):")
        print("-" * 50)
        for node in self.get_z_ordered_entities():
            print(f"  z={node.z_order}: {node.name} ({node.category})")
            print(f"       depth={node.relative_depth:.3f}, pixels={node.depth_stats.get('pixel_count', 0)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Z-Ordered Spatial Graph Visualization")

    # Input
    parser.add_argument("--input", "-i", required=True, help="Input spatial_graph.json")

    # Visualization
    parser.add_argument("--viz", action="store_true", help="Generate visualization")
    parser.add_argument("--viz_type", default="hierarchical",
                        choices=["hierarchical", "depth_layers", "circular", "spring"],
                        help="Visualization layout type")
    parser.add_argument("--viz_with_image", help="Visualize overlaid on image")
    parser.add_argument("--output", "-o", help="Output path for visualization")

    # Export
    parser.add_argument("--export", choices=["graphml", "gexf", "json"],
                        help="Export to format")
    parser.add_argument("--export_path", help="Export output path")

    # Query
    parser.add_argument("--query", choices=["closer", "farther", "summary"],
                        help="Query type")
    parser.add_argument("--entity", help="Entity name/id for queries")

    args = parser.parse_args()

    # Load graph
    print(f"Loading z-ordered graph from {args.input}...")
    graph = SpatialSceneGraph.from_json(args.input)

    # Print summary
    graph.print_summary()

    # Handle queries
    if args.query:
        if args.query == "summary":
            pass  # Already printed
        elif args.query == "closer" and args.entity:
            # Find entity by name or id
            entity_id = None
            for nid, node in graph.nodes.items():
                if node.name == args.entity or nid == args.entity:
                    entity_id = nid
                    break

            if entity_id:
                closer = graph.get_entities_closer_than(entity_id)
                print(f"\nEntities closer than {args.entity}:")
                for node in sorted(closer, key=lambda n: n.z_order):
                    print(f"  z={node.z_order}: {node.name}")
            else:
                print(f"Entity '{args.entity}' not found")

        elif args.query == "farther" and args.entity:
            entity_id = None
            for nid, node in graph.nodes.items():
                if node.name == args.entity or nid == args.entity:
                    entity_id = nid
                    break

            if entity_id:
                farther = graph.get_entities_farther_than(entity_id)
                print(f"\nEntities farther than {args.entity}:")
                for node in sorted(farther, key=lambda n: n.z_order):
                    print(f"  z={node.z_order}: {node.name}")
            else:
                print(f"Entity '{args.entity}' not found")

    # Export
    if args.export:
        export_path = args.export_path or f"spatial_graph.{args.export}"
        if args.export == "graphml":
            graph.export_graphml(export_path)
        elif args.export == "gexf":
            graph.export_gexf(export_path)
        elif args.export == "json":
            graph.export_json(export_path)
        print(f"Exported to {export_path}")

    # Visualization
    if args.viz:
        output_path = args.output or "z_order_graph.png"
        graph.visualize(
            output_path=output_path,
            viz_type=args.viz_type,
            title=f"Z-Order Graph: {Path(args.input).stem}"
        )

    if args.viz_with_image:
        output_path = args.output or "z_order_with_image.png"
        graph.visualize_with_image(
            image_path=args.viz_with_image,
            output_path=output_path
        )


if __name__ == "__main__":
    main()
