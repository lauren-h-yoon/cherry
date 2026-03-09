#!/usr/bin/env python3
"""
analyze_query_capacity.py - Analyze how many queries can be generated per image/task

For each image with a spatial graph, calculates the maximum number of unique
queries that can be generated for each task type.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from spatial_agent.ground_truth import GroundTruth
from spatial_agent.tasks import TaskGenerator
from spatial_agent.object_descriptor import ObjectDescriptorGenerator


def analyze_graph(graph_path: str) -> dict:
    """Analyze query capacity for a single spatial graph."""

    with open(graph_path) as f:
        graph_data = json.load(f)

    gt = GroundTruth(graph_path)
    descriptor_gen = ObjectDescriptorGenerator(gt)

    # Get all objects and describable objects
    all_objects = gt.get_all_objects()
    describable_objects = descriptor_gen.get_describable_objects()

    # Group by category
    by_category = defaultdict(list)
    for obj in all_objects:
        cat = obj.name.rsplit('_', 1)[0] if '_' in obj.name else obj.name
        by_category[cat].append(obj)

    num_objects = len(all_objects)
    num_describable = len(describable_objects)
    num_categories = len(by_category)

    # Unique objects (only one in their category)
    unique_objects = [obj for obj in all_objects
                      if len(by_category[obj.name.rsplit('_', 1)[0] if '_' in obj.name else obj.name]) == 1]
    num_unique = len(unique_objects)

    # Task capacity calculations
    capacity = {}

    # 1. Object Localization: one query per describable object
    capacity['object_localization'] = num_describable

    # 2. Path Planning: one query per target object (from fixed start position)
    #    Could also vary start positions, but typically use 1 start
    capacity['path_planning'] = num_objects

    # 3. Egocentric Spatial QA: directions × unique objects
    #    5 directions: left, right, foreground, background, nearest
    #    We use unique objects to avoid position hints in the question
    egocentric_directions = 5
    capacity['egocentric_spatial_qa'] = egocentric_directions * max(num_unique, 1)

    # 4. Allocentric Spatial QA: anchor_objects × target_objects × relations × facings
    #    NEW FORMAT: Reference object IS the anchor (no separate viewpoint positions)
    #    Anchor objects: any object can be anchor
    #    Target objects: any object from different category than anchor
    #    Relations: 4 (left, right, in_front, behind)
    #    Facings: 4 (camera, away, left, right)
    relations = 4
    facings = 4
    # For each category as anchor, can pair with objects from other categories
    # Total pairs = sum over each category of (objects in category × objects in other categories)
    anchor_target_pairs = 0
    for cat, objs in by_category.items():
        num_in_cat = len(objs)
        num_in_other_cats = num_objects - num_in_cat
        anchor_target_pairs += num_in_cat * num_in_other_cats

    capacity['allocentric_spatial_qa'] = anchor_target_pairs * relations * facings

    # Total
    capacity['total'] = sum(capacity.values())

    return {
        'graph_path': graph_path,
        'image_path': graph_data.get('image_path', ''),
        'num_objects': num_objects,
        'num_describable': num_describable,
        'num_categories': num_categories,
        'num_unique_objects': num_unique,
        'categories': list(by_category.keys()),
        'objects': [obj.name for obj in all_objects],
        'capacity': capacity
    }


def print_analysis(analysis: dict):
    """Pretty print the analysis."""
    print(f"\n{'='*70}")
    print(f"Image: {analysis['image_path']}")
    print(f"Graph: {analysis['graph_path']}")
    print(f"{'='*70}")

    print(f"\nScene Statistics:")
    print(f"  Total objects: {analysis['num_objects']}")
    print(f"  Describable objects: {analysis['num_describable']}")
    print(f"  Object categories: {analysis['num_categories']}")
    print(f"  Unique objects (1 per category): {analysis['num_unique_objects']}")

    print(f"\nCategories: {', '.join(analysis['categories'])}")
    print(f"Objects: {', '.join(analysis['objects'])}")

    print(f"\nQuery Capacity by Task Type:")
    print(f"  {'Task Type':<30} {'Max Queries':>12} {'Formula'}")
    print(f"  {'-'*30} {'-'*12} {'-'*30}")

    cap = analysis['capacity']
    print(f"  {'Object Localization':<30} {cap['object_localization']:>12} (# describable objects)")
    print(f"  {'Path Planning':<30} {cap['path_planning']:>12} (# target objects)")
    print(f"  {'Egocentric Spatial QA':<30} {cap['egocentric_spatial_qa']:>12} (5 directions × # unique objects)")
    print(f"  {'Allocentric Spatial QA':<30} {cap['allocentric_spatial_qa']:>12} (anchor×target pairs × 4 relations × 4 facings)")
    print(f"  {'-'*30} {'-'*12}")
    print(f"  {'TOTAL':<30} {cap['total']:>12}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze query capacity for spatial graphs")
    parser.add_argument("--graph", "-g", help="Path to specific spatial graph JSON")
    parser.add_argument("--all", "-a", action="store_true", help="Analyze all spatial graphs in spatial_outputs/")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    graphs = []

    if args.graph:
        graphs = [args.graph]
    elif args.all:
        graphs = list(Path("spatial_outputs").glob("*_spatial_graph.json"))
    else:
        # Default: analyze all available
        graphs = list(Path("spatial_outputs").glob("*_spatial_graph.json"))

    if not graphs:
        print("No spatial graphs found. Generate them first with the pipeline.")
        return

    all_analyses = []
    for graph_path in graphs:
        analysis = analyze_graph(str(graph_path))
        all_analyses.append(analysis)

        if not args.json:
            print_analysis(analysis)

    if args.json:
        print(json.dumps(all_analyses, indent=2))
    elif len(all_analyses) > 1:
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL IMAGES")
        print(f"{'='*70}")

        total_by_task = defaultdict(int)
        for a in all_analyses:
            for task, count in a['capacity'].items():
                total_by_task[task] += count

        print(f"\n  {'Task Type':<30} {'Total Queries':>12}")
        print(f"  {'-'*30} {'-'*12}")
        for task in ['object_localization', 'path_planning', 'egocentric_spatial_qa', 'allocentric_spatial_qa']:
            print(f"  {task:<30} {total_by_task[task]:>12}")
        print(f"  {'-'*30} {'-'*12}")
        print(f"  {'GRAND TOTAL':<30} {total_by_task['total']:>12}")


if __name__ == "__main__":
    main()
