#!/usr/bin/env python3
"""
attention_visualizer.py - Visualize attention patterns in Qwen-VL.

This module provides visualization tools for understanding how Qwen-VL
attends to different parts of the image when answering spatial queries.

Key visualizations:
- Attention heatmaps overlaid on images
- Token-level attention patterns
- Cross-attention between text and image tokens
- Comparison of attention patterns for success vs failure cases
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


def get_image_token_positions(
    input_ids: torch.Tensor,
    processor,
    image_token_id: int = None
) -> Tuple[int, int]:
    """
    Find the start and end positions of image tokens in the sequence.

    Args:
        input_ids: Token IDs (batch, seq_len)
        processor: Qwen-VL processor
        image_token_id: ID of the image placeholder token

    Returns:
        (start_idx, end_idx) of image token range
    """
    if image_token_id is None:
        # Qwen2.5-VL uses <|image_pad|> token for image positions
        # Look for the vision token range
        # In Qwen2.5-VL, image tokens are marked differently
        # We'll estimate based on typical patterns
        ids = input_ids[0].tolist()

        # Find contiguous range of specific token IDs that represent image patches
        # This is a heuristic - actual implementation may vary
        start_idx = 0
        end_idx = len(ids)

        # Look for typical patterns
        for i, token_id in enumerate(ids):
            # Image tokens in Qwen are typically in a specific range
            # This is approximate - adjust based on actual tokenizer
            if token_id < 1000:  # Low IDs often indicate special tokens
                start_idx = i
                break

        return start_idx, end_idx

    ids = input_ids[0].tolist()
    start_idx = None
    end_idx = None

    for i, token_id in enumerate(ids):
        if token_id == image_token_id:
            if start_idx is None:
                start_idx = i
            end_idx = i + 1

    return start_idx or 0, end_idx or len(ids)


def attention_to_heatmap(
    attention_weights: torch.Tensor,
    image_size: Tuple[int, int],
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    head_idx: int = None
) -> np.ndarray:
    """
    Convert attention weights to a 2D heatmap that can be overlaid on an image.

    Args:
        attention_weights: Attention tensor of shape (batch, heads, q_len, kv_len)
        image_size: Original image size (height, width)
        patch_size: Size of each image patch (default 14 for Qwen)
        spatial_merge_size: Spatial merge factor (default 2 for Qwen)
        head_idx: If specified, use only this attention head; otherwise average

    Returns:
        Heatmap array of shape (height, width)
    """
    # Handle different attention shapes
    if attention_weights.dim() == 4:
        attn = attention_weights[0]  # Remove batch dim
    else:
        attn = attention_weights

    # Select or average heads
    if head_idx is not None:
        attn = attn[head_idx:head_idx+1]  # Keep dim
    attn = attn.mean(dim=0)  # Average over heads -> (q_len, kv_len)

    # Get the attention from last token (generation position) to all keys
    # Or average over query positions
    if attn.dim() == 2:
        attn = attn[-1]  # Last query position -> (kv_len,)

    attn = attn.cpu().numpy()

    # Calculate number of image patches
    h, w = image_size
    effective_patch = patch_size * spatial_merge_size
    num_patches_h = h // effective_patch
    num_patches_w = w // effective_patch
    num_image_tokens = num_patches_h * num_patches_w

    # Extract attention to image tokens (assuming they're at the start)
    if len(attn) >= num_image_tokens:
        image_attn = attn[:num_image_tokens]
    else:
        # Pad if necessary
        image_attn = np.zeros(num_image_tokens)
        image_attn[:len(attn)] = attn

    # Reshape to spatial grid
    try:
        heatmap = image_attn.reshape(num_patches_h, num_patches_w)
    except ValueError:
        # Fallback: create approximate grid
        side = int(np.sqrt(len(image_attn)))
        if side * side == len(image_attn):
            heatmap = image_attn.reshape(side, side)
        else:
            heatmap = image_attn[:side*side].reshape(side, side)

    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Resize to original image size
    from PIL import Image as PILImage
    heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize((w, h), PILImage.BILINEAR)
    heatmap = np.array(heatmap_pil) / 255.0

    return heatmap


def visualize_attention_on_image(
    image_path: str,
    attention_weights: torch.Tensor,
    output_path: str,
    title: str = "",
    alpha: float = 0.5,
    cmap: str = "jet"
):
    """
    Overlay attention heatmap on original image.

    Args:
        image_path: Path to original image
        attention_weights: Attention tensor
        output_path: Path to save visualization
        title: Plot title
        alpha: Transparency of heatmap overlay
        cmap: Colormap name
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Get heatmap
    heatmap = attention_to_heatmap(
        attention_weights,
        image_size=(img_array.shape[0], img_array.shape[1])
    )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    im = axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img_array)
    axes[2].imshow(heatmap, cmap=cmap, alpha=alpha)
    axes[2].set_title("Attention Overlay")
    axes[2].axis("off")

    if title:
        plt.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def visualize_layer_attention_progression(
    image_path: str,
    layer_attentions: Dict[str, torch.Tensor],
    output_path: str,
    title: str = ""
):
    """
    Visualize how attention evolves across layers.

    Args:
        image_path: Path to original image
        layer_attentions: Dict mapping layer names to attention tensors
        output_path: Path to save visualization
        title: Plot title
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    img_size = (img_array.shape[0], img_array.shape[1])

    # Sort layers by name (assumes numeric suffixes)
    sorted_layers = sorted(layer_attentions.keys())

    n_layers = len(sorted_layers)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for idx, layer_name in enumerate(sorted_layers):
        row = idx // cols
        col = idx % cols

        heatmap = attention_to_heatmap(layer_attentions[layer_name], img_size)

        ax = axes[row][col] if rows > 1 else axes[col]
        ax.imshow(img_array)
        ax.imshow(heatmap, cmap="jet", alpha=0.5)

        # Short layer name for title
        short_name = layer_name.split(".")[-1]
        if "layers" in layer_name:
            layer_num = layer_name.split("layers.")[-1].split(".")[0]
            short_name = f"Layer {layer_num}"
        ax.set_title(short_name, fontsize=10)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(n_layers, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def compare_success_failure_attention(
    image_path: str,
    success_attentions: List[Dict[str, torch.Tensor]],
    failure_attentions: List[Dict[str, torch.Tensor]],
    layer_name: str,
    output_path: str,
    success_queries: List[str] = None,
    failure_queries: List[str] = None
):
    """
    Compare attention patterns between success and failure cases.

    Args:
        image_path: Path to benchmark image
        success_attentions: List of attention dicts for success cases
        failure_attentions: List of attention dicts for failure cases
        layer_name: Layer to visualize
        output_path: Output path
        success_queries: Query texts for success cases
        failure_queries: Query texts for failure cases
    """
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    img_size = (img_array.shape[0], img_array.shape[1])

    n_success = min(3, len(success_attentions))
    n_failure = min(3, len(failure_attentions))

    fig, axes = plt.subplots(2, max(n_success, n_failure) + 1, figsize=(5*(max(n_success, n_failure)+1), 10))

    # Success cases
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Image", fontsize=10)
    axes[0, 0].axis("off")

    for idx, attn_dict in enumerate(success_attentions[:n_success]):
        if layer_name not in attn_dict:
            continue
        heatmap = attention_to_heatmap(attn_dict[layer_name], img_size)

        ax = axes[0, idx + 1]
        ax.imshow(img_array)
        ax.imshow(heatmap, cmap="Greens", alpha=0.6)

        title = f"Success {idx+1}"
        if success_queries and idx < len(success_queries):
            title = success_queries[idx][:30] + "..."
        ax.set_title(title, fontsize=9, color="green")
        ax.axis("off")

    # Failure cases
    axes[1, 0].imshow(img_array)
    axes[1, 0].set_title("Original Image", fontsize=10)
    axes[1, 0].axis("off")

    for idx, attn_dict in enumerate(failure_attentions[:n_failure]):
        if layer_name not in attn_dict:
            continue
        heatmap = attention_to_heatmap(attn_dict[layer_name], img_size)

        ax = axes[1, idx + 1]
        ax.imshow(img_array)
        ax.imshow(heatmap, cmap="Reds", alpha=0.6)

        title = f"Failure {idx+1}"
        if failure_queries and idx < len(failure_queries):
            title = failure_queries[idx][:30] + "..."
        ax.set_title(title, fontsize=9, color="red")
        ax.axis("off")

    # Hide unused
    for row in range(2):
        n_used = (n_success if row == 0 else n_failure) + 1
        for col in range(n_used, max(n_success, n_failure) + 1):
            axes[row, col].axis("off")

    plt.suptitle(f"Attention Comparison: {layer_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def extract_and_visualize_attention(
    model,
    processor,
    image_path: str,
    prompt: str,
    output_dir: str,
    layers_to_visualize: List[str] = None
):
    """
    End-to-end attention extraction and visualization.

    Args:
        model: Qwen-VL model
        processor: Qwen-VL processor
        image_path: Path to image
        prompt: Query prompt
        output_dir: Output directory
        layers_to_visualize: List of layer names (default: select key layers)
    """
    import torch
    from qwen_vl_utils import process_vision_info
    from .activation_extractor import ActivationExtractor

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default layers
    if layers_to_visualize is None:
        layers_to_visualize = [
            "model.language_model.layers.0.self_attn",
            "model.language_model.layers.7.self_attn",
            "model.language_model.layers.14.self_attn",
            "model.language_model.layers.21.self_attn",
            "model.language_model.layers.27.self_attn",
        ]

    # Set up extractor
    extractor = ActivationExtractor(model)
    extractor.register_hooks(layers_to_visualize, capture_attention=True)

    # Build input
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention from hooks
    attention_weights = extractor.get_attention_weights()

    # Visualize
    for layer_name, attn in attention_weights.items():
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        visualize_attention_on_image(
            image_path,
            attn,
            str(output_path / f"attention_{safe_name}.png"),
            title=f"Attention: {layer_name}"
        )

    # Visualize progression
    if len(attention_weights) > 1:
        visualize_layer_attention_progression(
            image_path,
            attention_weights,
            str(output_path / "attention_progression.png"),
            title="Attention Across Layers"
        )

    extractor.remove_hooks()

    print(f"Visualizations saved to {output_path}")


if __name__ == "__main__":
    print("Attention visualizer module loaded.")
    print("Use visualize_attention_on_image() or extract_and_visualize_attention() for analysis.")
