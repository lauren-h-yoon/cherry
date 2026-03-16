#!/usr/bin/env python3
"""
prompt_sources.py - Unified prompt generation for SAM3 object detection.

Supports three sources for object prompts:
1. COCO Vocabulary: Use predefined COCO category names
2. COCO Ground-Truth: Look up actual objects in COCO annotations for a specific image
3. GPT-4o Auto-Detection: Use vision model to detect objects in any image

Usage:
    from prompt_sources import PromptGenerator

    generator = PromptGenerator(
        coco_annotations_path="annotations/instances_val2017.json",  # Optional
        cache_dir="prompt_cache",
    )

    # Option 1: COCO vocabulary (indoor scene subset)
    prompts = generator.from_coco_vocabulary(scene_type="indoor")

    # Option 2: COCO ground-truth for specific image
    prompts = generator.from_coco_annotations(image_filename="000000397133.jpg")

    # Option 3: GPT-4o auto-detection
    prompts = generator.from_gpt4o(image_path="photos/living_room.jpg")
"""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PIL import Image


# =============================================================================
# COCO Category Definitions
# =============================================================================

# All 80 COCO categories
COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}

# Scene-specific subsets
COCO_INDOOR_CATEGORIES = [
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "remote", "keyboard", "cell phone", "sink",
    "book", "clock", "vase", "teddy bear", "bottle", "cup", "bowl",
]

COCO_LIVING_ROOM_CATEGORIES = [
    "chair", "couch", "potted plant", "tv", "remote", "book", "clock",
    "vase", "bottle", "cup", "lamp",  # lamp not in COCO but commonly needed
]

COCO_KITCHEN_CATEGORIES = [
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "microwave", "oven", "toaster", "sink", "refrigerator", "chair",
    "dining table",
]

COCO_BEDROOM_CATEGORIES = [
    "bed", "chair", "clock", "book", "laptop", "cell phone", "teddy bear",
    "potted plant", "vase",
]

COCO_OUTDOOR_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "bench",
    "bird", "cat", "dog", "backpack", "umbrella", "handbag",
]

# Extended categories (not in COCO but useful for indoor scenes)
EXTENDED_INDOOR_CATEGORIES = [
    "lamp", "painting", "mirror", "rug", "curtain", "pillow", "blanket",
    "shelf", "cabinet", "drawer", "door", "window", "sofa", "armchair",
    "coffee table", "side table", "desk", "wardrobe", "nightstand",
]


# =============================================================================
# Prompt Generator Class
# =============================================================================

class PromptGenerator:
    """
    Unified prompt generator supporting multiple sources.
    """

    def __init__(
        self,
        coco_annotations_path: Optional[str] = None,
        cache_dir: str = "prompt_cache",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the prompt generator.

        Args:
            coco_annotations_path: Path to COCO instances JSON (e.g., instances_val2017.json)
            cache_dir: Directory to cache GPT-4o detection results
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._coco_data: Optional[Dict] = None
        self._coco_path = coco_annotations_path

        self._openai_client = None
        self._openai_api_key = openai_api_key

    # -------------------------------------------------------------------------
    # Option 1: COCO Vocabulary
    # -------------------------------------------------------------------------

    def from_coco_vocabulary(
        self,
        scene_type: str = "indoor",
        include_extended: bool = True,
        custom_additions: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get prompts from predefined COCO category vocabulary.

        Args:
            scene_type: One of "all", "indoor", "living_room", "kitchen", "bedroom", "outdoor"
            include_extended: Include non-COCO categories useful for indoor scenes
            custom_additions: Additional custom prompts to include

        Returns:
            List of object prompts
        """
        if scene_type == "all":
            prompts = list(COCO_CATEGORIES.values())
        elif scene_type == "indoor":
            prompts = COCO_INDOOR_CATEGORIES.copy()
        elif scene_type == "living_room":
            prompts = COCO_LIVING_ROOM_CATEGORIES.copy()
        elif scene_type == "kitchen":
            prompts = COCO_KITCHEN_CATEGORIES.copy()
        elif scene_type == "bedroom":
            prompts = COCO_BEDROOM_CATEGORIES.copy()
        elif scene_type == "outdoor":
            prompts = COCO_OUTDOOR_CATEGORIES.copy()
        else:
            raise ValueError(f"Unknown scene_type: {scene_type}")

        if include_extended and scene_type in ("indoor", "living_room", "bedroom"):
            prompts.extend(EXTENDED_INDOOR_CATEGORIES)

        if custom_additions:
            prompts.extend(custom_additions)

        # Remove duplicates while preserving order
        seen = set()
        unique_prompts = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique_prompts.append(p)

        return unique_prompts

    # -------------------------------------------------------------------------
    # Option 2: COCO Ground-Truth Annotations
    # -------------------------------------------------------------------------

    def _load_coco_annotations(self) -> Dict:
        """Load COCO annotations lazily."""
        if self._coco_data is None:
            if self._coco_path is None:
                raise ValueError(
                    "COCO annotations path not provided. "
                    "Initialize with coco_annotations_path parameter."
                )
            with open(self._coco_path, "r") as f:
                self._coco_data = json.load(f)

            # Build lookup tables
            self._coco_data["_category_lookup"] = {
                cat["id"]: cat["name"]
                for cat in self._coco_data["categories"]
            }
            self._coco_data["_image_lookup"] = {
                img["file_name"]: img["id"]
                for img in self._coco_data["images"]
            }
            self._coco_data["_annotations_by_image"] = {}
            for ann in self._coco_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in self._coco_data["_annotations_by_image"]:
                    self._coco_data["_annotations_by_image"][img_id] = []
                self._coco_data["_annotations_by_image"][img_id].append(ann)

        return self._coco_data

    def from_coco_annotations(
        self,
        image_filename: Optional[str] = None,
        image_id: Optional[int] = None,
        min_area: float = 0.0,
    ) -> List[str]:
        """
        Get prompts from COCO ground-truth annotations for a specific image.

        Args:
            image_filename: COCO image filename (e.g., "000000397133.jpg")
            image_id: COCO image ID (alternative to filename)
            min_area: Minimum annotation area to include

        Returns:
            List of unique object category names in the image
        """
        coco = self._load_coco_annotations()

        # Resolve image ID
        if image_filename is not None:
            if image_filename not in coco["_image_lookup"]:
                raise ValueError(f"Image not found in COCO: {image_filename}")
            img_id = coco["_image_lookup"][image_filename]
        elif image_id is not None:
            img_id = image_id
        else:
            raise ValueError("Provide either image_filename or image_id")

        # Get annotations for this image
        annotations = coco["_annotations_by_image"].get(img_id, [])

        # Extract unique categories
        categories: Set[str] = set()
        for ann in annotations:
            if ann["area"] >= min_area:
                cat_name = coco["_category_lookup"].get(ann["category_id"])
                if cat_name:
                    categories.add(cat_name)

        return sorted(list(categories))

    def get_coco_image_list(self) -> List[str]:
        """Get list of all image filenames in COCO annotations."""
        coco = self._load_coco_annotations()
        return list(coco["_image_lookup"].keys())

    # -------------------------------------------------------------------------
    # Option 3: GPT-4o Auto-Detection
    # -------------------------------------------------------------------------

    def _get_openai_client(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            self._openai_client = OpenAI(api_key=self._openai_api_key)

        return self._openai_client

    def _pil_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _make_cache_key(self, image_path: Path) -> str:
        """Generate cache key for image."""
        stat = image_path.stat()
        raw = f"{image_path.resolve()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def from_gpt4o(
        self,
        image_path: str,
        use_cache: bool = True,
        scene_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect objects in image using GPT-4o vision.

        Args:
            image_path: Path to the image file
            use_cache: Whether to use cached results
            scene_hint: Optional hint about scene type (e.g., "living room")

        Returns:
            Dict with keys:
                - prompts: List of detected object names
                - suggested_anchor: Suggested anchor object
                - scene_type: Detected scene type
                - confidence: Detection confidence
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Check cache
        cache_key = self._make_cache_key(image_path)
        cache_file = self.cache_dir / f"gpt4o_detection_{cache_key}.json"

        if use_cache and cache_file.exists():
            with open(cache_file, "r") as f:
                cached = json.load(f)
                print(f"  Using cached GPT-4o detection: {cache_file.name}")
                return cached

        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        image_b64 = self._pil_to_base64(image)

        # Build prompt
        scene_context = f"This appears to be a {scene_hint}. " if scene_hint else ""

        prompt = f"""{scene_context}Analyze this image and list all distinct objects you can see.

Return ONLY valid JSON with these keys:
- "objects": List of object names (use simple, common names like "table", "chair", "lamp", "sofa", "painting", "vase", etc.)
- "suggested_anchor": The most prominent/central object that would make a good reference point
- "confidence": A number from 0 to 1 indicating your confidence in the detection
- "scene_type": Brief description of the scene (e.g., "living room", "office", "bedroom")

Guidelines:
- Use singular, lowercase names (e.g., "chair" not "chairs" or "Chair")
- For multiple similar objects, just list the category once (e.g., "chair" not "chair_1, chair_2")
- Include people as "person"
- Be specific but not overly detailed (e.g., "lamp" not "brass floor lamp")
- Only include objects that are clearly visible and distinct
- For art/decorations use: "painting", "poster", "photograph", "mirror"
- For furniture use common names: "sofa", "couch", "armchair", "coffee table", "side table"

Do not include any text outside the JSON."""

        # Call GPT-4o
        print(f"  Detecting objects with GPT-4o...")
        client = self._get_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        content = response.choices[0].message.content
        if content is None:
            result = {
                "prompts": [],
                "suggested_anchor": None,
                "confidence": 0.0,
                "scene_type": "unknown",
                "error": "Model returned no content",
            }
        else:
            try:
                data = json.loads(content.strip())
                result = {
                    "prompts": data.get("objects", []),
                    "suggested_anchor": data.get("suggested_anchor"),
                    "confidence": data.get("confidence", 0.0),
                    "scene_type": data.get("scene_type", "unknown"),
                }
            except json.JSONDecodeError as e:
                result = {
                    "prompts": [],
                    "suggested_anchor": None,
                    "confidence": 0.0,
                    "scene_type": "unknown",
                    "error": f"JSON parse error: {e}",
                }

        # Cache result
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Cached detection result: {cache_file.name}")

        return result

    # -------------------------------------------------------------------------
    # Combined / Smart Detection
    # -------------------------------------------------------------------------

    def auto_detect(
        self,
        image_path: str,
        method: str = "gpt4o",
        fallback_to_vocabulary: bool = True,
        scene_type: str = "indoor",
    ) -> List[str]:
        """
        Auto-detect objects using specified method with fallback.

        Args:
            image_path: Path to image
            method: Detection method - "gpt4o", "coco_gt", or "vocabulary"
            fallback_to_vocabulary: Fall back to vocabulary if detection fails
            scene_type: Scene type for vocabulary fallback

        Returns:
            List of object prompts
        """
        if method == "vocabulary":
            return self.from_coco_vocabulary(scene_type=scene_type)

        if method == "coco_gt":
            # Try to extract COCO image ID from filename
            image_name = Path(image_path).name
            try:
                return self.from_coco_annotations(image_filename=image_name)
            except (ValueError, KeyError) as e:
                if fallback_to_vocabulary:
                    print(f"  COCO lookup failed ({e}), using vocabulary fallback")
                    return self.from_coco_vocabulary(scene_type=scene_type)
                raise

        if method == "gpt4o":
            try:
                result = self.from_gpt4o(image_path)
                if result.get("prompts"):
                    return result["prompts"]
                elif fallback_to_vocabulary:
                    print("  GPT-4o returned no objects, using vocabulary fallback")
                    return self.from_coco_vocabulary(scene_type=scene_type)
            except Exception as e:
                if fallback_to_vocabulary:
                    print(f"  GPT-4o detection failed ({e}), using vocabulary fallback")
                    return self.from_coco_vocabulary(scene_type=scene_type)
                raise

        raise ValueError(f"Unknown detection method: {method}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate object prompts for SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # COCO vocabulary for indoor scenes
    python prompt_sources.py --method vocabulary --scene-type indoor

    # COCO ground-truth for specific image
    python prompt_sources.py --method coco_gt --image 000000397133.jpg \\
        --coco-annotations annotations/instances_val2017.json

    # GPT-4o auto-detection
    python prompt_sources.py --method gpt4o --image photos/living_room.jpg

    # Save prompts to file
    python prompt_sources.py --method gpt4o --image scene.jpg -o prompts.json
""",
    )

    parser.add_argument(
        "--method", "-m",
        choices=["vocabulary", "coco_gt", "gpt4o"],
        default="vocabulary",
        help="Prompt generation method",
    )
    parser.add_argument("--image", "-i", help="Image path (for coco_gt or gpt4o)")
    parser.add_argument("--scene-type", default="indoor",
                        choices=["all", "indoor", "living_room", "kitchen", "bedroom", "outdoor"],
                        help="Scene type for vocabulary method")
    parser.add_argument("--coco-annotations", help="Path to COCO instances JSON")
    parser.add_argument("--output", "-o", help="Output JSON file for prompts")
    parser.add_argument("--cache-dir", default="prompt_cache", help="Cache directory")

    args = parser.parse_args()

    generator = PromptGenerator(
        coco_annotations_path=args.coco_annotations,
        cache_dir=args.cache_dir,
    )

    if args.method == "vocabulary":
        prompts = generator.from_coco_vocabulary(scene_type=args.scene_type)
        print(f"COCO vocabulary ({args.scene_type}): {len(prompts)} prompts")

    elif args.method == "coco_gt":
        if not args.image:
            parser.error("--image required for coco_gt method")
        if not args.coco_annotations:
            parser.error("--coco-annotations required for coco_gt method")
        prompts = generator.from_coco_annotations(image_filename=args.image)
        print(f"COCO ground-truth: {len(prompts)} objects in {args.image}")

    elif args.method == "gpt4o":
        if not args.image:
            parser.error("--image required for gpt4o method")
        result = generator.from_gpt4o(args.image)
        prompts = result["prompts"]
        print(f"GPT-4o detection: {len(prompts)} objects")
        print(f"  Scene type: {result.get('scene_type')}")
        print(f"  Suggested anchor: {result.get('suggested_anchor')}")

    print(f"\nPrompts: {prompts}")

    if args.output:
        output_data = {"prompts": prompts, "method": args.method}
        if args.method == "gpt4o":
            output_data.update(result)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
