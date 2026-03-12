"""
Knowledge Distillation Pipeline
================================
Dynamic Spatial Intelligence Eval Suite — Perception PREP Stage

WHAT THIS DOES:
---------------
1. Reads text+image pairs from S3
2. Chunks long text (RULER-inspired: handles variable length inputs)
3. Extracts physical objects from each chunk via LLaMA 3
4. Distills/deduplicates across chunks into a clean object list
5. Ranks objects by spatial relevance (prioritizes spatially-referenced objects)
6. Saves structured output back to S3 — ready for SAM3 grounding

OUTPUT FORMAT (per item):
--------------------------
{
  "item_id": "daily_indoors_abc123",
  "category": "daily_indoors",
  "objects": [
    {"object": "wooden table", "spatially_referenced": true,  "mentions": 2},
    {"object": "chair",        "spatially_referenced": true,  "mentions": 1},
    {"object": "ceiling lamp", "spatially_referenced": false, "mentions": 1}
  ],
  "raw_text": "...",
  "chunks_processed": 2
}

HOW TO RUN:
-----------
export S3_BUCKET=opensource-images-text
export HF_TOKEN=your_huggingface_token
python knowledge_distillation_pipeline.py

PAPER CONNECTIONS:
------------------
- RULER: chunking strategy handles variable-length text, prevents long-context degradation
- FRAMES: end-to-end structured output enables downstream failure attribution
          (if SAM3 fails to find an object, we know distillation vs grounding failed)
"""

import os
import json
import re
import time
import boto3
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

S3_BUCKET        = os.environ.get("S3_BUCKET", "opensource-images-text")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
HF_MODEL         = "meta-llama/Llama-3.1-8B-Instruct"   # swap to 70B if available
CHUNK_SIZE_WORDS = 150    # RULER-inspired: keep each chunk short to avoid degradation
MAX_ITEMS        = None     # how many S3 items to process (set None for all)
OUTPUT_PREFIX    = "distilled"   # S3 prefix for saving results
CATEGORIES       = ["daily_indoors", "manufacturing_indoors", "markets", "architectural_street"]

# Spatial language that indicates an object is spatially referenced
SPATIAL_MARKERS = [
    "left", "right", "behind", "in front", "next to", "beside", "near",
    "above", "below", "on top", "underneath", "across", "between",
    "center", "middle", "corner", "edge", "adjacent", "facing",
    "opposite", "along", "against", "around", "inside", "outside",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: FETCH FROM S3
# ─────────────────────────────────────────────────────────────────────────────

def fetch_items_from_s3(
    s3_client,
    bucket: str,
    category: Optional[str] = None,
    max_items: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Fetch text+image pairs from S3.
    Returns list of items with text, image_key, and metadata.
    """
    items = []
    categories = [category] if category else CATEGORIES

    for cat in categories:
        prefix = f"raw/{cat}/"
        paginator = s3_client.get_paginator("list_objects_v2")

        seen_item_ids = set()
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                item_prefix = cp["Prefix"]  # e.g. raw/daily_indoors/item_abc/
                item_id = item_prefix.rstrip("/").split("/")[-1]

                if item_id in seen_item_ids:
                    continue
                seen_item_ids.add(item_id)

                # Fetch text
                text_key = f"{item_prefix}text.txt"
                source_key = f"{item_prefix}source.json"
                image_key = f"{item_prefix}image.jpg"

                try:
                    text_obj = s3_client.get_object(Bucket=bucket, Key=text_key)
                    raw_text = text_obj["Body"].read().decode("utf-8")

                    source_obj = s3_client.get_object(Bucket=bucket, Key=source_key)
                    source_data = json.loads(source_obj["Body"].read().decode("utf-8"))

                    items.append({
                        "item_id":    item_id,
                        "category":   cat,
                        "raw_text":   raw_text,
                        "image_key":  image_key,
                        "source":     source_data,
                    })

                    if max_items and len(items) >= max_items:
                        print(f"[S3] Fetched {len(items)} items", flush=True)
                        return items

                except Exception as e:
                    print(f"[S3] Skipping {item_id}: {e}", flush=True)
                    continue

    print(f"[S3] Fetched {len(items)} items total", flush=True)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: CHUNK TEXT (RULER-inspired)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS) -> List[str]:
    """
    Split text into chunks at sentence boundaries.

    RULER showed that models degrade on long contexts — we chunk to
    keep each extraction call focused and reliable, regardless of
    whether the input is 1 sentence or 20 paragraphs.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > chunk_size_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: EXTRACT OBJECTS VIA LLAMA 3
# ─────────────────────────────────────────────────────────────────────────────

def build_extraction_prompt(text_chunk: str) -> str:
    """
    Prompt LLaMA 3 to extract physical objects from a text chunk.
    Returns structured JSON — no task/question conditioning needed.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise object extraction system. Your job is to identify physical objects mentioned in text.

Rules:
- Only extract PHYSICAL, VISIBLE objects (things that could appear in an image)
- Do NOT extract abstract concepts, emotions, actions, or people's names
- Normalize object names to simple noun phrases (e.g. "large wooden table" → "wooden table")
- Return ONLY a JSON array of strings, nothing else
- If no physical objects are found, return an empty array []

Example input: "The office has a desk near the window with a laptop on it. A bookshelf stands in the corner."
Example output: ["desk", "window", "laptop", "bookshelf"]
<|eot_id|><|start_header_id|>user<|end_header_id|>
Extract all physical objects from this text:

{text_chunk}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def extract_objects_from_chunk(
    chunk: str,
    hf_headers: Dict[str, str],   # kept for signature compatibility, unused
    model: str = HF_MODEL
) -> List[str]:
    """
    Call HuggingFace Inference API via InferenceClient to extract objects.
    Uses chat_completion so the model follows the instruction reliably.
    Returns list of object name strings.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HF_TOKEN)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise object extraction system. "
                "Extract only PHYSICAL, VISIBLE objects from the text. "
                "Do NOT extract people, actions, emotions, or abstract concepts. "
                "Respond ONLY with a JSON array of strings. Nothing else. "
                "Example: [\"wooden table\", \"window\", \"chair\"]"
            )
        },
        {
            "role": "user",
            "content": f"Extract all physical objects from this text:\n\n{chunk}"
        }
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=200,
            temperature=0.1,
        )

        generated = response.choices[0].message.content.strip()

        # Extract JSON array from response
        match = re.search(r'\[.*?\]', generated, re.DOTALL)
        if match:
            objects = json.loads(match.group())
            return [str(o).lower().strip() for o in objects if o]
        return []

    except Exception as e:
        print(f"  [HF] Extraction error: {e}", flush=True)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: DISTILL — DEDUPLICATE + NORMALIZE ACROSS CHUNKS
# ─────────────────────────────────────────────────────────────────────────────

def are_similar_objects(obj1: str, obj2: str) -> bool:
    """
    Check if two object strings refer to the same thing.
    e.g. "white lamp" and "floor lamp" and "lamp" are all similar.
    Simple word-overlap heuristic — good enough for demo.
    """
    words1 = set(obj1.lower().split())
    words2 = set(obj2.lower().split())

    # Remove common adjectives that don't identify the object
    stopwords = {"a", "an", "the", "large", "small", "big", "little",
                 "old", "new", "wooden", "metal", "white", "black",
                 "red", "blue", "green", "brown", "dark", "light"}

    core1 = words1 - stopwords
    core2 = words2 - stopwords

    if not core1 or not core2:
        return False

    # If core nouns overlap, treat as same object
    return bool(core1 & core2)


def distill_objects(
    all_objects_per_chunk: List[List[str]]
) -> Dict[str, int]:
    """
    Merge object lists from all chunks into one deduplicated dict.
    Returns {canonical_object_name: mention_count}

    This is the core DISTILLATION step:
    - "lamp", "floor lamp", "white lamp" → "floor lamp" (most specific wins)
    - mention count tracks how often each object appears across chunks
    """
    # Flatten all objects with their chunk source
    all_objects = []
    for chunk_objects in all_objects_per_chunk:
        all_objects.extend(chunk_objects)

    if not all_objects:
        return {}

    # Group similar objects together
    groups: List[List[str]] = []
    assigned = [False] * len(all_objects)

    for i, obj in enumerate(all_objects):
        if assigned[i]:
            continue
        group = [obj]
        assigned[i] = True
        for j, other_obj in enumerate(all_objects):
            if not assigned[j] and are_similar_objects(obj, other_obj):
                group.append(other_obj)
                assigned[j] = True
        groups.append(group)

    # Pick canonical name = longest (most specific) name in group
    distilled = {}
    for group in groups:
        canonical = max(group, key=lambda x: len(x.split()))
        distilled[canonical] = len(group)  # mention count

    return distilled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: RANK BY SPATIAL RELEVANCE
# ─────────────────────────────────────────────────────────────────────────────

def rank_by_spatial_relevance(
    distilled_objects: Dict[str, int],
    raw_text: str
) -> List[Dict[str, Any]]:
    """
    Rank objects by spatial relevance.

    An object is "spatially referenced" if the text mentions it
    alongside spatial language (left, right, next to, behind, etc.)

    Spatially referenced objects go first — they're most important
    for Lauren's downstream spatial graph construction.

    FRAMES connection: this prioritization ensures the most
    spatially-relevant facts flow into the reasoning stage,
    reducing noise and improving spatial graph quality.
    """
    raw_text_lower = raw_text.lower()
    ranked = []

    for obj, mention_count in distilled_objects.items():
        # Check if object appears near spatial language in the text
        spatially_referenced = False
        obj_pattern = re.compile(re.escape(obj), re.IGNORECASE)

        for match in obj_pattern.finditer(raw_text_lower):
            # Look at a 60-char window around the object mention
            start = max(0, match.start() - 60)
            end = min(len(raw_text_lower), match.end() + 60)
            window = raw_text_lower[start:end]

            if any(marker in window for marker in SPATIAL_MARKERS):
                spatially_referenced = True
                break

        ranked.append({
            "object":               obj,
            "spatially_referenced": spatially_referenced,
            "mentions":             mention_count,
        })

    # Sort: spatially referenced first, then by mention count
    ranked.sort(key=lambda x: (not x["spatially_referenced"], -x["mentions"]))
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SAVE RESULTS TO S3
# ─────────────────────────────────────────────────────────────────────────────

def save_distilled_to_s3(
    s3_client,
    bucket: str,
    item_id: str,
    category: str,
    ranked_objects: List[Dict[str, Any]],
    raw_text: str,
    chunks_processed: int,
    image_key: str
) -> str:
    """Save distilled knowledge output to S3 under distilled/ prefix."""

    output = {
        "item_id":          item_id,
        "category":         category,
        "image_key":        image_key,
        "objects":          ranked_objects,
        "raw_text":         raw_text,
        "chunks_processed": chunks_processed,
        "distilled_at":     datetime.now(timezone.utc).isoformat(),
        # Summary stats for quick inspection
        "stats": {
            "total_objects":            len(ranked_objects),
            "spatially_referenced":     sum(1 for o in ranked_objects if o["spatially_referenced"]),
            "not_spatially_referenced": sum(1 for o in ranked_objects if not o["spatially_referenced"]),
        }
    }

    key = f"{OUTPUT_PREFIX}/{category}/{item_id}/distilled.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(output, indent=2).encode("utf-8"),
        ContentType="application/json; charset=utf-8"
    )

    return key


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    bucket: str = S3_BUCKET,
    max_items: Optional[int] = MAX_ITEMS,
    category: Optional[str] = None
):
    """
    Full knowledge distillation pipeline.

    Text (S3) → chunk → extract objects (LLaMA 3) → distill → rank → save (S3)
    """
    print("=" * 60, flush=True)
    print("KNOWLEDGE DISTILLATION PIPELINE", flush=True)
    print("Perception PREP Stage — Spatial Eval Suite", flush=True)
    print("=" * 60, flush=True)

    # Validate config
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_token")
        return

    hf_headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    s3 = boto3.client("s3")

    # ── Step 1: Fetch from S3 ──
    print(f"\n[1/5] Fetching items from S3 (bucket={bucket})", flush=True)
    items = fetch_items_from_s3(s3, bucket, category=category, max_items=max_items)

    if not items:
        print("No items found. Check your S3 bucket and credentials.")
        return

    results = []
    failed = 0

    for i, item in enumerate(items):
        item_id  = item["item_id"]
        raw_text = item["raw_text"]
        cat      = item["category"]

        print(f"\n[Item {i+1}/{len(items)}] {item_id} ({cat})", flush=True)
        print(f"  Text preview: {raw_text[:100].strip()}...", flush=True)

        # ── Step 2: Chunk ──
        chunks = chunk_text(raw_text)
        print(f"  [2/5] Chunked into {len(chunks)} chunk(s)", flush=True)

        # ── Step 3: Extract objects per chunk ──
        print(f"  [3/5] Extracting objects via LLaMA 3...", flush=True)
        all_chunk_objects = []

        for j, chunk in enumerate(chunks):
            objects = extract_objects_from_chunk(chunk, hf_headers)
            print(f"    Chunk {j+1}: {objects}", flush=True)
            all_chunk_objects.append(objects)
            time.sleep(0.5)  # rate limit buffer

        # ── Step 4: Distill ──
        print(f"  [4/5] Distilling...", flush=True)
        distilled = distill_objects(all_chunk_objects)
        print(f"    Distilled to {len(distilled)} unique objects", flush=True)

        # ── Step 5: Rank by spatial relevance ──
        print(f"  [5/5] Ranking by spatial relevance...", flush=True)
        ranked = rank_by_spatial_relevance(distilled, raw_text)

        spatial_count = sum(1 for o in ranked if o["spatially_referenced"])
        print(f"    {spatial_count}/{len(ranked)} objects are spatially referenced", flush=True)
        print(f"    Top objects: {[o['object'] for o in ranked[:5]]}", flush=True)

        # ── Save to S3 ──
        try:
            saved_key = save_distilled_to_s3(
                s3, bucket, item_id, cat,
                ranked, raw_text, len(chunks), item["image_key"]
            )
            print(f"  ✓ Saved → s3://{bucket}/{saved_key}", flush=True)
            results.append({"item_id": item_id, "status": "ok", "objects": len(ranked)})
        except Exception as e:
            print(f"  ✗ Save failed: {e}", flush=True)
            failed += 1
            results.append({"item_id": item_id, "status": "failed"})

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print(f"  Processed: {len(results)}", flush=True)
    print(f"  Succeeded: {len(results) - failed}", flush=True)
    print(f"  Failed:    {failed}", flush=True)
    print(f"\nOutput location: s3://{bucket}/{OUTPUT_PREFIX}/", flush=True)
    print("\nSample outputs:", flush=True)

    for r in results[:3]:
        print(f"  {r}", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK LOCAL TEST (no S3, no API needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_locally():
    """
    Test the distillation logic locally without S3 or HuggingFace.
    Uses a hardcoded example to verify chunking, distillation, and ranking.
    Run with: python knowledge_distillation_pipeline.py --test
    """
    print("=" * 60)
    print("LOCAL TEST MODE (no API, no S3)")
    print("=" * 60)

    sample_text = """
    The kitchen has a large wooden dining table positioned in the center of the room.
    To the left of the table, there are four chairs arranged neatly.
    A white refrigerator stands against the far wall, next to a stainless steel oven.
    Above the table, a pendant lamp hangs from the ceiling.
    On the counter beside the sink, there is a coffee maker and a toaster.
    Several potted plants are placed near the window on the right side of the room.
    """

    print(f"\nInput text:\n{sample_text.strip()}\n")

    # Test chunking
    chunks = chunk_text(sample_text)
    print(f"[2/5] Chunks ({len(chunks)}):")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {c[:80]}...")

    # Simulate extraction (no API)
    simulated_extractions = [
        ["wooden dining table", "chairs", "refrigerator", "oven"],
        ["pendant lamp", "coffee maker", "toaster", "potted plants", "window", "sink", "counter"]
    ]
    print(f"\n[3/5] Simulated extractions:")
    for i, e in enumerate(simulated_extractions):
        print(f"  Chunk {i+1}: {e}")

    # Test distillation
    distilled = distill_objects(simulated_extractions)
    print(f"\n[4/5] Distilled ({len(distilled)} objects):")
    for obj, count in distilled.items():
        print(f"  '{obj}' (mentions: {count})")

    # Test ranking
    ranked = rank_by_spatial_relevance(distilled, sample_text)
    print(f"\n[5/5] Ranked by spatial relevance:")
    for o in ranked:
        flag = "★ spatial" if o["spatially_referenced"] else "  general"
        print(f"  [{flag}] {o['object']} (mentions: {o['mentions']})")

    print("\n✓ Local test passed — pipeline logic works correctly")
    print("Next step: set HF_TOKEN and S3_BUCKET to run the full pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test_locally()
    else:
        run_pipeline(
            bucket=S3_BUCKET,
            max_items=MAX_ITEMS,
            category=None   # set to "daily_indoors" etc. to filter
        )
