
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Dict

from openai import OpenAI


MODEL_NAME = "gpt-5"
OUT_DIR = Path("../../datasets/CUB_200_2011/cub_concept_bank")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = OUT_DIR / "cub_concepts_raw.txt"
CLEANED_PATH = OUT_DIR / "cub_concepts_cleaned.txt"
FINAL_PATH = OUT_DIR / "cub_concepts_final_1000.txt"


PART_SPECS = {
    "head_face": 180,
    "beak_bill": 140,
    "eye_region": 80,
    "neck_throat_breast_belly": 220,
    "back_wing": 180,
    "tail": 100,
    "leg_foot": 50,
    "body_shape_pose_visibility": 120,
}


N_ROUNDS_PER_PART = 2


FINAL_TARGET = 1000


# =========================
# OpenAI 
# =========================
def get_client() -> OpenAI:
    api_key = "KEY"
    return OpenAI(api_key=api_key)


SYSTEM_PROMPT = """
You are an expert in fine-grained bird image understanding and concept bottleneck modeling.

Your job is to generate high-quality visual concepts for bird classification on the CUB dataset.

Important requirements:
- Concepts must be visually observable from bird images.
- Concepts must be fine-grained, detailed, and discriminative.
- Concepts must be useful for image-based bird classification.
- Concepts must be short attribute phrases, not sentences.
- Concepts should be independent and suitable as binary or soft concepts.
- Avoid species names.
- Avoid habitat, geography, sound, behavior, diet, or non-visual biological facts.
- Avoid vague concepts like "bird", "nice feathers", "wing shape", "small bird".
- Avoid duplicates and near-duplicates.
- Prefer part-based and morphology-based concepts.
- Use natural English concept phrases.
- Return only concepts, one per line, with no numbering and no explanation.
""".strip()


def build_part_prompt(part_name: str, n_concepts: int) -> str:
    focus_map = {
        "head_face": """
Focus on:
- crown
- forehead
- nape
- cheek
- throat
- face markings
- eyebrow stripe / supercilium
- malar stripe
- head patch
- local color contrast
- local streaking / spotting / barring
""",
        "beak_bill": """
Focus on:
- bill length
- bill thickness
- bill curvature
- bill tip shape
- upper bill shape
- lower bill shape
- bill base shape
- bill color
- bill contrast
- bill markings
- bill visibility
""",
        "eye_region": """
Focus on:
- eye size
- eye color
- eye ring
- orbital markings
- eye stripe
- contrast around the eye
- local visibility states
""",
        "neck_throat_breast_belly": """
Focus on:
- throat
- neck
- upper breast
- lower breast
- belly
- flank
- chest markings
- color transitions
- local spots
- streaks
- bars
- patch shapes
""",
        "back_wing": """
Focus on:
- back color
- scapular region
- mantle
- wing coverts
- primary feathers
- secondary feathers
- wing bars
- wing patches
- feather contrast
- local wing patterns
- folded wing visibility
""",
        "tail": """
Focus on:
- tail length
- tail shape
- tail tip shape
- forked / rounded / squared tail
- tail band
- tail edge color
- tail contrast
- tail markings
- tail visibility
""",
        "leg_foot": """
Focus on:
- leg length
- leg thickness
- tarsus appearance
- leg color
- foot color
- claw visibility
- foot posture
""",
        "body_shape_pose_visibility": """
Focus on:
- overall body slenderness
- compactness
- posture
- pose
- side view / frontal view / rear-biased view
- perched pose
- wing folded state
- head turned state
- partial occlusion
- part visibility
- silhouette-like traits
""",
    }

    if part_name not in focus_map:
        raise ValueError(f"Unknown part_name: {part_name}")

    return f"""
Generate exactly {n_concepts} high-quality visual bird attributes for the CUB dataset.

{focus_map[part_name]}

Additional rules:
- Make the concepts detailed and specific.
- Prefer attributes that distinguish visually similar bird species.
- Include color, pattern, shape, texture, local contrast, and visibility traits when relevant.
- Keep each concept short.
- One concept per line.
- No numbering.
- No explanation.
""".strip()


def build_refine_prompt(concepts: List[str], target_n: int) -> str:
    joined = "\n".join(concepts)
    return f"""
Below is a large candidate concept list for fine-grained bird classification on the CUB dataset.

Your task:
1. Remove exact duplicates.
2. Remove near-duplicates with essentially the same visual meaning.
3. Remove vague, broad, or low-quality concepts.
4. Keep only visually observable, fine-grained, discriminative bird attributes.
5. Preserve diversity across bird parts and appearance patterns.
6. Return the best {target_n} concepts.

Return only the cleaned concepts, one per line, with no numbering and no explanation.

Candidate concepts:
{joined}
""".strip()


def call_model(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=system_prompt,
        input=user_prompt,
    )
    return response.output_text.strip()


def parse_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        item = line.strip()
        if not item:
            continue


        item = re.sub(r"^\d+[\).\-\s]+", "", item).strip()
        item = re.sub(r"^[\-\*\•\s]+", "", item).strip()


        item = item.lower()
        
        item = item.rstrip(" .;:")

        if item:
            lines.append(item)
    return lines


def normalize_concept(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    s = s.strip(" .;:,")
    return s


def local_dedup(concepts: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in concepts:
        key = normalize_concept(c)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def save_list(path: Path, items: List[str]) -> None:
    path.write_text("\n".join(items), encoding="utf-8")



def generate_raw_concepts(client: OpenAI) -> List[str]:
    raw_concepts: List[str] = []

    for part_name, quota in PART_SPECS.items():
        per_round = quota // N_ROUNDS_PER_PART
        remainder = quota % N_ROUNDS_PER_PART

        for r in range(N_ROUNDS_PER_PART):
            n = per_round + (1 if r < remainder else 0)
            prompt = build_part_prompt(part_name, n)
            print(f"[generate] part={part_name} round={r+1}/{N_ROUNDS_PER_PART} n={n}")

            text = call_model(client, SYSTEM_PROMPT, prompt)
            concepts = parse_lines(text)

            raw_concepts.extend(concepts)
            time.sleep(0.5)

    return raw_concepts


def refine_concepts_with_llm(client: OpenAI, concepts: List[str], target_n: int) -> List[str]:
    prompt = build_refine_prompt(concepts, target_n)
    print(f"[refine] input={len(concepts)} target={target_n}")

    text = call_model(client, SYSTEM_PROMPT, prompt)
    refined = parse_lines(text)
    return refined


def main():
    client = get_client()


    raw_concepts = generate_raw_concepts(client)
    raw_concepts = local_dedup(raw_concepts)
    save_list(RAW_PATH, raw_concepts)
    print(f"[saved] raw unique concepts: {len(raw_concepts)} -> {RAW_PATH}")


    refined = refine_concepts_with_llm(client, raw_concepts, target_n=1100)
    refined = local_dedup(refined)
    save_list(CLEANED_PATH, refined)
    print(f"[saved] cleaned concepts: {len(refined)} -> {CLEANED_PATH}")


    final_concepts = refined[:FINAL_TARGET]
    save_list(FINAL_PATH, final_concepts)
    print(f"[saved] final concepts: {len(final_concepts)} -> {FINAL_PATH}")

    print("\n[preview: first 30 concepts]")
    for i, c in enumerate(final_concepts[:30], 1):
        print(f"{i:02d}. {c}")


if __name__ == "__main__":
    main()