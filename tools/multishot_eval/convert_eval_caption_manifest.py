from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHARACTER_PATTERN = re.compile(r"\[character(?P<idx>\d+)\]", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert eval_caption_multishot_t2v JSON into the multi-shot evaluation manifest format."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input eval_caption_multishot_t2v JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest JSON.")
    parser.add_argument("--result_root", default="result", help="Root containing generated video folders.")
    parser.add_argument(
        "--video_id_template",
        default="video{one_based}",
        help=(
            "Template for video folder ids. Supported fields: {zero_based}, {one_based}, "
            "{generation_index}."
        ),
    )
    parser.add_argument("--shot_file_template", default="shot_{shot_id}.mp4")
    parser.add_argument("--full_video_name", default="full.mp4")
    parser.add_argument(
        "--latent_fps",
        type=float,
        default=None,
        help="Optional FPS for converting switch_latent_frames to target_boundaries_sec.",
    )
    parser.add_argument(
        "--strip_shot_cut_token",
        action="store_true",
        help="Remove the leading '[shot cut]' token from shot captions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    manifest: dict[str, Any] = {}
    for zero_based, item in enumerate(data):
        video_id = build_video_id(item, zero_based, args.video_id_template)
        video_dir = f"{args.result_root}/{video_id}"
        prompts = item.get("prompts") or []

        shots = []
        character_occurrences: dict[str, list[int]] = {}
        for shot_idx, prompt in enumerate(prompts, start=1):
            caption = normalise_caption(prompt, strip_shot_cut_token=args.strip_shot_cut_token)
            characters = extract_characters(prompt)
            for character in characters:
                character_occurrences.setdefault(character, []).append(shot_idx)
            shots.append(
                {
                    "id": shot_idx,
                    "file": args.shot_file_template.format(shot_id=shot_idx),
                    "caption": caption,
                    "characters": characters,
                }
            )

        video_entry: dict[str, Any] = {
            "dir": video_dir,
            "full_video": f"{video_dir}/{args.full_video_name}",
            "global_caption": item.get("random_concept_summary"),
            "shots": shots,
            "characters": {
                name: {
                    "appears_in": appears_in,
                    "description_hint": extract_character_description(prompts, name),
                }
                for name, appears_in in sorted(character_occurrences.items())
            },
            "source": {
                "generation_index": item.get("_meta", {}).get("generation_index"),
                "latent_total_frames": item.get("latent_total_frames"),
                "switch_latent_frames": item.get("switch_latent_frames", []),
                "meta": item.get("_meta", {}),
            },
            "target_boundaries_frames": latent_to_video_frames(item.get("switch_latent_frames", [])),
        }

        if args.latent_fps:
            video_entry["target_boundaries_sec"] = [
                round(float(frame) / args.latent_fps, 6)
                for frame in video_entry["target_boundaries_frames"]
            ]

        manifest[video_id] = video_entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest with {len(manifest)} videos to {args.output}")


def build_video_id(item: dict[str, Any], zero_based: int, template: str) -> str:
    generation_index = item.get("_meta", {}).get("generation_index", zero_based)
    return template.format(
        zero_based=zero_based,
        one_based=zero_based + 1,
        generation_index=generation_index,
    )


def latent_to_video_frames(latent_frames: list[Any]) -> list[int]:
    return [(int(frame) - 1) * 4 + 1 for frame in latent_frames]


def normalise_caption(prompt: str, strip_shot_cut_token: bool) -> str:
    caption = prompt.strip()
    if strip_shot_cut_token:
        caption = re.sub(r"^\s*\[shot\s+cut\]\s*", "", caption, flags=re.IGNORECASE)
    return caption


def extract_characters(prompt: str) -> list[str]:
    characters = []
    for match in CHARACTER_PATTERN.finditer(prompt):
        name = f"character{int(match.group('idx'))}"
        if name not in characters:
            characters.append(name)
    return characters


def extract_character_description(prompts: list[str], character: str) -> str | None:
    token = f"[{character}]"
    for prompt in prompts:
        start = prompt.lower().find(token.lower())
        if start < 0:
            continue
        snippet = prompt[start + len(token):].lstrip(" ,")
        snippet = snippet.split(", positioned", 1)[0]
        snippet = snippet.split(", a ", 1)[0] if len(snippet) > 260 else snippet
        snippet = snippet.strip(" ,")
        return snippet or None
    return None


if __name__ == "__main__":
    main()
