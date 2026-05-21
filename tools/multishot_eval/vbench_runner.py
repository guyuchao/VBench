from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


import cv2

from manifest import ShotSpec


TEXT_ALIGNMENT_METRICS = {"overall_consistency", "clip_score"}


class VBenchMetricRunner:
    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda",
        load_ckpt_from_local: bool = False,
        read_frame: bool = False,
        keep_meta: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.meta_dir = output_dir / "_vbench_meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        import torch

        self.device = torch.device(device)
        self.load_ckpt_from_local = load_ckpt_from_local
        self.read_frame = read_frame
        self.keep_meta = keep_meta
        self._submodules: dict[str, Any] = {}
        if not self.keep_meta:
            self._cleanup_stale_meta()

    def run_text_alignment(self, video_id: str, shots: list[ShotSpec], metric: str) -> dict[str, Any]:
        if metric not in TEXT_ALIGNMENT_METRICS:
            raise ValueError(f"Unsupported text alignment metric: {metric}")
        shots_with_caption = [shot for shot in shots if shot.caption]
        if not shots_with_caption:
            return {"status": "skipped", "reason": "No shot captions found."}
        dimension = "overall_consistency" if metric == "overall_consistency" else "clip_score"
        return self.run_dimension(video_id, shots_with_caption, dimension, prompts_required=True)

    def run_intra_quality(self, video_id: str, shots: list[ShotSpec], dimensions: list[str]) -> dict[str, Any]:
        results = {}
        for dimension in dimensions:
            results[dimension] = self.run_dimension(video_id, shots, dimension, prompts_required=False)
        return results

    def run_character_subject_consistency(
        self,
        video_id: str,
        shots: list[ShotSpec],
        characters: dict[str, Any],
        frame_strategy: str = "middle",
    ) -> dict[str, Any]:
        character_videos_dir = self.output_dir / "_character_videos" / video_id
        character_videos_dir.mkdir(parents=True, exist_ok=True)
        shots_by_id = {shot.id: shot for shot in shots}

        results = {}
        for character, info in sorted(characters.items()):
            appears_in = info.get("appears_in", []) if isinstance(info, dict) else info
            character_shots = [shots_by_id[shot_id] for shot_id in appears_in if shot_id in shots_by_id]
            if len(character_shots) < 2:
                results[character] = {
                    "status": "skipped",
                    "reason": "Character appears in fewer than two available shots.",
                    "appears_in": [shot.id for shot in character_shots],
                }
                continue

            character_video_path = character_videos_dir / f"{character}.mp4"
            build_character_video(character_shots, character_video_path, frame_strategy=frame_strategy)
            synthetic_shot = ShotSpec(
                id=0,
                file=character_video_path,
                caption=info.get("description_hint") if isinstance(info, dict) else character,
                characters=[character],
            )
            metric = self.run_dimension(
                video_id=f"{video_id}_{character}",
                shots=[synthetic_shot],
                dimension="subject_consistency",
                prompts_required=False,
            )
            results[character] = {
                "average": metric["average"],
                "appears_in": [shot.id for shot in character_shots],
                "character_video": str(character_video_path),
                "description_hint": info.get("description_hint") if isinstance(info, dict) else None,
            }

        valid_scores = [
            value["average"]
            for value in results.values()
            if isinstance(value, dict) and "average" in value
        ]
        return {
            "average": sum(valid_scores) / len(valid_scores) if valid_scores else None,
            "characters": results,
            "frame_strategy": frame_strategy,
            "vbench_dimension": "subject_consistency",
        }

    def run_dimension(
        self,
        video_id: str,
        shots: list[ShotSpec],
        dimension: str,
        prompts_required: bool,
    ) -> dict[str, Any]:
        json_path = self._write_full_info(video_id, shots, dimension, prompts_required)
        module_name = "competitions.clip_score" if dimension == "clip_score" else f"vbench.{dimension}"
        module = importlib.import_module(module_name)
        compute = getattr(module, f"compute_{dimension}")

        submodules = [] if dimension == "clip_score" else self._get_submodules(dimension)
        try:
            score, details = compute(str(json_path), self.device, submodules)
            result = {
                "average": _to_jsonable(score),
                "per_shot": _details_to_per_shot(details),
                "vbench_dimension": dimension,
            }
            if self.keep_meta:
                result["full_info_json"] = str(json_path)
            return result
        finally:
            pass

    def _get_submodules(self, dimension: str) -> Any:
        if dimension not in self._submodules:
            from vbench.utils import init_submodules

            self._submodules.update(
                init_submodules(
                    [dimension],
                    local=self.load_ckpt_from_local,
                    read_frame=self.read_frame,
                )
            )
        return self._submodules[dimension]

    def _write_full_info(
        self,
        video_id: str,
        shots: list[ShotSpec],
        dimension: str,
        prompts_required: bool,
    ) -> Path:
        items = []
        for shot in shots:
            prompt = shot.caption or f"{video_id}_shot_{shot.id}"
            if prompts_required and not shot.caption:
                continue
            items.append(
                {
                    "prompt_en": prompt,
                    "dimension": [dimension],
                    "video_list": [str(shot.file)],
                    "shot_id": shot.id,
                }
            )
        path = self.meta_dir / f"{video_id}_{dimension}_full_info.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        return path

    def _cleanup_stale_meta(self) -> None:
        pass


def _details_to_per_shot(details: list[dict[str, Any]]) -> dict[str, Any]:
    per_shot = {}
    for item in details:
        video_path = Path(item["video_path"])
        per_shot[video_path.stem] = _to_jsonable(item.get("video_results"))
    return per_shot


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def build_character_video(shots: list[ShotSpec], output_path: Path, frame_strategy: str = "middle") -> None:
    frames = []
    for shot in shots:
        frame = read_representative_frame(shot.file, frame_strategy=frame_strategy)
        if frame is None:
            raise ValueError(f"Could not read representative frame from {shot.file}")
        frames.append(frame)

    if not frames:
        raise ValueError("No frames available for character video.")

    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create character video: {output_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def read_representative_frame(video_path: Path, frame_strategy: str = "middle") -> Any:
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None
        if frame_strategy == "first":
            target = 0
        elif frame_strategy == "last":
            target = max(0, frame_count - 1)
        elif frame_strategy == "middle":
            target = frame_count // 2
        else:
            raise ValueError(f"Unsupported frame strategy: {frame_strategy}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
