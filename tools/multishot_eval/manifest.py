from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class ShotSpec:
    id: int
    file: Path
    caption: str | None = None
    characters: list[str] = field(default_factory=list)


@dataclass
class VideoSpec:
    id: str
    root: Path
    full_video: Path | None
    shots: list[ShotSpec]
    target_boundaries_frames: list[int] = field(default_factory=list)
    target_boundaries_sec: list[float] = field(default_factory=list)
    global_caption: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def load_manifest(manifest_path: Path | None, result_root: Path) -> list[VideoSpec]:
    result_root = result_root.resolve()
    if manifest_path is None:
        return discover_videos(result_root)

    manifest_path = manifest_path.resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    videos_data = _normalise_video_items(data)
    videos = []
    for video_id, video_data in videos_data:
        videos.append(_parse_video_spec(video_id, video_data, result_root, manifest_path.parent))
    return videos


def discover_videos(result_root: Path) -> list[VideoSpec]:
    videos = []
    for video_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        shots = _discover_shots(video_dir, manifest_dir=result_root, result_root=result_root)
        if not shots:
            continue
        full_video = _find_full_video(video_dir)
        videos.append(
            VideoSpec(
                id=video_dir.name,
                root=video_dir,
                full_video=full_video,
                shots=shots,
                target_boundaries_frames=[],
                target_boundaries_sec=_derive_boundaries_from_shots(shots),
            )
        )
    return videos


def _normalise_video_items(data: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict) and "videos" in data:
        videos = data["videos"]
        if isinstance(videos, dict):
            return [(str(k), v) for k, v in videos.items()]
        return [(str(v.get("id", v.get("name", idx))), v) for idx, v in enumerate(videos)]

    if isinstance(data, dict):
        return [(str(k), v) for k, v in data.items()]

    if isinstance(data, list):
        return [(str(v.get("id", v.get("name", idx))), v) for idx, v in enumerate(data)]

    raise ValueError("Manifest must be a dict, a list, or contain a 'videos' field.")


def _parse_video_spec(
    video_id: str,
    video_data: dict[str, Any],
    result_root: Path,
    manifest_dir: Path,
) -> VideoSpec:
    root_value = (
        video_data.get("dir")
        or video_data.get("root")
        or video_data.get("path")
        or video_data.get("video_dir")
        or video_id
    )
    root = _resolve_path(root_value, result_root, manifest_dir, result_root).resolve()
    if not root.exists():
        fallback = (result_root / video_id).resolve()
        if fallback.exists():
            root = fallback

    full_value = video_data.get("full_video") or video_data.get("full") or video_data.get("full_mp4")
    full_video = _resolve_path(full_value, root, manifest_dir, result_root).resolve() if full_value else _find_full_video(root)
    if full_video is not None and not full_video.exists():
        full_video = _find_full_video(root)

    shots = _parse_shots(video_data.get("shots"), root, result_root, manifest_dir)
    if not shots:
        shots = _discover_shots(root, manifest_dir, result_root)

    target_boundaries_sec = [
        float(x)
        for x in (
            video_data.get("target_boundaries_sec")
            or video_data.get("target_boundaries")
            or video_data.get("boundaries_sec")
            or []
        )
    ]
    if not target_boundaries_sec:
        target_boundaries_sec = _derive_boundaries_from_shots(shots)

    target_boundaries_frames = [
        int(x)
        for x in (
            video_data.get("target_boundaries_frames")
            or []
        )
    ]

    return VideoSpec(
        id=video_id,
        root=root,
        full_video=full_video,
        shots=shots,
        target_boundaries_frames=target_boundaries_frames,
        target_boundaries_sec=target_boundaries_sec,
        global_caption=video_data.get("global_caption") or video_data.get("caption"),
        raw=video_data,
    )


def _parse_shots(
    shots_data: Any,
    video_root: Path,
    result_root: Path,
    manifest_dir: Path,
) -> list[ShotSpec]:
    if not shots_data:
        return []

    if isinstance(shots_data, dict):
        items = []
        for key, value in shots_data.items():
            if isinstance(value, dict):
                value = {"id": value.get("id", key), **value}
            else:
                value = {"id": key, "file": value}
            items.append(value)
    else:
        items = list(shots_data)

    shots = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            item = {"id": idx, "file": item}

        shot_id = _parse_shot_id(item.get("id", idx))
        file_value = item.get("file") or item.get("path") or f"shot_{shot_id}.mp4"
        file_path = _resolve_path(file_value, video_root, manifest_dir, result_root).resolve()
        shots.append(
            ShotSpec(
                id=shot_id,
                file=file_path,
                caption=item.get("caption") or item.get("prompt"),
                characters=list(item.get("characters") or []),
            )
        )

    return sorted(shots, key=lambda shot: shot.id)


def _discover_shots(video_root: Path, manifest_dir: Path, result_root: Path) -> list[ShotSpec]:
    if not video_root.exists():
        return []
    candidates = [
        p
        for p in video_root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and _looks_like_shot(p.name)
    ]
    shots = []
    for path in candidates:
        shots.append(ShotSpec(id=_parse_shot_id(path.stem), file=path.resolve()))
    return sorted(shots, key=lambda shot: shot.id)


def _find_full_video(video_root: Path) -> Path | None:
    if not video_root.exists():
        return None
    for path in video_root.iterdir():
        if path.is_file() and path.stem.lower() == "full" and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path.resolve()
    return None


def _derive_boundaries_from_shots(shots: list[ShotSpec]) -> list[float]:
    boundaries = []
    elapsed = 0.0
    for shot in shots[:-1]:
        duration = get_video_duration_sec(shot.file)
        if duration is None:
            return []
        elapsed += duration
        boundaries.append(elapsed)
    return boundaries


def get_video_duration_sec(path: Path) -> float | None:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frames <= 0:
            return None
        return float(frames / fps)
    finally:
        cap.release()


def _resolve_path(value: Any, base: Path, manifest_dir: Path, result_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [base / path, manifest_dir / path, result_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _looks_like_shot(filename: str) -> bool:
    return re.match(r"shot[_-]?\d+", Path(filename).stem, flags=re.IGNORECASE) is not None


def _parse_shot_id(value: Any) -> int:
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"Cannot parse shot id from {value!r}")
    return int(match.group(0))
