from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


@dataclass
class SCAMatch:
    target_frame: int
    detected_frame: int
    error_frames: int
    target_sec: float
    detected_sec: float
    error_sec: float


@dataclass
class SCAResult:
    sca: float
    nsd: float
    e_matched_frames: float
    e_penalty_frames: float
    unmatched_penalty_frames: float
    total_frames: int
    boundary_match_rate: float
    cut_precision: float
    cut_recall: float
    cut_count_accuracy: float
    mean_boundary_timing_error_frames: float | None
    mean_boundary_timing_error_sec: float | None
    detected_boundaries_frames: list[int]
    target_boundaries_frames: list[int]
    detected_boundaries_sec: list[float]
    target_boundaries_sec: list[float]
    matched_boundaries: list[SCAMatch]
    missed_boundaries_frames: list[int]
    missed_boundaries_sec: list[float]
    false_positive_boundaries_frames: list[int]
    false_positive_boundaries_sec: list[float]
    detector: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["matched_boundaries"] = [asdict(match) for match in self.matched_boundaries]
        return data


def evaluate_sca(
    full_video: Path,
    target_boundaries_frames: list[int] | None = None,
    target_boundaries_sec: list[float] | None = None,
    tolerance_sec: float | None = None,
    detector: str = "transnetv2",
    threshold: float = 0.5,
    min_gap_sec: float = 0.35,
    unmatched_penalty_frames: float | None = None,
    transnetv2_path: Path | None = None,
    transnetv2_weights: Path | None = None,
) -> SCAResult:
    if not full_video.exists():
        raise FileNotFoundError(f"Full video not found: {full_video}")

    fps, total_frames = get_video_properties(full_video)
    targets = _resolve_target_frames(target_boundaries_frames, target_boundaries_sec, fps)
    if not targets:
        raise ValueError("SCA requires target boundaries in frames or seconds.")

    if detector == "transnetv2":
        detected = detect_cuts_transnetv2(
            full_video,
            threshold=threshold,
            transnetv2_path=transnetv2_path,
            transnetv2_weights=transnetv2_weights,
        )
    elif detector == "scenedetect":
        detected = [int(round(sec * fps)) for sec in detect_cuts_scenedetect(full_video)]
    elif detector == "opencv":
        detected = [
            int(round(sec * fps))
            for sec in detect_cuts_opencv(full_video, threshold=threshold, min_gap_sec=min_gap_sec)
        ]
    else:
        raise ValueError(f"Unsupported SCA detector: {detector}")

    tolerance_frames = int(round(tolerance_sec * fps)) if tolerance_sec is not None else None
    return score_boundaries(
        detected_boundaries_frames=detected,
        target_boundaries_frames=targets,
        fps=fps,
        total_frames=total_frames,
        unmatched_penalty_frames=unmatched_penalty_frames,
        tolerance_frames=tolerance_frames,
        detector=detector,
    )


def score_boundaries(
    detected_boundaries_frames: list[int],
    target_boundaries_frames: list[int],
    fps: float,
    total_frames: int,
    unmatched_penalty_frames: float | None,
    tolerance_frames: int | None,
    detector: str,
) -> SCAResult:
    detected = sorted(_clip_frame(x, total_frames) for x in detected_boundaries_frames)
    targets = sorted(_clip_frame(x, total_frames) for x in target_boundaries_frames)
    if total_frames <= 0:
        raise ValueError("SCA requires a positive total frame count.")
    if unmatched_penalty_frames is None:
        unmatched_penalty_frames = total_frames / max(len(targets) + 1, 1)

    pair_indices = _match_cut_indices(targets, detected)
    matched: list[SCAMatch] = []
    used_targets: set[int] = set()
    used_detected: set[int] = set()
    for target_idx, detected_idx in pair_indices:
        target = targets[target_idx]
        cut = detected[detected_idx]
        error = abs(cut - target)
        if tolerance_frames is not None and error > tolerance_frames:
            continue
        used_targets.add(target_idx)
        used_detected.add(detected_idx)
        matched.append(
            SCAMatch(
                target_frame=target,
                detected_frame=cut,
                error_frames=error,
                target_sec=target / fps,
                detected_sec=cut / fps,
                error_sec=error / fps,
            )
        )

    missed_frames = [target for idx, target in enumerate(targets) if idx not in used_targets]
    false_positive_frames = [cut for idx, cut in enumerate(detected) if idx not in used_detected]
    match_count = len(matched)
    precision = match_count / len(detected) if detected else (1.0 if not targets else 0.0)
    recall = match_count / len(targets) if targets else 1.0
    cut_count_accuracy = 1.0 - min(1.0, abs(len(detected) - len(targets)) / max(len(targets), 1))

    matched_errors = [match.error_frames for match in matched]
    e_matched = float(np.sum(matched_errors)) if matched_errors else 0.0
    e_penalty = float((len(missed_frames) + len(false_positive_frames)) * unmatched_penalty_frames)
    nsd = float((e_matched + e_penalty) / total_frames)
    sca = float(math.exp(-nsd))
    mean_error_frames = float(np.mean(matched_errors)) if matched_errors else None
    mean_error_sec = mean_error_frames / fps if mean_error_frames is not None else None

    return SCAResult(
        sca=sca,
        nsd=nsd,
        e_matched_frames=e_matched,
        e_penalty_frames=e_penalty,
        unmatched_penalty_frames=float(unmatched_penalty_frames),
        total_frames=int(total_frames),
        boundary_match_rate=float(recall),
        cut_precision=float(precision),
        cut_recall=float(recall),
        cut_count_accuracy=float(cut_count_accuracy),
        mean_boundary_timing_error_frames=mean_error_frames,
        mean_boundary_timing_error_sec=mean_error_sec,
        detected_boundaries_frames=detected,
        target_boundaries_frames=targets,
        detected_boundaries_sec=[frame / fps for frame in detected],
        target_boundaries_sec=[frame / fps for frame in targets],
        matched_boundaries=matched,
        missed_boundaries_frames=missed_frames,
        missed_boundaries_sec=[frame / fps for frame in missed_frames],
        false_positive_boundaries_frames=false_positive_frames,
        false_positive_boundaries_sec=[frame / fps for frame in false_positive_frames],
        detector=detector,
    )


def _resolve_target_frames(
    target_boundaries_frames: list[int] | None,
    target_boundaries_sec: list[float] | None,
    fps: float,
) -> list[int]:
    if target_boundaries_frames:
        return [int(round(frame)) for frame in target_boundaries_frames]
    if target_boundaries_sec:
        return [int(round(sec * fps)) for sec in target_boundaries_sec]
    return []


def _match_cut_indices(targets: list[int], detected: list[int]) -> list[tuple[int, int]]:
    if not targets or not detected:
        return []

    pairs: list[tuple[int, int]] = []
    used_targets: set[int] = set()
    used_detected: set[int] = set()
    candidates = sorted(
        (abs(target - cut), target_idx, detected_idx)
        for target_idx, target in enumerate(targets)
        for detected_idx, cut in enumerate(detected)
    )
    for _, target_idx, detected_idx in candidates:
        if target_idx in used_targets or detected_idx in used_detected:
            continue
        used_targets.add(target_idx)
        used_detected.add(detected_idx)
        pairs.append((target_idx, detected_idx))
        if len(pairs) >= min(len(targets), len(detected)):
            break
    return pairs


def _clip_frame(frame: int, total_frames: int) -> int:
    return max(0, min(int(frame), max(total_frames - 1, 0)))


def get_video_properties(full_video: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(full_video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"Cannot read FPS from {full_video}")
    if total_frames <= 0:
        raise ValueError(f"Cannot read frame count from {full_video}")
    return float(fps), total_frames


_TRANSNETV2_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}


def detect_cuts_transnetv2(
    full_video: Path,
    threshold: float = 0.5,
    transnetv2_path: Path | None = None,
    transnetv2_weights: Path | None = None,
) -> list[int]:
    model = _load_transnetv2(transnetv2_path=transnetv2_path, transnetv2_weights=transnetv2_weights)
    _, single_frame_predictions, _ = model.predict_video(str(full_video))
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
    return [int(scene[0]) for scene in scenes[1:]]


def _load_transnetv2(transnetv2_path: Path | None, transnetv2_weights: Path | None) -> Any:
    if transnetv2_path is None:
        transnetv2_path = Path(__file__).resolve().parents[1] / "TransNetV2-master" / "inference-pytorch"
    transnetv2_path = transnetv2_path.resolve()
    weights_key = str(transnetv2_weights.resolve()) if transnetv2_weights else None
    cache_key = (str(transnetv2_path), weights_key)
    if cache_key in _TRANSNETV2_MODEL_CACHE:
        return _TRANSNETV2_MODEL_CACHE[cache_key]

    module_path = transnetv2_path / "transnetv2_pytorch.py"
    if module_path.exists():
        model = _load_transnetv2_pytorch(module_path, transnetv2_weights)
        _TRANSNETV2_MODEL_CACHE[cache_key] = model
        return model

    module_path = transnetv2_path / "transnetv2.py"
    if not module_path.exists():
        raise FileNotFoundError(f"TransNetV2 inference file not found under: {transnetv2_path}")

    sys.path.insert(0, str(transnetv2_path))
    spec = importlib.util.spec_from_file_location("multishot_eval_transnetv2", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import TransNetV2 from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_dir = str(transnetv2_weights.resolve()) if transnetv2_weights else None
    model = module.TransNetV2(model_dir)
    _TRANSNETV2_MODEL_CACHE[cache_key] = model
    return model


class _TorchTransNetV2Adapter:
    def __init__(self, model: Any, device: Any) -> None:
        self.model = model
        self.device = device

    def predict_video(self, video_fn: str):
        frames = _read_transnetv2_frames(Path(video_fn))
        return (frames, *self.predict_frames(frames))

    def predict_frames(self, frames: np.ndarray):
        import torch

        assert len(frames.shape) == 4 and list(frames.shape[1:]) == [27, 48, 3], \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        predictions = []
        with torch.no_grad():
            for inp in _transnetv2_input_windows(frames):
                tensor = torch.from_numpy(inp).to(self.device)
                single_frame_pred, all_frames_pred = self.model(tensor)
                single_frame_pred = torch.sigmoid(single_frame_pred).detach().cpu().numpy()
                all_frames_pred = torch.sigmoid(all_frames_pred["many_hot"]).detach().cpu().numpy()
                predictions.append((single_frame_pred[0, 25:75, 0], all_frames_pred[0, 25:75, 0]))

        single_frame_pred = np.concatenate([single_ for single_, _ in predictions])
        all_frames_pred = np.concatenate([all_ for _, all_ in predictions])
        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)


def _load_transnetv2_pytorch(module_path: Path, transnetv2_weights: Path | None) -> _TorchTransNetV2Adapter:
    import torch

    weights_path = _resolve_transnetv2_pytorch_weights(module_path.parent, transnetv2_weights)
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("multishot_eval_transnetv2_pytorch", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import TransNetV2 PyTorch from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.TransNetV2()
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return _TorchTransNetV2Adapter(model, device)


def _resolve_transnetv2_pytorch_weights(transnetv2_path: Path, transnetv2_weights: Path | None) -> Path:
    if transnetv2_weights is None:
        weights_path = transnetv2_path / "transnetv2-pytorch-weights.pth"
    else:
        weights_path = transnetv2_weights.resolve()
        if weights_path.is_dir():
            weights_path = weights_path / "transnetv2-pytorch-weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            "TransNetV2 PyTorch weights not found. Expected "
            f"{weights_path}. Run tools/TransNetV2-master/inference-pytorch/convert_weights.py "
            "or pass --transnetv2_weights PATH_TO_PTH."
        )
    return weights_path


def _read_transnetv2_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Could not read frames from {video_path}")
    return np.asarray(frames, dtype=np.uint8)


def _transnetv2_input_windows(frames: np.ndarray):
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded_inputs = np.concatenate(
        [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end,
        axis=0,
    )

    ptr = 0
    while ptr + 100 <= len(padded_inputs):
        out = padded_inputs[ptr:ptr + 100]
        ptr += 50
        yield out[np.newaxis]


def detect_cuts_scenedetect(full_video: Path, threshold: float = 27.0, min_scene_len: int = 8) -> list[float]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(full_video))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    return [scene[0].get_seconds() for scene in scene_list[1:]]


def detect_cuts_opencv(full_video: Path, threshold: float = 0.42, min_gap_sec: float = 0.35) -> list[float]:
    cap = cv2.VideoCapture(str(full_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Cannot read FPS from {full_video}")

    cuts = []
    last_cut_frame = -10**9
    min_gap_frames = max(1, int(round(min_gap_sec * fps)))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return cuts

    prev_small, prev_hist = _frame_features(prev)
    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur_small, cur_hist = _frame_features(frame)
        diff_score = float(np.mean(cv2.absdiff(prev_small, cur_small))) / 255.0
        hist_score = float(cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_BHATTACHARYYA))
        score = 0.7 * diff_score + 0.3 * hist_score
        if score >= threshold and frame_idx - last_cut_frame >= min_gap_frames:
            cuts.append(frame_idx / fps)
            last_cut_frame = frame_idx
        prev_small, prev_hist = cur_small, cur_hist
        frame_idx += 1

    cap.release()
    return cuts


def _frame_features(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return small, hist
