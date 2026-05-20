from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import torch
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from manifest import VideoSpec, load_manifest
from sca import evaluate_sca
from vbench_runner import VBenchMetricRunner


DEFAULT_METRICS = [
    "overall_quality",
    "shot_structure",
    "intra_shot_quality",
]
METRIC_CHOICES = [
    "overall_quality",
    "shot_structure",
    "intra_shot_quality",
    "inter_shot_quality",
]

DEFAULT_OVERALL_QUALITY_DIMS = [
    "aesthetic_quality",
    "dynamic_degree",
]
DEFAULT_INTRA_SHOT_QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-shot video folders.")
    parser.add_argument("--result_root", type=Path, required=True, help="Root like result/ containing video folders.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest JSON with captions and boundaries.")
    parser.add_argument("--output", type=Path, default=Path("multishot_eval_results"), help="Output directory.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=METRIC_CHOICES,
        help="Output sections to evaluate.",
    )
    parser.add_argument(
        "--text_alignment_metric",
        default="overall_consistency",
        choices=["overall_consistency", "clip_score"],
        help="VBench metric used to score each shot against its caption (text_alignment sub-output of overall_quality).",
    )
    parser.add_argument(
        "--overall_quality_dimensions",
        nargs="+",
        default=DEFAULT_OVERALL_QUALITY_DIMS,
        help="VBench dimensions evaluated per shot for overall_quality (in addition to text_alignment).",
    )
    parser.add_argument(
        "--intra_shot_quality_dimensions",
        nargs="+",
        default=DEFAULT_INTRA_SHOT_QUALITY_DIMS,
        help="VBench dimensions evaluated per shot for intra_shot_quality output.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for VBench metrics.")
    parser.add_argument("--load_ckpt_from_local", action="store_true", help="Use local VBench checkpoint paths.")
    parser.add_argument("--read_frame", action="store_true", help="Pass read_frame=True to VBench.")
    parser.add_argument(
        "--keep_vbench_meta",
        action="store_true",
        help="Keep temporary VBench full_info JSON files for debugging.",
    )
    parser.add_argument("--sca_detector", default="transnetv2", choices=["transnetv2", "opencv", "scenedetect"])
    parser.add_argument("--sca_tolerance_sec", type=float, default=None)
    parser.add_argument("--sca_threshold", type=float, default=0.5)
    parser.add_argument("--sca_min_gap_sec", type=float, default=0.35)
    parser.add_argument("--sca_unmatched_penalty_frames", type=float, default=None)
    parser.add_argument(
        "--transnetv2_path",
        type=Path,
        default=REPO_ROOT / "tools" / "TransNetV2-master" / "inference-pytorch",
        help="Path to the TransNetV2 inference-pytorch directory.",
    )
    parser.add_argument("--transnetv2_weights", type=Path, default=None, help="Optional TransNetV2 .pth weights path.")
    parser.add_argument(
        "--character_frame_strategy",
        default="middle",
        choices=["first", "middle", "last"],
        help="Which frame to sample from each character shot for inter_shot_quality.",
    )
    parser.add_argument("--continue_on_error", action="store_true", help="Record metric errors and continue.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "multishot_eval_results.json"

    videos = load_manifest(args.manifest, args.result_root)
    needs_vbench = bool(
        {"overall_quality", "intra_shot_quality", "inter_shot_quality"}
        & set(args.metrics)
    )
    runner = None
    if needs_vbench:
        runner = VBenchMetricRunner(
            output_dir=args.output,
            device=args.device,
            load_ckpt_from_local=args.load_ckpt_from_local,
            read_frame=args.read_frame,
            keep_meta=args.keep_vbench_meta,
        )

    results: dict[str, Any] = {}
    for video in videos:
        results[video.id] = evaluate_video(video, args, runner)
        _write_results(output_path, results)
        print(f"Saved progress after {video.id} to {output_path}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    print(f"Saved multi-shot evaluation results to {output_path}")


def evaluate_video(video: VideoSpec, args: argparse.Namespace, runner: VBenchMetricRunner | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "video_root": str(video.root),
        "full_video": str(video.full_video) if video.full_video else None,
        "num_shots": len(video.shots),
    }

    if "overall_quality" in args.metrics:
        if runner is None:
            raise RuntimeError("overall_quality requires VBenchMetricRunner.")
        overall: dict[str, Any] = {}
        overall["text_alignment"] = _safe_run(
            lambda: runner.run_text_alignment(video.id, video.shots, args.text_alignment_metric),
            args.continue_on_error,
        )
        dim_results = _safe_run(
            lambda: runner.run_intra_quality(
                video.id, video.shots, args.overall_quality_dimensions
            ),
            args.continue_on_error,
        )
        if isinstance(dim_results, dict) and "status" not in dim_results:
            overall.update(dim_results)
        else:
            overall["_dimensions"] = dim_results
        result["overall_quality"] = overall

    if "shot_structure" in args.metrics:
        result["shot_structure"] = _safe_run(
            lambda: _run_sca(video, args),
            args.continue_on_error,
        )

    if "intra_shot_quality" in args.metrics:
        if runner is None:
            raise RuntimeError("intra_shot_quality requires VBenchMetricRunner.")
        result["intra_shot_quality"] = _safe_run(
            lambda: runner.run_intra_quality(
                video.id, video.shots, args.intra_shot_quality_dimensions
            ),
            args.continue_on_error,
        )

    if "inter_shot_quality" in args.metrics:
        if runner is None:
            raise RuntimeError("inter_shot_quality requires VBenchMetricRunner.")
        result["inter_shot_quality"] = _safe_run(
            lambda: runner.run_character_subject_consistency(
                video.id,
                video.shots,
                _manifest_characters(video),
                frame_strategy=args.character_frame_strategy,
            ),
            args.continue_on_error,
        )
    return result


def _run_sca(video: VideoSpec, args: argparse.Namespace) -> dict[str, Any]:
    if video.full_video is None:
        return {"status": "skipped", "reason": "full.mp4 not found."}
    if not video.target_boundaries_frames and not video.target_boundaries_sec:
        return {"status": "skipped", "reason": "No target boundaries available."}
    sca_result = evaluate_sca(
        full_video=video.full_video,
        target_boundaries_frames=video.target_boundaries_frames,
        target_boundaries_sec=video.target_boundaries_sec,
        tolerance_sec=args.sca_tolerance_sec,
        detector=args.sca_detector,
        threshold=args.sca_threshold,
        min_gap_sec=args.sca_min_gap_sec,
        unmatched_penalty_frames=args.sca_unmatched_penalty_frames,
        transnetv2_path=args.transnetv2_path,
        transnetv2_weights=args.transnetv2_weights,
    )
    return sca_result.to_dict()


def _safe_run(fn, continue_on_error: bool) -> Any:
    try:
        return fn()
    except Exception as exc:
        if not continue_on_error:
            raise
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _write_results(output_path: Path, results: dict[str, Any]) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(output_path)


def _manifest_characters(video: VideoSpec) -> dict[str, Any]:
    characters = video.raw.get("characters")
    if isinstance(characters, dict) and characters:
        return characters

    inferred: dict[str, dict[str, Any]] = {}
    for shot in video.shots:
        for character in shot.characters:
            inferred.setdefault(character, {"appears_in": []})["appears_in"].append(shot.id)
    return inferred


if __name__ == "__main__":
    main()
