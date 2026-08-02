from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from vbench.distributed import barrier, gather_list_of_dict, get_rank, get_world_size


DEFAULT_METRICS = [
    "overall_quality",
    "shot_structure",
    "intra_shot_quality",
]
DEFAULT_OVERALL_QUALITY_DIMS = [
    "aesthetic_quality",
    "dynamic_degree",
]
DEFAULT_INTRA_SHOT_QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
]


class VBenchMultishot:
    def __init__(
        self,
        device: str = "cuda",
        output_dir: str | Path = "multishot_eval_results",
        text_alignment_metric: str = "overall_consistency",
        overall_quality_dimensions: list[str] | None = None,
        intra_shot_quality_dimensions: list[str] | None = None,
        load_ckpt_from_local: bool = False,
        read_frame: bool = False,
        keep_vbench_meta: bool = False,
        sca_detector: str = "transnetv2",
        sca_tolerance_sec: float | None = None,
        sca_threshold: float = 0.5,
        sca_min_gap_sec: float = 0.35,
        sca_unmatched_penalty_frames: float | None = None,
        transnetv2_path: str | Path | None = None,
        transnetv2_weights: str | Path | None = None,
        character_frame_strategy: str = "middle",
        continue_on_error: bool = False,
    ) -> None:
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_alignment_metric = text_alignment_metric
        self.overall_quality_dimensions = overall_quality_dimensions or DEFAULT_OVERALL_QUALITY_DIMS
        self.intra_shot_quality_dimensions = intra_shot_quality_dimensions or DEFAULT_INTRA_SHOT_QUALITY_DIMS
        self.load_ckpt_from_local = load_ckpt_from_local
        self.read_frame = read_frame
        self.keep_vbench_meta = keep_vbench_meta
        self.sca_detector = sca_detector
        self.sca_tolerance_sec = sca_tolerance_sec
        self.sca_threshold = sca_threshold
        self.sca_min_gap_sec = sca_min_gap_sec
        self.sca_unmatched_penalty_frames = sca_unmatched_penalty_frames
        default_transnetv2_path = Path(__file__).resolve().parents[1] / 'tools' / 'transnetv2' / 'inference_pytorch'
        default_transnetv2_weights = Path(os.environ.get('VBENCH_CACHE_DIR', 'experiments/pretrained_models/vbench')) / 'transnetv2' / 'transnetv2-pytorch-weights.pth'
        self.transnetv2_path = Path(transnetv2_path) if transnetv2_path is not None else default_transnetv2_path
        self.transnetv2_weights = Path(transnetv2_weights) if transnetv2_weights is not None else default_transnetv2_weights
        self.character_frame_strategy = character_frame_strategy
        self.continue_on_error = continue_on_error

    def evaluate(
        self,
        result_root: str | Path,
        manifest: str | Path | None = None,
        metrics: list[str] | None = None,
        save_json: bool = True,
        return_raw_results: bool = False,
    ) -> dict[str, Any]:
        multishot_tools = Path(__file__).resolve().parents[1] / "tools" / "multishot_eval"
        if str(multishot_tools) not in sys.path:
            sys.path.insert(0, str(multishot_tools))

        from tools.multishot_eval.manifest import load_manifest
        from tools.multishot_eval.run_eval import evaluate_video
        from tools.multishot_eval.vbench_runner import VBenchMetricRunner
        import torch

        metrics = metrics or DEFAULT_METRICS
        args = Namespace(
            metrics=metrics,
            text_alignment_metric=self.text_alignment_metric,
            overall_quality_dimensions=self.overall_quality_dimensions,
            intra_shot_quality_dimensions=self.intra_shot_quality_dimensions,
            sca_detector=self.sca_detector,
            sca_tolerance_sec=self.sca_tolerance_sec,
            sca_threshold=self.sca_threshold,
            sca_min_gap_sec=self.sca_min_gap_sec,
            sca_unmatched_penalty_frames=self.sca_unmatched_penalty_frames,
            transnetv2_path=self.transnetv2_path,
            transnetv2_weights=self.transnetv2_weights,
            character_frame_strategy=self.character_frame_strategy,
            continue_on_error=self.continue_on_error,
        )

        all_videos = load_manifest(Path(manifest) if manifest is not None else None, Path(result_root))
        rank = get_rank()
        world_size = get_world_size()
        videos = all_videos[rank::world_size]

        needs_vbench = bool({"overall_quality", "intra_shot_quality", "inter_shot_quality"} & set(metrics))
        runner = None
        if needs_vbench:
            runner = VBenchMetricRunner(
                output_dir=self.output_dir,
                device=self.device,
                load_ckpt_from_local=self.load_ckpt_from_local,
                read_frame=self.read_frame,
                keep_meta=self.keep_vbench_meta,
            )

        local_items: list[dict[str, Any]] = []
        for video in videos:
            local_items.append({"video_id": video.id, "result": evaluate_video(video, args, runner)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if world_size > 1:
            gathered_items = gather_list_of_dict(local_items)
        else:
            gathered_items = local_items
        results = {item["video_id"]: item["result"] for item in gathered_items}
        summary = summarize_results(results)

        if save_json and rank == 0:
            output_path = self.output_dir / "multishot_eval_results.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                f.write("\n")
            summary_path = self.output_dir / "multishot_eval_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                f.write("\n")
        barrier()
        return results if return_raw_results else summary


def summarize_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    summary = {"num_videos": len(raw_results)}
    collected: dict[str, list[float]] = {}

    def collect(metric_name: str, data: Any, path: list[str]) -> None:
        value = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return
            value = value[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            collected.setdefault(metric_name, []).append(float(value))

    for video_result in raw_results.values():
        collect("overall_quality_text_alignment", video_result, ["overall_quality", "text_alignment", "average"])
        collect("inter_shot_quality", video_result, ["inter_shot_quality", "average"])
        for key in ["sca", "nsd", "boundary_match_rate", "cut_precision", "cut_recall", "cut_count_accuracy"]:
            collect(f"shot_structure_{key}", video_result, ["shot_structure", key])

        for section in ["overall_quality", "intra_shot_quality"]:
            section_result = video_result.get(section)
            if not isinstance(section_result, dict):
                continue
            for dimension, dimension_result in section_result.items():
                if dimension == "text_alignment":
                    continue
                if isinstance(dimension_result, dict):
                    collect(f"{section}_{dimension}", video_result, [section, dimension, "average"])

    for key, values in collected.items():
        if values:
            summary[key] = sum(values) / len(values)
    return summary
