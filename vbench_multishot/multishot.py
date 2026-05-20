from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


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
        default_transnetv2_path = Path(__file__).resolve().parents[1] / 'tools' / 'TransNetV2-master' / 'inference-pytorch'
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

        videos = load_manifest(Path(manifest) if manifest is not None else None, Path(result_root))
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

        results: dict[str, Any] = {}
        for video in videos:
            results[video.id] = evaluate_video(video, args, runner)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if save_json:
            output_path = self.output_dir / "multishot_eval_results.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                f.write("\n")
        return results


