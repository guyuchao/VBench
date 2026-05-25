#!/usr/bin/env python
"""Run all VBench-Long dimensions and summarize per-dimension averages.

This mirrors vbench2_beta_long/evaluate_long.sh, but works from an installed
Python package when vbench2_beta_long is importable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean


DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "appearance_style",
]

FOLDERS = [
    "subject_consistency",
    "scene",
    "overall_consistency",
    "overall_consistency",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "subject_consistency",
    "subject_consistency",
    "appearance_style",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all VBench-Long dimensions and write score summaries."
    )
    parser.add_argument(
        "--videos_path",
        required=True,
        help="Base video folder. Dimension subfolders are used automatically when present.",
    )
    parser.add_argument(
        "--output_path",
        default="./evaluation_results_long_all",
        help="Directory for raw VBench outputs and final summaries.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Root cache directory for VBench and common model backends. "
            "Sets VBENCH_CACHE_DIR, TORCH_HOME, HF_HOME, and related variables."
        ),
    )
    parser.add_argument(
        "--vbench_repo",
        default=None,
        help=(
            "Optional path to a VBench source checkout. Use this when the installed "
            "vbench package does not include vbench2_beta_long."
        ),
    )
    parser.add_argument(
        "--mode",
        default="long_vbench_standard",
        choices=["long_vbench_standard", "long_custom_input"],
        help="VBench-Long evaluation mode.",
    )
    parser.add_argument(
        "--full_json_dir",
        default=None,
        help="Optional path to VBench_full_info.json. Defaults to the installed package copy.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. VBench-Long normally expects cuda.",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        action="store_true",
        help="Load checkpoints from VBench local default paths.",
    )
    parser.add_argument(
        "--read_frame",
        action="store_true",
        help="Read frames directly instead of videos.",
    )
    parser.add_argument(
        "--use_semantic_splitting",
        action="store_true",
        help="Enable semantic scene splitting before fixed-length clipping.",
    )
    parser.add_argument(
        "--num_of_samples_per_prompt",
        type=int,
        default=5,
        help="Samples per prompt for long_vbench_standard mode.",
    )
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="Continue with later dimensions if one dimension fails.",
    )
    return parser.parse_args()


def configure_cache_dir(cache_dir: str | None) -> Path | None:
    if not cache_dir:
        return None

    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    env_paths = {
        "VBENCH_CACHE_DIR": root,
        "TORCH_HOME": root / "torch",
        "HF_HOME": root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": root / "huggingface" / "transformers",
        "HF_DATASETS_CACHE": root / "huggingface" / "datasets",
        "XDG_CACHE_HOME": root,
        "PYTORCH_PRETRAINED_BERT_CACHE": root / "pytorch_pretrained_bert",
        "PYTORCH_TRANSFORMERS_CACHE": root / "pytorch_transformers",
    }
    for key, path in env_paths.items():
        os.environ[key] = str(path)
        Path(path).mkdir(parents=True, exist_ok=True)

    original_expanduser = os.path.expanduser

    def expanduser_with_cache(path: str) -> str:
        normalized = path.replace(r"\\", "/")
        if normalized == "~/.cache":
            return str(root)
        if normalized.startswith("~/.cache/"):
            return str(root / normalized[len("~/.cache/") :])
        return original_expanduser(path)

    os.path.expanduser = expanduser_with_cache
    return root


def require_existing_file(path: Path, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return str(path)


def add_vbench_repo_to_path(vbench_repo: str | None) -> Path | None:
    candidates = []
    if vbench_repo:
        candidates.append(Path(vbench_repo).resolve())
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.home() / "Desktop" / "VBench")

    for candidate in candidates:
        if (candidate / "vbench2_beta_long" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return candidate
    return None


def load_long_package(vbench_repo: str | None):
    source_repo = add_vbench_repo_to_path(vbench_repo)
   
    import torch
    import vbench2_beta_long
    from vbench2_beta_long import VBenchLong
   

    package_dir = Path(vbench2_beta_long.__file__).resolve().parent
    return torch, VBenchLong, package_dir, source_repo


def extract_score(results: dict, dimension: str) -> float:
    value = results[dimension]
    if isinstance(value, list) and value:
        return float(value[0])
    if isinstance(value, tuple) and value:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Cannot extract average score for {dimension}: {value!r}")


def main() -> None:
    args = parse_args()
    cache_root = configure_cache_dir(args.cache_dir)
    torch, VBenchLong, package_dir, source_repo = load_long_package(args.vbench_repo)
    if cache_root is not None:
        try:
            torch.hub.set_dir(str(cache_root / "torch" / "hub"))
        except Exception:
            pass

    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    full_json_dir = (
        Path(args.full_json_dir).resolve()
        if args.full_json_dir
        else package_dir / "VBench_full_info.json"
    )

    kwargs = {
        "sb_clip2clip_feat_extractor": "dinov2",
        "bg_clip2clip_feat_extractor": "dreamsim",
        "imaging_quality_preprocessing_mode": "longer",
        "clip_length_config": "clip_length_mix.yaml",
        "w_inclip": 1.0,
        "w_clip2clip": 0.0,
        "use_semantic_splitting": args.use_semantic_splitting,
        "slow_fast_eval_config": require_existing_file(
            package_dir / "configs" / "slow_fast_params.yaml",
            "slow-fast config",
        ),
        "dev_flag": True,
        "sb_mapping_file_path": require_existing_file(
            package_dir / "configs" / "subject_mapping_table.yaml",
            "subject mapping table",
        ),
        "bg_mapping_file_path": require_existing_file(
            package_dir / "configs" / "background_mapping_table.yaml",
            "background mapping table",
        ),
        "num_of_samples_per_prompt": args.num_of_samples_per_prompt,
        "static_filter_flag": False,
    }

    base_path = Path(args.videos_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"videos_path does not exist: {base_path}")

    has_dimension_subdirs = any((base_path / folder).is_dir() for folder in FOLDERS)
    device = torch.device(args.device)
    runner = VBenchLong(device, str(full_json_dir), str(output_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []
    errors = []

    for dimension, folder in zip(DIMENSIONS, FOLDERS):
        videos_path = base_path / folder if has_dimension_subdirs else base_path
        run_name = f"long_all_{timestamp}_{dimension}"
        raw_result_file = output_path / f"{run_name}_eval_results.json"

        print(f"\n=== Evaluating {dimension} ===")
        print(f"videos_path: {videos_path}")

        try:
            run_kwargs = dict(kwargs)
            run_kwargs["static_filter_flag"] = dimension == "temporal_flickering"
            runner.evaluate(
                videos_path=str(videos_path),
                name=run_name,
                prompt_list=[],
                dimension_list=[dimension],
                local=args.load_ckpt_from_local,
                read_frame=args.read_frame,
                mode=args.mode,
                **run_kwargs,
            )

            with raw_result_file.open("r", encoding="utf-8") as f:
                raw_results = json.load(f)
            score = extract_score(raw_results, dimension)
            summary_rows.append(
                {
                    "dimension": dimension,
                    "score": score,
                    "videos_path": str(videos_path),
                    "result_file": str(raw_result_file),
                }
            )
            print(f"{dimension}: {score}")
        except Exception as exc:
            errors.append(
                {
                    "dimension": dimension,
                    "videos_path": str(videos_path),
                    "error": repr(exc),
                }
            )
            print(f"FAILED {dimension}: {exc!r}")
            if not args.keep_going:
                raise

    scores = [row["score"] for row in summary_rows]
    summary = {
        "videos_path": str(base_path),
        "mode": args.mode,
        "cache_dir": str(cache_root) if cache_root is not None else None,
        "vbench_repo": str(source_repo) if source_repo is not None else None,
        "created_at": timestamp,
        "dimension_scores": summary_rows,
        "overall_mean": mean(scores) if scores else None,
        "errors": errors,
    }

    summary_json = output_path / f"long_all_summary_{timestamp}.json"
    summary_csv = output_path / f"long_all_summary_{timestamp}.csv"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dimension", "score", "videos_path", "result_file"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(f"{row['dimension']}: {row['score']}")
    print(f"overall_mean: {summary['overall_mean']}")
    print(f"summary_json: {summary_json}")
    print(f"summary_csv: {summary_csv}")
    if errors:
        print(f"errors: {len(errors)} dimension(s), see summary_json")


if __name__ == "__main__":
    main()
