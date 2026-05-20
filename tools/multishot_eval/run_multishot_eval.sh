#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="C:\Users\Administrator\Desktop\processed_videos"
PROMPT_JSON="eval_caption_multishot_t2v.json"
MANIFEST="eval_caption_multishot_t2v_manifest.json"
OUTPUT="eval_results/ours"
DEVICE="cuda"
TEXT_METRIC="overall_consistency"
SCA_DETECTOR="transnetv2"
SCA_TOLERANCE_SEC=""
SCA_THRESHOLD="0.5"
SCA_MIN_GAP_SEC="0.35"
SCA_UNMATCHED_PENALTY_FRAMES=""
TRANSNETV2_PATH="tools/TransNetV2-master/inference-pytorch"
TRANSNETV2_WEIGHTS=""
CHARACTER_FRAME_STRATEGY="middle"
LOAD_CKPT_FROM_LOCAL=0
READ_FRAME=0
CONTINUE_ON_ERROR=0
FORCE_CONVERT_MANIFEST=0
KEEP_VBENCH_META=0

usage() {
  cat <<'EOF'
Usage: tools/multishot_eval/run_multishot_eval.sh [options]

Options:
  --result-root PATH              Generated result root. Default: result
  --prompt-json PATH              Prompt JSON. Default: eval_caption_multishot_t2v.json
  --manifest PATH                 Manifest JSON. Default: eval_caption_multishot_t2v_manifest.json
  --output PATH                   Output directory. Default: eval_results/multishot
  --device DEVICE                 Torch device. Default: cuda
  --text-metric NAME              overall_consistency or clip_score. Default: overall_consistency
  --sca-detector NAME             transnetv2, opencv, or scenedetect. Default: transnetv2
  --sca-tolerance-sec FLOAT       Optional matching tolerance; omit for minimum-error matching.
  --sca-threshold FLOAT           Detector threshold. Default: 0.5
  --sca-min-gap-sec FLOAT         Minimum gap between detected cuts. Default: 0.35
  --sca-unmatched-penalty-frames FLOAT
                                  Penalty per missed/extra cut. Default: average shot length.
  --transnetv2-path PATH          TransNetV2 inference-pytorch dir. Default: tools/TransNetV2-master/inference-pytorch
  --transnetv2-weights PATH       Optional TransNetV2 .pth weights path.
  --character-frame-strategy NAME first, middle, or last. Default: middle
  --load-ckpt-from-local          Pass --load_ckpt_from_local to VBench.
  --read-frame                    Pass --read_frame to VBench.
  --keep-vbench-meta              Keep temporary VBench full_info JSON files for debugging.
  --continue-on-error             Record metric errors and continue.
  --force-convert-manifest        Rebuild manifest even if it already exists.
  -h, --help                      Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-root)
      RESULT_ROOT="$2"
      shift 2
      ;;
    --prompt-json)
      PROMPT_JSON="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --text-metric)
      TEXT_METRIC="$2"
      shift 2
      ;;
    --sca-detector)
      SCA_DETECTOR="$2"
      shift 2
      ;;
    --sca-tolerance-sec)
      SCA_TOLERANCE_SEC="$2"
      shift 2
      ;;
    --sca-threshold)
      SCA_THRESHOLD="$2"
      shift 2
      ;;
    --sca-min-gap-sec)
      SCA_MIN_GAP_SEC="$2"
      shift 2
      ;;
    --sca-unmatched-penalty-frames)
      SCA_UNMATCHED_PENALTY_FRAMES="$2"
      shift 2
      ;;
    --transnetv2-path)
      TRANSNETV2_PATH="$2"
      shift 2
      ;;
    --transnetv2-weights)
      TRANSNETV2_WEIGHTS="$2"
      shift 2
      ;;
    --character-frame-strategy)
      CHARACTER_FRAME_STRATEGY="$2"
      shift 2
      ;;
    --load-ckpt-from-local)
      LOAD_CKPT_FROM_LOCAL=1
      shift
      ;;
    --read-frame)
      READ_FRAME=1
      shift
      ;;
    --keep-vbench-meta)
      KEEP_VBENCH_META=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --force-convert-manifest)
      FORCE_CONVERT_MANIFEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "$FORCE_CONVERT_MANIFEST" == "1" || ! -f "$MANIFEST" ]]; then
  if [[ ! -f "$PROMPT_JSON" ]]; then
    echo "Prompt JSON not found: $PROMPT_JSON" >&2
    exit 1
  fi

  python tools/multishot_eval/convert_eval_caption_manifest.py \
    --input "$PROMPT_JSON" \
    --output "$MANIFEST" \
    --result_root "$RESULT_ROOT"
fi

eval_args=(
  tools/multishot_eval/run_eval.py
  --result_root "$RESULT_ROOT"
  --manifest "$MANIFEST"
  --output "$OUTPUT"
  --metrics text_alignment sca intra_quality inter_shot_quality
  --text_metric "$TEXT_METRIC"
  --intra_quality_dimensions
    aesthetic_quality
    dynamic_degree
    subject_consistency
    background_consistency
  --device "$DEVICE"
  --sca_detector "$SCA_DETECTOR"
  --sca_threshold "$SCA_THRESHOLD"
  --sca_min_gap_sec "$SCA_MIN_GAP_SEC"
  --transnetv2_path "$TRANSNETV2_PATH"
  --character_frame_strategy "$CHARACTER_FRAME_STRATEGY"
)

if [[ -n "$SCA_TOLERANCE_SEC" ]]; then
  eval_args+=(--sca_tolerance_sec "$SCA_TOLERANCE_SEC")
fi
if [[ -n "$SCA_UNMATCHED_PENALTY_FRAMES" ]]; then
  eval_args+=(--sca_unmatched_penalty_frames "$SCA_UNMATCHED_PENALTY_FRAMES")
fi
if [[ -n "$TRANSNETV2_WEIGHTS" ]]; then
  eval_args+=(--transnetv2_weights "$TRANSNETV2_WEIGHTS")
fi
if [[ "$LOAD_CKPT_FROM_LOCAL" == "1" ]]; then
  eval_args+=(--load_ckpt_from_local)
fi
if [[ "$READ_FRAME" == "1" ]]; then
  eval_args+=(--read_frame)
fi
if [[ "$KEEP_VBENCH_META" == "1" ]]; then
  eval_args+=(--keep_vbench_meta)
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  eval_args+=(--continue_on_error)
fi

python "${eval_args[@]}"
