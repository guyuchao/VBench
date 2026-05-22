# Multi-shot Evaluation

This helper evaluates folders shaped like:

```text
result/
  video1/
    full.mp4
    shot_1.mp4
    shot_2.mp4
    shot_3.mp4
```

The first implementation covers:

- Per-shot controllability: text alignment via VBench `overall_consistency` or `clip_score`.
- Shot structure: transition control / SCA on `full.mp4`.
- Intra-shot quality: VBench `aesthetic_quality`, `dynamic_degree`, `subject_consistency`, and `background_consistency`.
- Inter-shot quality: character-wise subject consistency by sampling representative frames from each character's shots.

## Metrics

| Category | Metric | Evaluation object | Calculation | Code interface |
| --- | --- | --- | --- | --- |
| Per-shot controllability | Text Alignment | Each `shot_X.mp4` and its shot caption | Runs VBench `overall_consistency` by default, or `clip_score` if selected. Scores whether each shot follows its own text prompt. | `VBenchMetricRunner.run_text_alignment(..., metric="overall_consistency")`; CLI: `--metrics text_alignment --text_metric overall_consistency` |
| Shot structure | Transition Control / SCA | `full.mp4` in each result video folder | Runs TransNetV2 on `full.mp4` to predict cut frames, compares them with target cut frames derived from cumulative saved `shot_X.mp4` frame counts, then computes `SCA = exp(-NSD)` where `NSD = (E_matched + E_penalty) / total_frames`. | `evaluate_sca(...)`; CLI: `--metrics sca --sca_detector transnetv2 --sca_threshold 0.5` |
| Intra-shot quality | Aesthetic Quality | Each `shot_X.mp4` | Runs VBench `aesthetic_quality` on each shot, then reports per-shot and average scores. | `VBenchMetricRunner.run_intra_quality(..., ["aesthetic_quality"])`; CLI: `--metrics intra_quality --intra_quality_dimensions aesthetic_quality` |
| Intra-shot quality | Dynamic Degree | Each `shot_X.mp4` | Runs VBench `dynamic_degree` to measure whether the shot has enough motion dynamics. | `VBenchMetricRunner.run_intra_quality(..., ["dynamic_degree"])`; CLI: `--intra_quality_dimensions dynamic_degree` |
| Intra-shot quality | Intra-shot Subject Consistency | Each `shot_X.mp4` | Runs VBench `subject_consistency` within each shot. This is the native VBench version, not YOLO/SAM-masked character matching. | `VBenchMetricRunner.run_intra_quality(..., ["subject_consistency"])`; CLI: `--intra_quality_dimensions subject_consistency` |
| Intra-shot quality | Background Consistency | Each `shot_X.mp4` | Runs VBench `background_consistency` within each shot. | `VBenchMetricRunner.run_intra_quality(..., ["background_consistency"])`; CLI: `--intra_quality_dimensions background_consistency` |
| Inter-shot quality | Character-wise Subject Consistency | Character-specific shot groups, for example `characterA` in shots `[1, 3, 4]` | For each character in the manifest, samples one representative frame from every shot where the character appears, builds a temporary character video, then runs VBench `subject_consistency` on that temporary video. This is a current VBench baseline and compares whole representative frames. | `VBenchMetricRunner.run_character_subject_consistency(...)`; CLI: `--metrics inter_shot_quality --character_frame_strategy middle` |

Example manifest:

```json
{
  "video1": {
    "dir": "result/video1",
    "full_video": "result/video1/full.mp4",
    "shots": [
      {
        "id": 1,
        "file": "shot_1.mp4",
        "caption": "A woman in a red coat enters a dim restaurant.",
        "characters": ["characterA"]
      },
      {
        "id": 2,
        "file": "shot_2.mp4",
        "caption": "A waiter places a candle on the table.",
        "characters": ["characterB"]
      }
    ],
    "target_boundaries_frames": [89]
  }
}
```

Run:

```bash
python tools/multishot_eval/run_eval.py \
  --result_root result \
  --manifest manifest.json \
  --output eval_results/multishot \
  --metrics text_alignment sca intra_quality inter_shot_quality
```

For SCA, target cut frames are derived first from the actual saved `shot_X.mp4` files by cumulatively summing their frame counts. If any saved shot frame count cannot be read, the tool falls back to manifest `target_boundaries_frames`, then manifest `target_boundaries_sec`, then cumulative `shot_X.mp4` durations. When converting from `eval_caption_multishot_t2v*.json`, manifest frame targets are still recorded from source `switch_latent_frames` with `target_frame = (switch_latent_frame - 1) * 4 + 1`, but they are now only a fallback when saved shot frame counts are unavailable.

Convert prompts from `eval_caption_multishot_t2v_sample.json`:

```bash
python tools/multishot_eval/convert_eval_caption_manifest.py \
  --input eval_caption_multishot_t2v_sample.json \
  --output eval_caption_multishot_t2v_sample_manifest.json \
  --result_root result
```

By default, sample 0 becomes `result/video1`, sample 1 becomes `result/video2`, and so on.

One-shot run:

```bash
bash tools/multishot_eval/run_multishot_eval.sh \
  --result-root result \
  --prompt-json eval_caption_multishot_t2v.json \
  --manifest eval_caption_multishot_t2v_manifest.json \
  --output eval_results/multishot \
  --device cuda
```

The script converts the prompt JSON to a manifest when needed, then runs text alignment, SCA, intra-shot quality, and the current VBench-based character-wise subject consistency in one pass.
