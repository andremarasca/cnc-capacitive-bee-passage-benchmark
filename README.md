# CNC Capacitive Bee Passage Benchmark

This repository contains the code, processed dataset, frozen experiment plans, trained checkpoints, per-run evaluations, and aggregate reports used by the current RF165 article study. It evaluates directional passage episodes detected from a single capacitive signal on a CNC-controlled bench.

## Release status

This is a private pre-release snapshot under scientific and editorial review. No license is granted at this stage. Public reuse terms will be selected only when the repository is approved for release.

The license notice for the MS-TCN++ reference implementation is preserved in `THIRD_PARTY_NOTICES`. That notice does not grant a license for the original code, data, or documentation in this repository.

## Scientific scope

The packaged study contains 36 acquisition sessions and 22,571 complete passage episodes. The final 30 Hz sequences were derived from 100 kHz recordings using the documented causal Bessel band-pass filter and CNC-position labels.

The results apply only to the controlled bench protocol. They do not establish field-hive performance, population-level biological generalization, orientation-independent directional sensing, or separate individual counts for the two touching specimens used in collection C3.

## Included artifacts

| Path | Content |
|---|---|
| `src/bee_counter_bench/` | Acquisition, preprocessing, splitting, normalization, MS-TCN++, training, event matching, evaluation, and reporting code. |
| `configs/` | RF165 architecture, article campaign, preprocessing, hardware, and causal-normalization configurations. |
| `data/gcode/` and `outputs/gcode/` | Frozen and deterministically regenerated CNC motion programs. |
| `data/metadata/` | Collection, specimen, hardware, G-code, raw-manifest, and figure-example metadata. |
| `outputs/processed/dataset/` | Complete processed 30 Hz dataset. |
| `outputs/splits/` | Session-grouped five-fold and leave-one-collection-out assignments used by the article. |
| `outputs/normalization/` | Training-only normalization manifests for both article protocols. |
| `outputs/experiments/article_rf165/` | Curated RF165 checkpoints, evaluations, training histories, manifests, audits, and aggregate reports. |
| `outputs/baseline/main_session_grouped.json` | Threshold-and-lobe-order heuristic comparator used by the article. |
| `outputs/reports/temporal_examples/` | Data provenance for the temporal signal examples. |
| `tests/` | Automated tests for the scientific and software contracts. |

The 1,142 raw NPZ chunks, totaling approximately 11.5 GiB, are not included. Their expected relative paths, sizes, and SHA-256 hashes remain documented in `data/metadata/raw_manifest.json`. No raw signal chunk is required to inspect the processed dataset or reproduce training and evaluation from the released 30 Hz sequences.

## Environment

Python 3.11 to 3.13 and `uv` are supported. Install the training, reporting, and test dependencies with:

```powershell
uv sync --extra train --extra reporting --extra test
```

Hardware acquisition dependencies are optional:

```powershell
uv sync --extra acquisition
```

The canonical GPU campaign recorded Python 3.13.14, PyTorch 2.12.1+cu130, CUDA 13.0, cuDNN 9.2, NumPy 2.4.6, SciPy 1.18.0, and PyYAML 6.0.3 in each run manifest.

## Verification

Run the release-integrity audit before using or publishing the snapshot:

```powershell
uv run python scripts/release_audit.py
```

Run the software and RF165 model checks with:

```powershell
uv run pytest -q -p no:cacheprovider
uv run bee-counter-bench verify-gcode
uv run bee-counter-bench smoke-test --device cpu --model-config configs/model_rf165.yaml
```

The release audit verifies the curated file manifest, processed sequence hashes and schemas, dataset totals, RF165 configuration, campaign completeness, absence of raw chunks, and absence of transient or legacy result artifacts.

The byte-exact G-code check reproduces the Windows acquisition environment. The pinned G-code artifacts retain the CRLF line endings used during acquisition; the motion commands themselves are controller text and do not depend on the host operating system.

## Reproduction

The article campaign is defined by `configs/article_rf165.yaml`. It contains two normalization variants, two evaluation protocols, and seeds 42, 43, and 44, totaling 48 training and evaluation tasks.

The general `configs/model.yaml` and `configs/experiment.yaml` defaults are retained unchanged because they are part of the source hash embedded in every article artifact. The article plan explicitly selects `configs/model_rf165.yaml` and the corresponding 82-frame context on each side. Do not substitute the general defaults when reproducing the article campaign.

The packaged campaign is an immutable evidence snapshot. Validate it with `scripts/release_audit.py`; do not regenerate its reports on `main`, because its campaign manifest intentionally retains the original archive commit.

For a full independent rerun, create a disposable branch, remove only the packaged campaign from that branch, and ignore its regenerated output before execution. Neither the packaged outputs nor `.gitignore` participate in the scientific source hash.

```powershell
git switch -c reproduce-rf165
git rm -r outputs/experiments/article_rf165
Add-Content -Path .gitignore -Value "/outputs/experiments/article_rf165/"
git add .gitignore
git commit -m "🧪 test: prepare clean RF165 reproduction"
uv run bee-counter-bench run-article-suite --plan configs/article_rf165.yaml --parallel 1 --device cpu
uv run bee-counter-bench summarize-article-experiment --plan configs/article_rf165.yaml
```

GPU execution can replace `--device cpu` with `--device cuda` and increase `--parallel` according to available memory. The immutable guards require a clean Git tree and refuse silent overwrites of existing experiment artifacts.

## Provenance

The frozen RF165 campaign was generated from the private experimental archive at commit `7e66ee07dc5d4c59a7e7cce566909d0c38ef376f`. That identifier is retained inside the manifests as provenance and is not a public-history dependency. The curated repository starts with a clean history to avoid exposing experimental iterations unrelated to the article.

The canonical campaign contains 48 completed evaluations. Article-facing aggregates are stored in:

- `outputs/experiments/article_rf165/reports/final_summary.json`
- `outputs/experiments/article_rf165/reports/matching_sensitivity.csv`
- `outputs/experiments/article_rf165/reports/collection_summaries.csv`
- `outputs/experiments/article_rf165/audits/context_safety.json`
