# Training Corpus (Demo)

Multi-source PDF corpus for domain-adaptive continued pretraining

## Summary

This dataset is a JSONL text corpus produced by a configuration-driven preprocessing pipeline. It is intended for domain adaptation via continued pretraining with a next-token prediction objective.

## Metadata

| Field | Value |
|---|---|
| Run name | training |
| Dataset ID | training-out/training |
| Version | 0.1.0 |
| License | internal |
| Language | en |
| Domain | organisational documents |
| Created (UTC) | 2026-01-29 19:07 UTC |

## Owners and contacts

| Name | Affiliation | Contact |
|---|---|---|
| Amsterdam University of Applied Sciences | Amsterdam University of Applied Sciences (AUAS) | your.name@hva.nl |

## Artifacts

| Field | Value |
|---|---|
| Config | config_training.yaml |
| Corpus | out/training/corpus.jsonl |
| Diagnostics | out/training/diagnostics.json |

## Intended use

**primary:** Domain-adaptive continued pretraining (next-token prediction).
**secondary:**
- Exploratory analysis of organisational writing style and vocabulary.
- Benchmarking extraction and cleaning recipes on heterogeneous PDFs.

## Out of scope

- Supervised instruction tuning.
- Safety-critical deployment without additional evaluation.

## Data sources

**description:** Example PDFs bundled with the repository for reproducible experiments.
**collection_method:** Provided as tutorial materials.

## Sensitive content

**pii_expected:** True
**pii_types:**
- email
- phone
**mitigation:** Regex-based masking in the privacy-aware and training-ready runs.

## Preprocessing notes

- PDF text extraction may introduce layout artefacts, ordering issues, or missing characters.
- Boilerplate removal is conservative and should be adapted per document type if needed.

## Evaluation notes

- No model performance results are reported in this tutorial.
- Quality is assessed via diagnostics and qualitative spot checks.

## Preprocessing recipe (from config)

| Field | Value |
|---|---|
| Normalization | collapse_whitespace=True, remove_boilerplate_lines=True |
| Cleaning | min_doc_chars=600, max_doc_chars=400000, max_nonalpha_ratio=0.6 |
| Segmentation | chunk_chars=1600, overlap_chars=200, min_segment_chars=250, max_segment_chars=2500 |
| Deduplication | enabled=True |
| PII masking | enabled=True, replacement_format=[{name}] |
| Training export (MLX) | enabled=True, dir=data/mlx, train_frac=0.9, seed=42 |

### Enabled masking patterns

| Pattern | Status |
|---|---|
| email | enabled, line_hint=no |
| phone | enabled, line_hint=yes |

## Dataset size and basic statistics

| Field | Value |
|---|---|
| Documents ingested | 3 |
| Documents after clean | 3 |
| Segments after dedup | 293 |
| Dedup removed | 0 |
| Corpus lines scanned | 293 |

### Text length distribution (characters)

| Field | Value |
|---|---|
| Min | 569 |
| Median (p50) | 1600 |
| Mean | 1592.1 |
| p95 | 1600 |
| Max | 1600 |
| Total characters | 466474 |

**Top doc_type values (from meta)**

| Value | Count |
|---|---:|
| report | 265 |
| transcript | 27 |
| letter | 1 |

**Top source values (from meta)**

| Value | Count |
|---|---:|
| un_demo | 293 |

### Masking prevalence (from meta)

| Field | Value |
|---|---|
| pii_masked=False | 0 |
| pii_masked=True | 293 |

## PII masking report (from diagnostics)

| Pattern | Matches |
|---|---:|
| email | 2 |
| iban | 0 |
| phone | 1 |

## Runtime environment

| Field | Value |
|---|---|
| Python | 3.9.16 |
| Platform | Darwin 25.1.0 |

## Citation

```bibtex
@misc{training_corpus_demo_2026,
  title        = {Training Corpus (Demo)},
  author       = {Amsterdam University of Applied Sciences},
  year         = {2026},
  howpublished = {Tutorial artifact},
}
```
