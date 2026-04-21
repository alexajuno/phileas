# Recall eval: baseline

- **Timestamp:** 20260421T120234Z
- **Queries:** 2

## Global metrics

| metric | value |
| -- | --: |
| top_k_hit_rate | 0.500 |
| zero_result_rate | 0.500 |
| mean_rank (of hits) | 1.00 |

## By tag

| tag | n | hit_rate | zero_result_rate |
| -- | --: | --: | --: |
| alias-resolution | 2 | 0.500 | 0.500 |
| cross-lingual | 2 | 0.500 | 0.500 |
| positive-control | 1 | 0.000 | 1.000 |

## Per-query

| id | hit | rank | zero | tags |
| -- | :-: | --: | :-: | -- |
| alias-gap-chi-01 | ✓ | 1 |  | alias-resolution, cross-lingual |
| alias-present-01 | ✗ | — | ✓ | alias-resolution, cross-lingual, positive-control |
