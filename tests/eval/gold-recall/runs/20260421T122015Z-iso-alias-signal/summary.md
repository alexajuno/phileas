# Recall eval: iso-alias-signal

- **Timestamp:** 20260421T122015Z
- **Queries:** 3

## Global metrics

| metric | value |
| -- | --: |
| top_k_hit_rate | 1.000 |
| zero_result_rate | 0.000 |
| mean_rank (of hits) | 1.00 |

## By tag

| tag | n | hit_rate | zero_result_rate |
| -- | --: | --: | --: |
| alias-resolution | 3 | 1.000 | 0.000 |
| cross-lingual | 3 | 1.000 | 0.000 |
| positive-control | 1 | 1.000 | 0.000 |
| punctuation-regression | 1 | 1.000 | 0.000 |

## Per-query

| id | hit | rank | zero | tags |
| -- | :-: | --: | :-: | -- |
| alias-gap-chi-01 | ✓ | 1 |  | alias-resolution, cross-lingual |
| alias-present-01 | ✓ | 1 |  | alias-resolution, cross-lingual, positive-control |
| alias-present-punct-01 | ✓ | 1 |  | alias-resolution, cross-lingual, punctuation-regression |
