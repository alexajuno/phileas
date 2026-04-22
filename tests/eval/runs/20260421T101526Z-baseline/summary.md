# Eval run: baseline

- **Timestamp:** 20260421T101526Z
- **Gold cases:** 38
- **Repeats per case:** 1
- **Provider:** auto / **model:** anthropic/claude-haiku-4-5

## Global metrics (mean ± stdev across repeats)

| metric | mean | stdev | target |
| -- | --: | --: | --: |
| precision | 0.079 | 0.000 | 0.7 |
| recall | 0.300 | 0.000 | 0.7 |
| noise_rate | 1.000 | 0.000 | 0.1 |
| over_extraction_rate | 0.000 | 0.000 | 0.3 |
| miss_rate | 0.000 | 0.000 | 0.1 |
| non_english_rate | 0.289 | 0.000 | 0.0 |

## Graph metrics (informational — no gates until baseline)

| metric | mean | stdev |
| -- | --: | --: |
| entity_precision | 0.000 | 0.000 |
| entity_recall | 0.000 | 0.000 |
| entity_type_consistency | 0.000 | 0.000 |
| mean_entities_per_memory | 0.000 | 0.000 |
| relationship_precision_strict | 0.000 | 0.000 |
| relationship_recall_strict | 0.000 | 0.000 |
| relationship_recall_endpoint | 0.000 | 0.000 |
| mean_relationships_per_memory | 0.000 | 0.000 |

## Per-stratum

| stratum | n | precision | recall | mean_predicted_per_case |
| -- | --: | --: | --: | --: |
| coding-english | 8 | 0.000 | 0.000 | 1.00 |
| coding-life-mix | 5 | 0.000 | 0.000 | 1.00 |
| memory-request | 3 | 0.333 | 0.333 | 1.00 |
| short | 6 | 0.000 | 0.000 | 1.00 |
| system-reminder-heavy | 5 | 0.400 | 0.667 | 1.00 |
| trivial | 2 | 0.000 | 0.000 | 1.00 |
| vn-conversational | 6 | 0.000 | 0.000 | 1.00 |
| vn-en-mix | 3 | 0.000 | 0.000 | 1.00 |

## Latency

- p50: 5621 ms
- p95: 8983 ms
