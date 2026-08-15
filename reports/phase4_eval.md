# BankShield AI -- Phase 4: Investigation Agent Evaluation

Mode: **offline (AutoFakeLLMClient)**. 13 investigations across Tier 1/2/3 alerts (see POL-AML-001 SS2.1 for tier definitions).

| Metric | Value |
|---|---|
| Citation correctness | 1.000 |
| Evidence faithfulness | 1.000 |
| Tool-call success rate | 1.000 |
| Mean latency | 82.1 ms |
| Mean estimated cost | $0.00000 |
| Total estimated cost | $0.00000 |

## Per-transaction results

| Transaction | Tier | Citation correctness | Evidence faithfulness | Tool success | Tool calls | Citations claimed | Disposition |
|---|---|---|---|---|---|---|---|
| `185dd3d4f902...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `6e92e2dce5c1...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `745434a93942...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `bc8a01983a60...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `cc3ac6ca026d...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `3203f1017b11...` | tier_2 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `42f316953580...` | tier_2 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `a68ca2c2cb77...` | tier_2 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `0002881e9aa6...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `0071206c32b1...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `00d84cf66e19...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `0130520a5469...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `01a3156aa794...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |

## Metric definitions

- **Citation correctness** -- of every `[DOC-ID §section]` citation the narrative makes, the fraction that both exist in the policy corpus and were actually retrieved during the investigation (payload-building search or an explicit `search_policy` call) -- a citation to a real section the agent never retrieved still counts as incorrect (ungrounded).
- **Evidence faithfulness** -- whether numeric/boolean claims the narrative makes (e.g. the stated risk score, ATO-pattern claims) match the actual payload values.
- **Tool-call success rate** -- fraction of tool calls that did not return an error.
- **Latency** -- wall-clock time for the full agent loop, including tool execution.
- **Estimated cost** -- list-price token cost estimate (`config.BEDROCK_PRICE_PER_1K_TOKENS`); 0 for the offline fake client, which reports 0 tokens.
