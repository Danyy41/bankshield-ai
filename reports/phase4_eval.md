# BankShield AI -- Phase 4: Investigation Agent Evaluation

Mode: **offline (AutoFakeLLMClient)**. 10 investigations across Tier 1/2/3 alerts (see POL-AML-001 SS2.1 for tier definitions).

| Metric | Value |
|---|---|
| Citation correctness | 1.000 |
| Evidence faithfulness | 1.000 |
| Tool-call success rate | 1.000 |
| Mean latency | 64.5 ms |
| Mean estimated cost | $0.00000 |
| Total estimated cost | $0.00000 |

## Per-transaction results

| Transaction | Tier | Citation correctness | Evidence faithfulness | Tool success | Tool calls | Citations claimed | Disposition |
|---|---|---|---|---|---|---|---|
| `849e00a5ee8c...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `9f91426ecfe9...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `b5ec122e5cf2...` | tier_1 | 1.00 | 1.00 | 1.00 | 5 | 1 | confirmed_fraud |
| `8c8f9bd1eb11...` | tier_2 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `ba48a8194c8b...` | tier_2 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `bb00b55772b6...` | tier_2 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `0001dc147bdb...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `006f881c9b73...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `00c9e8e341e7...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |
| `0130f65f9b00...` | tier_3 | 1.00 | 1.00 | 1.00 | 4 | 1 | inconclusive_monitor |

## Metric definitions

- **Citation correctness** -- of every `[DOC-ID §section]` citation the narrative makes, the fraction that both exist in the policy corpus and were actually retrieved during the investigation (payload-building search or an explicit `search_policy` call) -- a citation to a real section the agent never retrieved still counts as incorrect (ungrounded).
- **Evidence faithfulness** -- whether numeric/boolean claims the narrative makes (e.g. the stated risk score, ATO-pattern claims) match the actual payload values.
- **Tool-call success rate** -- fraction of tool calls that did not return an error.
- **Latency** -- wall-clock time for the full agent loop, including tool execution.
- **Estimated cost** -- list-price token cost estimate (`config.BEDROCK_PRICE_PER_1K_TOKENS`); 0 for the offline fake client, which reports 0 tokens.
