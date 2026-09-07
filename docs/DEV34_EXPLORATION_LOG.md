# Dev34 Exploration Log

> 历史记录：本日志中的 `deepseek-chat` 运行属于旧候选探索，不得用于当前 DeepSeek V4
> Flash 的 Dev 双轮或冻结发布证明。

## Active protocol

- Evaluate development cases in batches of 10.
- Modify the agent when a batch contains at least 3 genuine LLM-as-Judge failures.
- Judge false negatives are recorded separately and do not trigger judge-chasing changes.
- Limit one recurring issue family to 3 modification rounds. If it still fails, skip further modification and record it.
- A final partial batch with fewer than 10 cases is reported separately and does not independently satisfy the 3-in-10 modification trigger.
- After exploration, freeze one candidate and rerun all Dev34 without code, prompt, or configuration changes.

## Batch dev011-dev020 remediation

Trigger: genuine Judge failures in dev013, dev014, and dev016 (3/10).

### Round 1

- Harness: 1/3 strict pass; dev013 and dev016 were blocked before Planner by missing meal evidence.
- Judge: dev014 score 65, not reasonable.
- Root cause: generic meal completeness was incorrectly treated as hard readiness; named-candidate verification result was not user-visible.

### Round 2

- Harness: 2/3 strict pass; dev013 remained blocked by missing route evidence.
- Judge: dev014 score 46; dev016 score 75; both not reasonable.
- Root cause: verified candidate was later dropped by scheduling; fixed-event alias/nearby points created duplicate cross-city visits.

### Round 3 (final)

- Harness: 3/3 strict pass; p50 36.279s, p95 45.038s; 107,748 total tokens; no model switch, timeout, tool/API/state failure.
- Judge: 3/3 completed, average 64, 0/3 reasonable.
- dev013: genuine failure. All must-visits were present, but itinerary used a 20-minute fallback estimate where bound route evidence showed 57 minutes, and route order backtracked.
- dev014: genuine failure. Tianjin Museum exclusion was correctly explained, but the verified-suitable Italian Style Town was still lost during post-processing.
- dev016: Judge lodging critical is a false negative because the selected hotel came from a bound hotel artifact omitted by compact Judge facts. Genuine remaining non-critical failure: fixed-event day has only one scheduled activity. Fixed-event transfer buffer was correctly emitted (92 minutes each way; 13:28 departure, 18:32 city return).

Decision: three modification rounds exhausted. Skip further changes for these issue families and continue exploration with dev021-dev030.

## Batch dev021-dev030 remediation

Trigger: initial Judge review found genuine failures in dev021, dev022, dev023, and dev025 (4/10).

### Round 1

- Targeted harness (dev021/dev022/dev023/dev025): 2/4 strict pass; p50 62.0s, p95 64.2s; 232,600 total tokens.
- Judge: average 67.5, 1/4 reasonable. dev023 passed; dev021, dev022, and dev025 remained genuine failures.
- Root causes: replacement must-visit state was dropped in the LLM merge; total budget in the constraint tree was not consumed by the budget plan; required lodging with no grounded candidate did not fail closed.

### Round 2

- Targeted harness (dev021/dev022/dev025): 3/3 strict pass; p50 49.193s, p95 56.365s; 195,101 total tokens; no switch, timeout, tool, routing, state, or API failure.
- Full unit suite: 556 passed, 1 skipped.
- Judge: scores 76/88/83, average 82.33; 1/3 reasonable as emitted.
- dev025: genuine pass. Budget constraint is propagated, grounded lodging is selected, and restaurant facts are aggregated across bound artifacts.
- dev022: Judge false negative. All hard constraints and opening hours pass; the only critical issue is missing accommodation, but the user never requested lodging. The Judge rubric explicitly says generic multi-day lodging absence alone is not critical.
- dev021: one genuine residual failure. The budget only fits the provider's low estimate scenario and walking inside large attractions remains unverified for a 72-year-old, despite point-to-point taxi fallback for unknown transit walking.

Decision: only 1 genuine failure remains after round 2, below the current >=3 modification threshold. Do not enter round 3; record dev021 and continue to the final exploratory cases.

## Final partial batch (lh_5_001, lh_8_001, lh_12_001, lh_12_002)

This is a 4-case tail, not a complete 10-case batch. It is reported in full but does not independently trigger another code-modification round under the active 3-in-10 rule.

### Infrastructure remediation

- The first run exposed recursive artifact propagation: each turn embedded the full previous itinerary including its own `domain_inputs.previous_itineraries`, growing individual pending artifacts as large as 2.5 GB.
- The previous-itinerary payload is now bounded and ancestry is stripped. The longest targeted case shrank from about 1.9 GB to about 12 MB (roughly 99.4%).
- Full unit suite after the remediation: 559 passed, 1 skipped.

### Content remediation round 1

- Harness: 0/4 strict; p50 269.473s, p95 323.766s; 1,407,136 total tokens; no timeout.
- Judge: scores 49/61/37/52, average 49.75, 0/4 reasonable.
- Improvements: removed POIs stayed removed, the Shanghai fixed dinner became a restaurant, and the People's Square hotel was grounded correctly.

### Content remediation round 2 and final tail assessment

- `lh_5_001`: harness gating/state/hard-constraint/critic pass; 66.541s; 107,631 tokens. Judge 62, not reasonable. Genuine failures: East-West-Lake lodging evidence unavailable and the elderly walking cap remains unverified.
- `lh_8_001`: harness gating/state/critic pass, hard-constraint annotation fail; 229.231s; 330,385 tokens. Judge 52, not reasonable. Genuine failures: seafood candidate survives the no-seafood constraint, Siming lodging evidence unavailable, and high budget estimate exceeds the cap.
- `lh_12_001`: harness gating/hard/state fail only on return-deadline representation (`17:00` versus canonical ISO), critic fail; 231.231s; 350,929 tokens. Judge 34, not reasonable. Budget now fits, removed Liangzhu stays removed, and the free-time window is preserved; genuine failures remain around explicit East Station return, sparse days, lodging evidence, and explicit rendering of the free-time block.
- `lh_12_002`: harness gating/state/hard pass, critic fail; 317.237s; 421,367 tokens. Judge 66, not reasonable. Budget and People's Square lodging now fit and Disney stays removed; genuine failures remain around sparse days and the fixed dinner location being duplicated as a must-visit attraction.
- Judge calls completed on SiliconFlow `Qwen/Qwen3.5-397B-A17B`; scores averaged 53.5 across the four cases. Judge latency was 37.896s, 53.071s, 53.942s, and 41.678s, with 35,262 total Judge tokens.
- No DeepSeek timeout, model switch, tool/API failure, state propagation failure, or routing failure occurred in the final round. The only initial Judge connection errors were sandbox-network failures and succeeded after authorized retry.

Decision: record all four genuine failures. Because this tail contains only four cases, do not start a third content-modification round under the active 3-in-10 rule. Freeze the current candidate and proceed to a clean, full Dev34 rerun with no intervening code, prompt, or configuration changes.

## Frozen full Dev34 rerun

- Run: `data/eval/product/runs/dev34_frozen_full_20260813a`.
- Freeze integrity: key implementation diff SHA-256 `19e7078a17ce07f7a417d1b96fa71ea6590788fab8f5032ecf67bf4ced6b9a65` before and after the run; runtime prompt fingerprint `472123f26358855597d96d8d28830079444625daddd4bee3c72cf5910f6624de`.
- Agent model: paid DeepSeek `deepseek-chat`. All 34 cases completed exactly once; no retry, timeout, model-call error, model switch, or unavailable model.
- Harness: strict 20/34 (58.8%), gating 33/34 (97.1%), grounding 34/34, authorization 34/34, hard constraints 26/34 (76.5%), critic 25/34 (73.5%). Long-horizon state pass 2/4 with mean accuracy 98.17%.
- Harness latency: p50 33.097s, p95 216.462s. Agent usage: 1,834,299 tokens total.
- Judge: SiliconFlow `Qwen/Qwen3.5-397B-A17B`; 19 applicable, 19 completed, 15 not applicable, 0 missing. Average score 65.11; 6/19 reasonable; 13/19 emitted critical issues. Judge usage 143,026 tokens; mean latency 51.433s.

### Frozen batch review

- dev001-dev010: 5 applicable; scores 82/90/55/83/45; 2 reasonable. Genuine failures: dev003 (invalid POI/category matches and missed coffee/seaside intent), dev004 (hotel location makes daily schedule infeasible), dev005 (wheelchair/accessibility evidence and route/POI mismatch). Result: 3 genuine failures in 10, threshold reached.
- dev011-dev020: 6 applicable; scores 88/40/64/49/75/78; 2 reasonable as emitted. Genuine failures: dev014 (closed museum retained), dev016 (fixed-event transfer not rendered into schedule), dev017 (must-visit dropped). dev018 is marked Judge false negative: its only critical finding is a 110-metre leg labelled public transport; this is a route-mode label defect, not a critical plan failure. Result after trace review: 3 genuine failures in 10, threshold reached.
- dev021-dev030: 4 applicable; scores 85/54/63/91; 2 reasonable. Genuine failures: dev023 (3-day request collapsed to 2 days and history coverage missed), dev025 (required lodging evidence unavailable). Result: 2 genuine failures in 10, below threshold.
- Final partial batch: scores 29/39/70/57; 0 reasonable. All four are genuine failures: return/deadline and incomplete days (`lh_12_001`), duration-state collapse (`lh_12_002`), lodging/walking evidence (`lh_5_001`), and dietary/lodging evidence (`lh_8_001`). This partial batch remains separately recorded and is not treated as a complete 10-case trigger.

### Frozen-candidate decision

The candidate is reproducibly evaluated but does not pass the active acceptance rule: two complete 10-case batches each contain 3 genuine Judge failures. No code, prompt, or configuration was changed during the frozen run. Dev34 exploration is complete; further remediation should begin as a new candidate cycle, preserving this run as the immutable baseline.
