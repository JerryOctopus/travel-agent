# Long-horizon multi-turn evaluation

The long-horizon cases are part of `production_v1.1`, expanding it from 180 to
192 cases.

## Coverage

- 12 independent cases and 100 user turns in total;
- `dev`: 4 cases for iteration;
- `core_frozen`: 4 cases and `challenge_frozen`: 4 cases; these must not be
  used for training, prompt tuning, or rule tuning;
- four 5-turn, four 8-turn, and four 12-turn conversations;
- constraint retention, replacement, deletion, distraction recovery, fixed
  events, transport changes, budget changes, and early-constraint recall;
- one partial structured-state assertion after every user turn, rather than
  final-answer-only scoring.

The primary diagnostic metrics are `turn_state_pass_rate` (all checkpoints in a
case pass) and `turn_state_accuracy` (mean checkpoint constraint score).

## Run

Offline smoke test:

```bash
PYTHONPATH=src python -m travel_agent.harness.cli \
  --suite agent-long-horizon --env offline --long-horizon-split dev --json
```

Real-agent evaluation:

```bash
PYTHONPATH=src python -m travel_agent.harness.cli \
  --suite agent-long-horizon --env real_agent --long-horizon-split frozen --json
```

Use `--limit N` for a smoke subset. The committed dataset is generated
deterministically by `python scripts/build_long_horizon_dataset.py`.
