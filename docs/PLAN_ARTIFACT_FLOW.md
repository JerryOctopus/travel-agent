# Plan Artifact delivery flow

```text
structured TravelProfile + explicitly bound evidence artifact IDs
                              │
                              ▼
Planner: plan_and_critique (bounded deterministic loop, max 3)
  ├─ itinerary + critic
  ├─ fixed-event / return / lodging / budget / evidence bindings
  └─ validation_result (final deterministic invariant set)
                              │
                         plan Artifact v1
                              │
                              ▼
Semantic Reviewer (full itinerary only, at most once)
  ├─ pass/critical ───────────────────────────────┐
  └─ recoverable                                  │
        │                                         │
        ▼                                         │
one bounded repair wave                           │
  ├─ optional one domain worker                   │
  └─ Planner creates Artifact v2                  │
       parent_plan_artifact_id = v1               │
       repair_targets / applied_changes           │
       unresolved_changes / validation_result     │
        │                                         │
        ▼                                         │
deterministic critic + constraints + schedule/route
+ budget + evidence binding rerun (no second Reviewer)
        │                                         │
        └──────────────────────┬──────────────────┘
                               ▼
Renderer Gate
  ├─ exact plan_artifact_id
  ├─ latest itinerary in the request
  ├─ allowed producer
  ├─ validation_result.passed for successful delivery
  └─ unresolved repair/error => incomplete, never “已排好/自动修正成功”
```

The artifact record envelope owns the artifact ID. Repair never mutates its
parent; provenance is represented by `parent_plan_artifact_id` in the child
payload. User-visible repair claims are sourced only from `applied_changes`.
