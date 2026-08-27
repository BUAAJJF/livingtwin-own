# `results/ra_hw0/`

| file | what |
|---|---|
| `audit.json` | the read-only capability audit, offline (no CAN opened) |
| `audit_hw_attempt.json` | the same with `--hardware`: refused, `can0` is not present |
| `preview.json` | every trajectory's duration, amplitude, envelope and simulated table clearance |
| `sessions/H0_demo.*` | a 30 s dry-run H0 session: append-only raw log, metadata, and the validator's derived file |

**No arm has been touched.** `audit_hw_attempt.json` is what the hardware path
does when there is nothing to talk to: it reports `NOT_FOUND` and opens
nothing. `configs/ra_hw0_safety_limits.json` carries `motion_authorised:
false` and 16 blocking `UNKNOWN` rows.
