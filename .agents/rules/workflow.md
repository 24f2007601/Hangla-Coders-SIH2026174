# Workflow Rules — Hangla-Coders-SIH2026174

These rules apply to every agent, every session, every model. No exceptions.

---

## Environment

- **Python runtime:** 3.14.6
- **Package manager:** `uv` (v0.10.7) — always use `uv run`, never bare `python` or `pip`
- **Install command:** `uv sync --all-extras`
- **Package:** `bas-assistant` (installed as editable via `uv sync`)

---

## Quality Gate — Run After EVERY Change

Run this exact sequence after any source code modification. All three must pass before considering a change done:

```bash
uv run ruff check .                          # lint — must be clean
uv run black --check src scripts tests       # format — must be unchanged
uv run pytest --tb=short -q                  # tests — all 49 must pass
```

If any step fails → fix it before proceeding. Never leave a failing gate.

---

## Before Making Any Change

1. **Read the relevant source file(s)** before touching anything.
2. **Check `AGENTS.md`** for architecture rules if the change touches a module boundary.
3. **Use canonical terms** from `CONTEXT.md` — never use avoided words.
4. **Do not train any model** during development sessions — the step classifier defaults to `dummy` mode; this is intentional.

---

## Commands Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync --all-extras` |
| Run lint | `uv run ruff check .` |
| Run format check | `uv run black --check src scripts tests` |
| Run all tests | `uv run pytest --tb=short -q` |
| Run pipeline (dummy) | `uv run python scripts/run_pipeline.py --source dummy --pose dummy` |
| Run demo | `uv run python scripts/run_demo.py --source dummy` |
| Run dashboard | `uv run python scripts/run_dashboard.py` |
| Run benchmark | `uv run python scripts/benchmark.py` |

---

## Key Verified State (as of 2026-08-27)

| Check | Result |
|-------|--------|
| `uv sync --all-extras` | ✅ 54 packages installed |
| `pytest` | ✅ 49/49 passed |
| `ruff check .` | ✅ Clean |
| `black --check` | ✅ 56 files unchanged |
| Pipeline smoke run | ✅ JSONL session log written |

---

## What Is NOT Done (do not claim otherwise)

- **No XGBoost model trained** — `classifier.model_type` is `dummy`; no dataset recorded yet
- Live webcam + MediaPipe not tested on hardware
- YOLO fine-tuning, voice alerts, ONNX export, streaming — all deferred (per ADR-0001)

## What Is DONE (do not claim otherwise)

- Live webcam + MediaPipe tested on hardware
  
---

## Git Convention

- Personal branch name: `prith`
- Push command (first time): `git push -u origin prith`
- Never commit `.venv/`, `data/processed/session_*.jsonl`, `models/*.onnx` — all gitignored

---

## Reporting Integrity (non-negotiable)

Never invent accuracy, FPS, latency, or dataset size.
Always distinguish: `Implemented / Tested / Planned / Proposed / Target / Assumption`.
