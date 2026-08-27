# Standards — Best Practices & Guidelines

Coding, logging, documentation, and reporting standards for this repository. The goal is a clean, replaceable, observable system that a 4–6 person team can build independently in 5 days without stepping on each other.

## Coding standards

- Type hints wherever practical. Small modules. Clear interfaces. Composition over inheritance. Dependency injection where useful.
- **No global state.** No hard-coded paths. Prefer configuration through Pydantic settings (`configs/default.yaml`).
- No business logic inside UI widgets; no DB logic inside ML classes.
- Keep components replaceable behind `Protocol`s (see `docs/architecture.md`).
- Prefer existing patterns over introducing new libraries. Do not add dependencies without justification.
- **Do NOT add comments unless asked, or unless they document a non-obvious algorithm.** Let code and interfaces speak.
- Follow the domain vocabulary in `CONTEXT.md` — never mix in the avoided words.

## Code quality tooling

- **Ruff** for linting; **Black** for formatting; **mypy** where practical.
- CI (`ci.yml`): install deps → Ruff → pytest → fail on failure.
- Run `ruff check .` and `pytest` before committing.

## Logging & error handling

- Structured logging. Log: pipeline startup, component init, frame/inference/database failures, shutdown.
- **Avoid excessive per-frame logging** — no spam.
- The pipeline must not crash because one frame fails — catch and log frame-level failures.
- **Never silently swallow errors.** Failures must be observable. If you catch, you log/record and continue only when that is the correct behavior.

## Testing standards

- Unit tests: pose normalization, feature extraction, dummy classifier, pipeline execution, database repository.
- At least one integration test: dummy frame → pipeline → classification → result — no GPU, no camera. Runnable with `pytest`.
- Tests must run offline and on CPU.

## Documentation standards

- `CONTEXT.md` is a **glossary and nothing else** — no implementation details, no spec, no decisions.
- Decisions that are hard to reverse, surprising, or a real trade-off → ADR in `docs/adr/` (`NNNN-short-title.md`). Record *that* a decision was made and *why*.
- `AGENTS.md` is the entry point; focused reference docs live in `docs/` (`architecture.md`, `implementation-roadmap.md`, `standards.md`, `success-criteria.md`).
- README must accurately reflect Implemented vs Not-Yet-Implemented. Never exaggerate.

## Reporting integrity (non-negotiable)

Never invent or assert without evidence:

- accuracy, FPS, latency, or dataset size
- that a model is trained, a feature is implemented, or a prototype exists
- "mission-ready", successful microgravity testing, or full orientation-agnostic 3D HMR

Always distinguish status in any report, slide, or PR description:

| Label | Means |
|---|---|
| Implemented | built and present in the codebase |
| Tested | verified by tests or a run |
| Planned | scheduled work, not started |
| Proposed | a suggestion, not committed |
| Target | a goal/number we aim for |
| Assumption | taken as true without verification |

## Git hygiene

- Never commit: virtualenvs, databases, model binaries, raw datasets, generated files, secrets, `.env`.
- `.gitignore` covers the above. Add `.gitkeep` files for empty dirs (data/, models/, notebooks/).
- Keep commits small and focused; run `reviewer` before committing or opening a PR.

## Agent workflow (this repo)

1. `grill` (or `grill-with-docs`) before writing code — stress-test the plan.
2. `domain-modeler` records new terms/ADRs inline.
3. `build` implements per `AGENTS.md` + `docs/architecture.md`.
4. `reviewer` reviews the diff; fix findings; re-run pytest + ruff.

## General guardrails

- This is a 5-day PoC: no over-engineering, microservices, Docker/K8s, or significant UI styling time.
- Keep components independently buildable so multiple team members can work in parallel.