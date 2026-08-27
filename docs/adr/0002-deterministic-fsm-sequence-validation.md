# Sequence validation is a deterministic FSM, not a learned model

The step classifier outputs discrete predicted step labels; validating their order against a known protocol is deterministic logic. We decided to hand-roll a finite state machine (a dict/graph of states + transition rules) rather than train an HMM/learned sequence model. This is more debuggable, more reliable, and directly produces the `confirmed` / `skipped` / `repeated` / `out-of-sequence` outcomes the problem statement requires.

**Considered options:** the `transitions` library (adds visualization niceties, not required for a PoC); learned sequence models (unnecessary complexity). We chose the plain Python FSM.

**Consequences:** the FSM is the "intelligence" of the product and must be rock-solid. It reads only confirmed step events; it never sees raw frames.