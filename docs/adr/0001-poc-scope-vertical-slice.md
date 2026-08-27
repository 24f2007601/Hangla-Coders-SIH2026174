# 5-day PoC is a skeleton + one full vertical slice

The full vision (YOLO detection, MediaPipe pose+hands, XGBoost step classifier, FSM validation, offline TTS, streaming, ONNX edge deployment) cannot be finished in 5 days. We decided the PoC is a clean modular skeleton plus ONE working vertical slice: webcam → MediaPipe pose/hands → hand-object distance features → XGBoost step classifier → FSM sequence validation of a simple toy protocol → structured JSON log → PySide6 GUI. YOLO fine-tuning, streaming, voice TTS, and ONNX export are deferred, but the architecture keeps them replaceable (interface-first design) so they slot in later.

**Considered options:** full pipeline with all subsystems wired (too risky for 5 days); architecture skeleton only (does not prove the product differentiator). We chose the vertical slice because it demonstrates the protocol-aware validation loop — the actual product function — end to end.

**Consequences:** the prototype uses heuristics-free perception from pretrained MediaPipe, a small trained step classifier, and a hand-rolled FSM. Detection is mock/stub where YOLO would go.