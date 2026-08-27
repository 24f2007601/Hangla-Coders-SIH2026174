# GUI is a PySide6 desktop application

The monitoring GUI is a PySide6 desktop app (live video panel, status panel, activity panel, event log, START/PAUSE/STOP controls). The docs considered both Streamlit and PySide6; we chose PySide6 because it reads as a native standalone offline application — a better fit for the "deployable mission software" pitch.

**Considered options:** Streamlit (faster to build, works offline once installed, but feels less like a standalone product); raw OpenCV window overlay (zero dependencies but unpolished).

**Consequences:** more GUI dev time than Streamlit. For the PoC we keep the UI a functional skeleton: no heavy styling, UI reads pipeline output through the event/result manager, and no business logic lives inside widgets.