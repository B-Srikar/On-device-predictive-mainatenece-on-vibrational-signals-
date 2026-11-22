Rules for this project:
- Never modify files in data/raw/. Write outputs to data/processed/ or outputs/.
- Never assume a file exists; check the workspace first.
- Never invent metrics. Run the code and report real outputs.
- All parameters (sampling rate, window size, overlap, EMA alpha) live in config.yaml.
- Split train/test by recording, never by random windows.
- Stop after each stage and wait for my confirmation.
- Keep the core method simple enough to port to C on an ESP32 (no heavy libraries in the core HI).
