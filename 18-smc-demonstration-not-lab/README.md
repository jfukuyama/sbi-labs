# Lecture 18: interactive ABC-SMC demonstrations

A Jupyter notebook with two interactive ABC-SMC examples: a one-parameter model with three modes and a two-parameter model with four. You draw proposals and move between generations by hand, choosing the kernel τ and the tolerances as you go.

## Install

You need Python 3.11 or later. From this folder:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```sh
.venv/bin/jupyter lab 18-abc-smc-demos.ipynb
```

Run the cells from the top. Once the packages are installed, no internet connection is needed. The notebook also works in classic Jupyter and in VS Code, as long as the kernel uses the environment above (it needs ipywidgets 8, plotly and anywidget).

If you edit `app.py` or `abcsmc.py`, restart the kernel to pick up the changes.

## Files

| file | contents |
|---|---|
| `18-abc-smc-demos.ipynb` | the lecture notebook |
| `app.py` | the widgets and plots |
| `abcsmc.py` | the two models and the ABC-SMC sampler |
| `checks/` | optional checks for after changing the code: `.venv/bin/python -m pytest checks/` and `.venv/bin/python checks/check_presets.py` |
