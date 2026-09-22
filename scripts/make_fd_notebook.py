"""One-off: build notebooks/tm_bakeoff_fd_colab.ipynb from the BTB notebook's
shape, swapped to the football-data.co.uk (modern) moves-classifier pipeline.
Run once locally, not part of the pipeline.
"""
import json

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [],
}


def md(src):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": src.splitlines(keepends=True)})


md("""# Tsetlin on modern data (football-data.co.uk)

**Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
Run is ~5-10 min (much smaller dataset than the BTB notebook).

No API key needed -- football-data.co.uk is a free public CSV site.

**Read this first:** a local CPU baseline check already found **no signal at
all** in this framing (all baselines 0.48-0.49 ROC-AUC, chance = 0.50) -- see
FINDINGS.md v0.9. Football-data only has 2
snapshots per match (opening capture, closing), so "which book moves next" here
just means "did the price differ at close from open over a 1-3 day gap" -- a
much weaker signal than the BTB notebook's true hourly snapshot-to-snapshot
task. This run exists to confirm the Tsetlin Machine doesn't find anything the
simpler models missed, not because a result is expected.""")

code("""# 1. GPU + code + baseline deps
!nvidia-smi -L || echo 'NO GPU — Runtime > Change runtime type > T4 GPU'
import os
if not os.path.isdir('tsetlin-market-lab'):
    !git clone --depth 1 https://github.com/naibwedi/tsetlin-market-lab.git
os.chdir('/content/tsetlin-market-lab')
!git pull -q
!pip -q install pandas pyarrow pyyaml python-dotenv scikit-learn xgboost lightgbm requests
print('cwd', os.getcwd())""")

code("""# 2. Pull football-data.co.uk -> panel -> features -> the 7 baselines
import glob
if not glob.glob('data/features_fd/X.parquet'):
    !python -m src.ingest.footballdata --seasons 2122 2223 2324 2425 2526 --divs E0 SP1 D1 I1 F1 N1 P1
    !python -m src.panel.build_panel --config config/features.fd.moves.yaml
    !python -m src.features.booleanize --config config/features.fd.moves.yaml
!python -m src.models.bakeoff --config config/bakeoff.fd.yaml
print('\\n' + open('results/summary.md').read())""")

code("""# 3. Tsetlin Machine on the GPU (Python 3.11 venv; tmu has no 3.13 wheel)
import os
if not os.path.isfile('/content/tm311/bin/python'):
    !pip -q install uv
    !UV_VENV_CLEAR=1 uv venv /content/tm311 --python 3.11 --quiet
    !uv pip install -q --python /content/tm311/bin/python \\
        'numpy<2' 'scikit-learn==1.5.2' pandas pyarrow pyyaml python-dotenv tmu pycuda
!cd /content/tsetlin-market-lab && git pull -q
!cd /content/tsetlin-market-lab && /content/tm311/bin/python -m scripts.tm_run \\
    --config config/bakeoff.fd.yaml --out-prefix tm_fd""")

code("""# 4. The Tsetlin result + the rules it learned
print(open('results/tm_fd_result.json').read())
print('\\n--- clauses ---')
print(open('results/tm_fd_clauses.txt').read())""")

md("""### After this run

Download `results/tm_fd_result.json` and `results/tm_fd_clauses.txt` from the
Colab file browser and drop them in the repo's `results/` folder, then commit.
Compare `tm_fd_result.json`'s `roc_auc` against the baseline table above and
the BTB result in `results/tm_result.json` (0.742) -- if it's near 0.5 like the
baselines, that closes the "has TM even been tried on modern data" question
with a clean no-signal answer rather than an open gap.""")

with open("notebooks/tm_bakeoff_fd_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/tm_bakeoff_fd_colab.ipynb")
