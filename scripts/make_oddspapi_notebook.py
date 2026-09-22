"""One-off: build notebooks/tm_bakeoff_oddspapi_colab.ipynb.

Unlike the BTB/football-data notebooks, this one does NOT re-pull data from
the API in Colab -- the boolean feature matrix (data/features_oddspapi/) was
already built locally from the 50-match ingest and is uploaded as a small zip
instead, so this run costs zero extra OddsPapi requests. Run once locally,
not part of the pipeline.
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


md("""# Tsetlin on OddsPapi's real sub-hourly data

**Runtime → Change runtime type → T4 GPU**, then run cells 1-2, then
**upload `features_oddspapi.zip`** when cell 3 prompts for it, then run the rest.
Total run is ~5-10 min.

**Why this run matters:** every other check this session used baselines only
(XGBoost, logistic). The Tsetlin Machine -- this project's actual subject --
has never touched OddsPapi's real sub-hourly price data. Baselines already
hit **XGBoost 0.769 / logistic 0.768 AUC** on it, edging past the 2015-16
BTB result (0.765) -- the strongest result in the whole project. This
notebook finds out whether TM matches that, and what clauses it learns.

**No API key needed.** The boolean feature matrix was already built locally
from a 50-match ingest (`config/features.oddspapi.yaml`) -- uploading it
costs zero additional OddsPapi requests (the free tier is 250/month).""")

code("""# 1. GPU + code + baseline deps
!nvidia-smi -L || echo 'NO GPU — Runtime > Change runtime type > T4 GPU'
import os
if not os.path.isdir('tsetlin-market-lab'):
    !git clone --depth 1 https://github.com/naibwedi/tsetlin-market-lab.git
os.chdir('/content/tsetlin-market-lab')
!git pull -q
!pip -q install pandas pyarrow pyyaml python-dotenv scikit-learn xgboost lightgbm
print('cwd', os.getcwd())""")

code("""# 2. Upload features_oddspapi.zip (from the local run) and unzip it
import glob
if not glob.glob('data/features_oddspapi/X.parquet'):
    from google.colab import files
    print('Select features_oddspapi.zip when prompted...')
    uploaded = files.upload()
    zip_name = next(iter(uploaded))
    !unzip -oq "{zip_name}" -d data/
assert glob.glob('data/features_oddspapi/X.parquet'), 'features_oddspapi/X.parquet still missing'
print('features ready:', glob.glob('data/features_oddspapi/*'))""")

code("""# 3. The 7 baselines, for context (should reproduce XGBoost ~0.769 / logistic ~0.768)
!python -m src.models.bakeoff --config config/bakeoff.oddspapi.yaml
print('\\n' + open('results/summary.md').read())""")

code("""# 4. Tsetlin Machine on the GPU (Python 3.11 venv; tmu has no 3.13 wheel)
import os
if not os.path.isfile('/content/tm311/bin/python'):
    !pip -q install uv
    !UV_VENV_CLEAR=1 uv venv /content/tm311 --python 3.11 --quiet
    !uv pip install -q --python /content/tm311/bin/python \\
        'numpy<2' 'scikit-learn==1.5.2' pandas pyarrow pyyaml python-dotenv tmu pycuda
!cd /content/tsetlin-market-lab && git pull -q
!cd /content/tsetlin-market-lab && /content/tm311/bin/python -m scripts.tm_run \\
    --config config/bakeoff.oddspapi.yaml --out-prefix tm_oddspapi""")

code("""# 5. The Tsetlin result + the rules it learned
print(open('results/tm_oddspapi_result.json').read())
print('\\n--- clauses ---')
print(open('results/tm_oddspapi_clauses.txt').read())""")

md("""### After this run

Download `results/tm_oddspapi_result.json` and `results/tm_oddspapi_clauses.txt`
from the Colab file browser and send them back (paste the JSON, or attach the
files) so the result and clauses can be written up in `results/FINDINGS.md`
alongside the baseline numbers. Compare `tm_oddspapi_result.json`'s `roc_auc`
against XGBoost's 0.769 here and the original BTB TM result (0.742,
`results/tm_result.json`) -- if it's in the same range as the baselines, that
closes the "has TM been tried on the best data" question with a real answer.""")

with open("notebooks/tm_bakeoff_oddspapi_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/tm_bakeoff_oddspapi_colab.ipynb")
