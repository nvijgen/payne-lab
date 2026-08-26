# Protein Corona Prediction

Predicting protein corona composition from nanoparticle and protein features. Random forest models over 6768 protein-nanoparticle pairs (376 serum proteins, 11 nanoparticles) predict both which proteins are enriched on a particle and how abundant they are, and the notebooks identify which of the 84 features drive those predictions.

## Setup

Python 3.12. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
uv sync                      # create .venv from uv.lock
uv run jupyter lab           # or: uv run jupyter notebook
```

**With pip:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

`requirements.txt` is generated from the lockfile, so the two paths install the same versions. Regenerate it after changing dependencies:

```bash
uv export --no-hashes --no-emit-project --no-dev > requirements.txt
```

## Notebooks

| Notebook | Report | Contents |
| --- | --- | --- |
| [`notebooks/unified.ipynb`](notebooks/unified.ipynb) | [html](reports/unified_report.html) | Main analysis: data transformations, bootstrap feature importance (RFR/RFC), SHAP, RFECV feature selection, the 6-feature parsimonious model, PDP/ICE, and protein clustering (detection vs enrichment response). Exports the intermediate tables in `notebooks/data/`. |
| [`notebooks/importance_significance.ipynb`](notebooks/importance_significance.ipynb) | [html](reports/importance_significance_report.html) | Significance testing for feature importance: bootstrap stability, Altmann permutation null, out-of-fold permutation importance, and pairwise rank stability. |
| [`notebooks/reviewer_response.ipynb`](notebooks/reviewer_response.ipynb) | [html](reports/reviewer_response_report.html) | Peer-review response analyses: leave-one-nanoparticle-out validation, CV grouping schemes, serum abundance as predictor vs label, evidence for dropping PEI, full performance metrics with confidence intervals, and cluster-count/stability checks. |

`notebooks/corona.py` holds the setup the three notebooks share (data loading and the `enriched` label, feature columns, RF hyperparameters, the parsimonious feature sets, and the zeta-sweep helpers used by the clustering). Its `RandomForestClassifier`/`RandomForestRegressor` fit in parallel but predict single-threaded, so every exported number is bit-reproducible run to run. Run the notebooks with `notebooks/` as the working directory so it is importable. `notebooks/data/` holds the CSV tables exported by these notebooks and consumed by the figures and reports.

The HTML reports in `reports/` are rendered from the executed notebooks (run the notebook first, then convert it). To regenerate them:

```bash
cd notebooks
for nb in unified importance_significance reviewer_response; do
    jupyter nbconvert --to html --output-dir ../reports --output "${nb}_report.html" "$nb.ipynb"
done
```
