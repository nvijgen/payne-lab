"""Setup shared by the notebooks in this folder. Run them with this folder as cwd."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 1234

INPUT_DIR = Path("../input_data")
DATA_DIR = Path("data")

rf_kwargs = dict(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


class _SerialPredict:
    """Fit in parallel, predict serially.

    Tree fitting is deterministic under n_jobs=-1, but parallel prediction sums the trees
    in whichever order the threads finish, so the last bit of each probability varies
    from call to call. Predicting with one thread makes it bit-reproducible.
    """

    def _serial(self, method, X):
        n_jobs, self.n_jobs = self.n_jobs, 1
        try:
            return method(X)
        finally:
            self.n_jobs = n_jobs

    def predict(self, X):
        return self._serial(super().predict, X)

    def predict_proba(self, X):
        return self._serial(super().predict_proba, X)


class RandomForestClassifier(_SerialPredict, ensemble.RandomForestClassifier):
    pass


class RandomForestRegressor(_SerialPredict, ensemble.RandomForestRegressor):
    pass


NOT_FEATURES = ["entry", "sample_num", "abundance", "npunid", "enriched"]

# 1-SD parsimonious set from the enrichment-classifier RFECV; PARSI_6 drops
# `ligand_pei`, which is collinear with `zeta_potential`.
PARSI_7 = [
    "abundance_controls",
    "secondary_structure_fraction_sheet",
    "zeta_potential",
    "ligand_pei",
    "nsp_secondary_structure_coil",
    "fraction_exposed_exposed_k",
    "frac_aa_a",
]
PARSI_6 = [f for f in PARSI_7 if f != "ligand_pei"]

ZETA_GRID = [-60, -40, -20, 0, 20, 40]


def load_data(input_dir=INPUT_DIR):
    """Read the workbook, normalise column names, add the `enriched` label."""
    df = pd.read_excel(sorted(Path(input_dir).glob("*.xlsx"))[0])
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_per_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    epsilon = df["abundance_controls"].drop_duplicates().nsmallest(2).iloc[-1]
    df["abundance_controls"] = df["abundance_controls"].replace(0, epsilon)
    return df.assign(
        enriched=lambda d: np.where(d.abundance / d.abundance_controls > 1, 1, 0)
    ).assign(abundance_controls=lambda d: np.log2(d.abundance_controls.values))


def feature_cols(df):
    return [c for c in df.columns if c not in NOT_FEATURES]


def protein_response_matrix(model, df, features=PARSI_6):
    """(n_proteins x len(ZETA_GRID)) matrix of P(class=1) as zeta_potential sweeps.

    Each protein's first row supplies the held-fixed values of the other features.
    """
    pids = df["entry"].unique()
    first = df.drop_duplicates("entry").set_index("entry").loc[pids, features]
    grid = pd.concat([first] * len(ZETA_GRID), ignore_index=True).astype(float)
    grid["zeta_potential"] = np.repeat(ZETA_GRID, len(pids))
    proba = model.predict_proba(grid)[:, 1].reshape(len(ZETA_GRID), len(pids)).T
    return np.asarray(pids), proba


def response_matrices(df, features=PARSI_6):
    """Detection (abundance > 0) and enrichment classifiers on `features`, each swept.

    Returns (protein_ids, resp_detect, resp_enrich); see `protein_response_matrix`.
    """
    x = df[features].astype(float)
    detected = (df["abundance"] > 0).astype(int)
    clf_detect = RandomForestClassifier(**rf_kwargs).fit(x, detected)
    clf_enrich = RandomForestClassifier(**rf_kwargs).fit(x, df["enriched"])
    pids, resp_detect = protein_response_matrix(clf_detect, df, features)
    _, resp_enrich = protein_response_matrix(clf_enrich, df, features)
    return pids, resp_detect, resp_enrich


def order_by_meanprob(resp, labels):
    """Relabel clusters 0..k-1 by ascending mean response probability."""
    prob_order = pd.Series(resp.mean(axis=1)).groupby(labels).mean().sort_values().index
    remap = {old: new for new, old in enumerate(prob_order)}
    return np.array([remap[lab] for lab in labels])


def cluster_response(resp, k=3, seed=RANDOM_SEED):
    """K-means on a response matrix, labels 0..k-1 by ascending mean response probability."""
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(resp)
    return order_by_meanprob(resp, labels)


def oof_permutation_importance(Model, x, y, groups, scoring, n_folds=10, n_repeats=10):
    """Out-of-fold permutation importance under GroupKFold (shuffled, RANDOM_SEED).

    Per fold: fit `Model(**rf_kwargs)` on train, then shuffle each column of the test set
    `n_repeats` times and record the drop in `scoring`. `x` is a DataFrame; returns an
    (n_folds * n_repeats, n_features) DataFrame, one row per (fold, repeat).
    """
    xv, yv = x.to_numpy(), np.asarray(y)
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    chunks = []
    for fold, (tr, te) in enumerate(gkf.split(xv, groups=groups), 1):
        model = Model(**{**rf_kwargs, "random_state": RANDOM_SEED + fold})
        model.fit(xv[tr], yv[tr])
        pi = permutation_importance(
            model,
            xv[te],
            yv[te],
            n_repeats=n_repeats,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
            scoring=scoring,
        )
        chunks.append(pi.importances.T)  # (n_repeats, n_features)
        print(f"  fold {fold}/{n_folds}")
    return pd.DataFrame(np.vstack(chunks), columns=x.columns)
