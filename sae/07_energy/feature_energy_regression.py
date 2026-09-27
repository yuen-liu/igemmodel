"""Design-level OLS: dE_interaction ~ binder length + per-feature presence +
per-feature activation strength (z-scored among designs where present, 0 if
absent). HC3 robust SEs, BH-FDR across feature terms. numpy/scipy only
(Schrodinger python has no statsmodels)."""
import numpy as np, pandas as pd, yaml
from scipy import stats

df = pd.read_csv("energy_profile_combined.csv")
y_ = yaml.load(open("sae/results/run4/yamls/interface_features_energy_sample.yaml"), Loader=yaml.SafeLoader)
act = df.pivot_table(index="design_id", columns="feature_id", values="max_activation")
dE = df.drop_duplicates("design_id").set_index("design_id").loc[act.index, "dE_interaction"]
nres = pd.Series([float(y_["designs"][d]["n_binder_residues"]) for d in act.index], index=act.index)

def build(act, with_activation=True):
    X = {"binder_len_per10res": (nres - nres.mean()) / 10}
    for f in act.columns:
        present = act[f].notna()
        if present.mean() < 1:
            X[f"present_{f}"] = present.astype(float)
        if with_activation:
            a = act[f][present]
            z = pd.Series(0.0, index=act.index)
            if a.std() > 0:
                z[present] = (a - a.mean()) / a.std()
            X[f"act_{f}_perSD"] = z
    X = pd.DataFrame(X, index=act.index)
    return X

def ols_hc3(X, y):
    Xm = np.column_stack([np.ones(len(X)), X.values]); yv = y.values
    XtXi = np.linalg.pinv(Xm.T @ Xm)
    beta = XtXi @ Xm.T @ yv
    e = yv - Xm @ beta
    h = np.einsum("ij,jk,ik->i", Xm, XtXi, Xm)
    meat = Xm.T @ (Xm * (e**2 / (1 - h)**2)[:, None])
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    dof = len(yv) - Xm.shape[1]
    t = beta / se; p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (e**2).sum() / ((yv - yv.mean())**2).sum()
    adj = 1 - (1 - r2) * (len(yv) - 1) / dof
    out = pd.DataFrame({"coef": beta, "se": se, "p": p}, index=["intercept"] + list(X.columns))
    return out, r2, adj

for name, X in [("FULL (presence + activation)", build(act)), ("PRESENCE ONLY", build(act, False))]:
    res, r2, adj = ols_hc3(X, dE)
    feat = res.index.str.startswith(("present_", "act_"))
    res["q_BH"] = np.nan
    res.loc[feat, "q_BH"] = stats.false_discovery_control(res.loc[feat, "p"].values)
    print(f"\n=== {name}: n={len(dE)}, params={X.shape[1]+1}, R2={r2:.3f}, adj R2={adj:.3f}")
    print(res.round(3).to_string())

X = build(act)
c = X.corr().abs().where(~np.eye(X.shape[1], dtype=bool)).stack().sort_values(ascending=False)
print("\nLargest |correlations| between predictors:"); print(c.iloc[:8:2].round(2).to_string())
