import pandas as pd, numpy as np, yaml
from scipy import stats
df = pd.read_csv("energy_profile_combined.csv")
y = yaml.load(open("sae/results/run4/yamls/interface_features_energy_sample.yaml"), Loader=yaml.SafeLoader)
nres = {d: v["n_binder_residues"] for d, v in y["designs"].items()}
des = df.drop_duplicates("design_id").set_index("design_id")[["dE_interaction","n_interface_hbonds","n_atoms_minimized"]]
des["n_binder_res"] = des.index.map(nres)
print("n designs", len(des))
print("Spearman dE vs binder residues: %.2f   vs n_atoms_minimized (interface size): %.2f" % (
    stats.spearmanr(des.dE_interaction, des.n_binder_res)[0], stats.spearmanr(des.dE_interaction, des.n_atoms_minimized)[0]))
# size-adjusted dE: residual of dE ~ n_atoms_minimized
b = np.polyfit(des.n_binder_res, des.dE_interaction, 1)
des["dE_resid"] = des.dE_interaction - np.polyval(b, des.n_binder_res)
print("linear fit dE = %.3f * n_binder_res + %.1f ; R2=%.2f" % (b[0], b[1], np.corrcoef(des.n_binder_res, des.dE_interaction)[0,1]**2))
act = df.pivot_table(index="design_id", columns="feature_id", values="max_activation")
rows = []
for f in act.columns:
    a = act[f]; present = a.notna()
    dp, da = des.loc[present[present].index], des.loc[present[~present].index]
    r = {"feature": f, "n_present": int(present.sum())}
    if len(da) >= 10:
        r["dE_present_minus_absent"] = dp.dE_interaction.median() - da.dE_interaction.median()
        r["p_presence_raw"] = stats.mannwhitneyu(dp.dE_interaction, da.dE_interaction).pvalue
        r["p_presence_sizeadj"] = stats.mannwhitneyu(dp.dE_resid, da.dE_resid).pvalue
        r["resid_diff"] = dp.dE_resid.median() - da.dE_resid.median()
    aa = a[present]
    r["rho_act_vs_dE"], r["p_act"] = stats.spearmanr(aa, des.loc[aa.index, "dE_interaction"])
    r["rho_act_vs_dEresid"], r["p_act_sizeadj"] = stats.spearmanr(aa, des.loc[aa.index, "dE_resid"])
    rows.append(r)
res = pd.DataFrame(rows)
for col in ["p_presence_raw","p_presence_sizeadj","p_act","p_act_sizeadj"]:
    p = res[col].values; ok = ~np.isnan(p); q = np.full_like(p, np.nan)
    q[ok] = stats.false_discovery_control(p[ok]) if hasattr(stats,"false_discovery_control") else p[ok]*ok.sum()
    res["q_"+col[2:]] = q
pd.set_option("display.width", 250)
print(res.drop(columns=[c for c in res.columns if c.startswith("p_")]).round(3).sort_values("feature").to_string(index=False))
