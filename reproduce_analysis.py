from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RNG = 42
CESD_WAVES = ["r1cesd10", "r2cesd10", "r3cesd10", "r4cesd10"]
FEATS_NUM = ["r1agey", "hh1ahous", "r1tr20", "r1ser7", "r1orient", "r1adla_c", "r1mbmi", "r1gripsum"]
FEATS_CAT = ["ragender", "raeduc_c", "r1mstat", "h1rural", "r1hibpe", "r1diabe", "r1cancre",
             "r1lunge", "r1hearte", "r1stroke", "r1arthre", "r1dyslipe", "r1walk1kma",
             "r1walk100a", "r1socwk", "r1work", "r1smoken", "r1drinkev", "r1shlt"]
MIN_AVE_PP, MIN_OCC, MIN_CLASS_PROP = 0.70, 5.0, 0.05


def enumerate_gmm(values: np.ndarray, waves: list[str], seed: int = RNG):
    """Enumerate k=2..6 for the prospective sensitivity analysis."""
    summaries, class_rows, fitted = [], [], {}
    n = len(values)
    for k in range(2, 7):
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=30,
                                max_iter=1000, random_state=seed, reg_covar=1e-4)
        raw_labels = model.fit_predict(values); raw_post = model.predict_proba(values)
        order = np.argsort([values[raw_labels == c].mean() for c in range(k)])
        remap = {raw: ordered for ordered, raw in enumerate(order)}
        labels = np.array([remap[x] for x in raw_labels]); post = raw_post[:, order]
        assigned = post[np.arange(n), labels]
        entropy = 1 + np.sum(post*np.log(np.clip(post, 1e-15, 1))) / (n*np.log(k))
        adequate = True
        for c in range(k):
            mask = labels == c; proportion = float(mask.mean()); ave_pp = float(assigned[mask].mean())
            odds_pp = np.inf if ave_pp >= 1 else ave_pp/(1-ave_pp)
            occ = odds_pp/(proportion/(1-proportion))
            adequate &= ave_pp >= MIN_AVE_PP and occ > MIN_OCC and proportion >= MIN_CLASS_PROP
            row = {"k": k, "class": c+1, "n": int(mask.sum()), "proportion": proportion,
                   "average_posterior_probability": ave_pp, "occ": occ}
            row.update({f"mean_{w}": float(values[mask, j].mean()) for j, w in enumerate(waves)})
            class_rows.append(row)
        summaries.append({"k": k, "bic": model.bic(values), "aic": model.aic(values),
                          "relative_entropy": float(entropy), "adequate": bool(adequate)})
        fitted[k] = (labels, post)
    eligible = [r["k"] for r in summaries if r["adequate"]]
    if not eligible: raise RuntimeError("No k=2..6 solution met all adequacy criteria")
    selected_k = max(eligible)
    return selected_k, fitted[selected_k], pd.DataFrame(summaries), pd.DataFrame(class_rows)

THRESHOLDS = [0.05, 0.10, 0.138, 0.20, 0.30, 0.40, 0.50]


def design(df: pd.DataFrame, features: list[str] | None = None):
    features = features or FEATS_NUM + FEATS_CAT
    num = [c for c in FEATS_NUM if c in features]
    cat = [c for c in features if c not in num]
    x = df[features].copy()
    for c in ("r1gripsum", "r1mbmi"):
        if c in x:
            x[c + "_miss"] = x[c].isna().astype(int)
            cat.append(c + "_miss")
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", SimpleImputer(strategy="most_frequent"), cat),
    ], verbose_feature_names_out=False)
    return x, pre


def estimators(pre, spw: float, quick: bool):
    trees = 80 if quick else 500
    return {
        "Logistic": Pipeline([("pre", clone(pre)), ("model", LogisticRegression(
            max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=RNG))]),
        "RandomForest": Pipeline([("pre", clone(pre)), ("model", RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
            random_state=RNG))]),
        "XGBoost": Pipeline([("pre", clone(pre)), ("model", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=1,
            random_state=RNG, scale_pos_weight=spw, max_depth=3, learning_rate=.01,
            n_estimators=trees, subsample=.8, colsample_bytree=.8))]),
    }


def calibration_metrics(y, p):
    """Cox slope plus calibration-in-the-large with slope fixed at one."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(q / (1 - q)); y = np.asarray(y, dtype=float)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones_like(y), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(slope), float(intercept)


def bootstrap_auc(y, p, n, seed=RNG):
    rng, vals = np.random.default_rng(seed), []
    y = np.asarray(y)
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            vals.append(roc_auc_score(y[ix], p[ix]))
    return np.percentile(vals, [2.5, 97.5])


def bootstrap_roc_band(y, p, n, seed=RNG):
    """Pointwise percentile band for an ROC curve on a fixed FPR grid."""
    rng = np.random.default_rng(seed); grid = np.linspace(0, 1, 201); curves = []
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size != 2:
            continue
        fpr, tpr, _ = roc_curve(y[ix], p[ix])
        curves.append(np.interp(grid, fpr, tpr))
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    return grid, lo, hi


def midrank(x):
    order = np.argsort(x); z = x[order]; ranks = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]: j += 1
        ranks[i:j] = .5 * (i + j - 1) + 1; i = j
    out = np.empty(len(x)); out[order] = ranks
    return out


def delong(y, a, b):
    order = np.argsort(-np.asarray(y)); m = int(np.sum(y)); preds = np.vstack([a, b])[:, order]
    pos, neg = preds[:, :m], preds[:, m:]; tx = np.array([midrank(x) for x in pos])
    ty = np.array([midrank(x) for x in neg]); tz = np.array([midrank(x) for x in preds])
    auc = (tz[:, :m].sum(1) / m - (m + 1) / 2) / len(neg)
    v01 = (tz[:, :m] - tx) / len(neg); v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / len(neg)
    var = max(float(np.array([1, -1]) @ cov @ np.array([1, -1])), 1e-15)
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return float(auc[0] - auc[1]), float(2 * (1 - NormalDist().cdf(abs(z))))


def evaluate(df, outcome, out: Path, quick=False, prefix="main", figures=True):
    out.mkdir(parents=True, exist_ok=True)
    X, pre = design(df); y = np.asarray(outcome, dtype=int)
    ix = np.arange(len(y)); tr, te = train_test_split(ix, test_size=.3, stratify=y,
                                                      random_state=RNG)
    spw = np.sum(y[tr] == 0) / np.sum(y[tr] == 1)
    models = estimators(pre, spw, quick); fitted, probs, cprobs, rows = {}, {}, {}, []
    boots = 100 if quick else 2000
    cv_rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    for fold, (fit_rel, val_rel) in enumerate(cv.split(X.iloc[tr], y[tr]), start=1):
        fit_ix, val_ix = tr[fit_rel], tr[val_rel]
        fold_spw = np.sum(y[fit_ix] == 0) / np.sum(y[fit_ix] == 1)
        for name, model in estimators(pre, fold_spw, quick).items():
            model.fit(X.iloc[fit_ix], y[fit_ix]); p_cv = model.predict_proba(X.iloc[val_ix])[:, 1]
            pred_cv = p_cv >= .5; tn, fp, fn, tp = confusion_matrix(y[val_ix], pred_cv).ravel()
            cv_rows.append({"fold": fold, "model": name, "auc": roc_auc_score(y[val_ix], p_cv),
                            "sensitivity": recall_score(y[val_ix], pred_cv), "specificity": tn/(tn+fp),
                            "f1": f1_score(y[val_ix], pred_cv), "brier": brier_score_loss(y[val_ix], p_cv)})
    cv_detail = pd.DataFrame(cv_rows)
    cv_detail.to_csv(out/f"{prefix}_fivefold_cv_folds.csv", index=False)
    cv_detail.groupby("model", as_index=False).agg(
        auc=("auc","mean"), sensitivity=("sensitivity","mean"), specificity=("specificity","mean"),
        f1=("f1","mean"), brier=("brier","mean")).to_csv(out/f"{prefix}_fivefold_cv_summary.csv", index=False)
    for name, model in models.items():
        model.fit(X.iloc[tr], y[tr]); p = model.predict_proba(X.iloc[te])[:, 1]
        cal = CalibratedClassifierCV(clone(model), method="sigmoid", cv=5).fit(X.iloc[tr], y[tr])
        pc = cal.predict_proba(X.iloc[te])[:, 1]
        pred = p >= .5; tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
        lo, hi = bootstrap_auc(y[te], p, boots); slope, intercept = calibration_metrics(y[te], p)
        cslope, cintercept = calibration_metrics(y[te], pc)
        rows.append({"model": name, "auc": roc_auc_score(y[te], p), "auc_lo": lo, "auc_hi": hi,
                     "pr_auc": average_precision_score(y[te], p), "precision_ppv": precision_score(y[te], pred),
                     "recall_sensitivity": recall_score(y[te], pred), "specificity": tn/(tn+fp),
                     "f1": f1_score(y[te], pred), "npv": tn/(tn+fn), "brier_raw": brier_score_loss(y[te], p),
                     "calibration_slope_raw": slope, "calibration_intercept_raw": intercept,
                     "auc_calibrated": roc_auc_score(y[te], pc), "brier_calibrated": brier_score_loss(y[te], pc),
                     "calibration_slope_calibrated": cslope, "calibration_intercept_calibrated": cintercept})
        fitted[name], probs[name], cprobs[name] = model, p, pc
    pd.DataFrame(rows).to_csv(out/f"{prefix}_performance.csv", index=False)

    burden, dca = [], []
    prevalence = y[te].mean()
    for name in models:
        for t in THRESHOLDS:
            flag = cprobs[name] >= t; tp = np.sum(flag & (y[te] == 1)); fp = np.sum(flag & (y[te] == 0))
            burden.append({"model": name, "threshold": t, "n_flagged": int(flag.sum()),
                           "pct_flagged": 100*flag.mean(), "sensitivity": tp/np.sum(y[te]),
                           "ppv": tp/flag.sum() if flag.sum() else np.nan})
            for probability, values in (("raw", probs[name]), ("platt", cprobs[name])):
                decision = values >= t
                dtp = np.sum(decision & (y[te] == 1)); dfp = np.sum(decision & (y[te] == 0))
                dca.append({"model": name, "probability": probability, "threshold": t,
                            "net_benefit": (dtp-dfp*t/(1-t))/len(te),
                            "treat_all": prevalence-(1-prevalence)*t/(1-t), "treat_none": 0})
    pd.DataFrame(burden).to_csv(out/f"{prefix}_predicted_positive_burden.csv", index=False)
    pd.DataFrame(dca).to_csv(out/f"{prefix}_decision_curve.csv", index=False)

    pairs = []
    names = list(models)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            delta, pval = delong(y[te], probs[names[i]], probs[names[j]])
            rng=np.random.default_rng(RNG); deltas=[]
            for _ in range(boots):
                bix=rng.integers(0,len(te),len(te))
                if np.unique(y[te][bix]).size==2:
                    deltas.append(roc_auc_score(y[te][bix],probs[names[i]][bix])-roc_auc_score(y[te][bix],probs[names[j]][bix]))
            dlo,dhi=np.percentile(deltas,[2.5,97.5])
            pairs.append({"model_a": names[i], "model_b": names[j], "delta_auc": delta,
                          "delta_auc_lo":dlo,"delta_auc_hi":dhi,"delong_p": pval})
    pd.DataFrame(pairs).to_csv(out/f"{prefix}_delong.csv", index=False)

    if figures:
        fig, ax = plt.subplots();
        for offset, (name, p) in enumerate(probs.items()):
            fpr, tpr, _ = roc_curve(y[te], p); line = ax.plot(fpr, tpr, label=name)[0]
            grid, lo_band, hi_band = bootstrap_roc_band(y[te], p, boots, RNG+offset)
            ax.fill_between(grid, lo_band, hi_band, color=line.get_color(), alpha=.15, linewidth=0)
        ax.plot([0,1],[0,1], "k--"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate"); ax.legend()
        fig.tight_layout(); fig.savefig(out/f"{prefix}_roc.png", dpi=300); plt.close(fig)
        fig, ax = plt.subplots()
        for name in names:
            obs, pred = calibration_curve(y[te], probs[name], n_bins=10)
            line = ax.plot(pred, obs, marker="o", linestyle="--", alpha=.7, label=f"{name} raw")[0]
            obs, pred = calibration_curve(y[te], cprobs[name], n_bins=10)
            brier = brier_score_loss(y[te], cprobs[name])
            ax.plot(pred, obs, marker="o", color=line.get_color(),
                    label=f"{name} Platt (Brier={brier:.3f})")
        ax.plot([0,1],[0,1], "k--"); ax.set(xlabel="Predicted probability", ylabel="Observed proportion"); ax.legend()
        fig.tight_layout(); fig.savefig(out/f"{prefix}_calibration.png", dpi=300); plt.close(fig)
        fig,ax=plt.subplots()
        ddf=pd.DataFrame(dca)
        for name in names:
            raw=ddf[(ddf.model==name)&(ddf.probability=="raw")]
            platt=ddf[(ddf.model==name)&(ddf.probability=="platt")]
            line=ax.plot(raw.threshold,raw.net_benefit,marker="o",linestyle="--",alpha=.7,label=f"{name} raw")[0]
            ax.plot(platt.threshold,platt.net_benefit,marker="o",color=line.get_color(),label=f"{name} Platt")
        z=ddf[(ddf.model==names[0])&(ddf.probability=="platt")]
        ax.plot(z.threshold,z.treat_all,"k--",label="Treat all"); ax.axhline(0,color="gray",label="Treat none")
        ax.set(xlabel="Threshold probability",ylabel="Net benefit"); ax.legend(); fig.tight_layout()
        fig.savefig(out/f"{prefix}_decision_curve.png",dpi=300); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(10,3))
        for ax,(name,p) in zip(axes,probs.items()):
            cm=confusion_matrix(y[te],p>=.5); image=ax.imshow(cm,cmap="Blues")
            for (r,c),v in np.ndenumerate(cm): ax.text(c,r,str(v),ha="center",va="center")
            ax.set(title=name,xlabel="Predicted",ylabel="Observed",xticks=[0,1],yticks=[0,1])
        fig.tight_layout(); fig.savefig(out/f"{prefix}_confusion_matrices.png",dpi=300); plt.close(fig)

    # Aggregate SHAP output; no participant predictions are saved.
    try:
        import shap
        xgb = fitted["XGBoost"]; xt = xgb.named_steps["pre"].transform(X.iloc[te])
        values = shap.TreeExplainer(xgb.named_steps["model"]).shap_values(xt)
        names_out = xgb.named_steps["pre"].get_feature_names_out()
        imp = pd.DataFrame({"feature": names_out, "mean_abs_shap": np.abs(values).mean(0)}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(out/f"{prefix}_shap_importance.csv", index=False)
        if figures:
            shap.summary_plot(values, xt, feature_names=names_out, show=False)
            plt.tight_layout(); plt.savefig(out/f"{prefix}_shap.png", dpi=300); plt.close()
    except ImportError:
        print("SHAP unavailable; install the locked requirements for SHAP outputs")
    return {"train_n": len(tr), "test_n": len(te), "positive_n": int(y.sum()), "rows": rows}


def repeated_splits(df, y, out, quick):
    X, pre = design(df); n = 5 if quick else 200; rows=[]
    for seed in range(n):
        tr, te = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=seed)
        spw=np.sum(y[tr]==0)/np.sum(y[tr]==1)
        for name, model in estimators(pre, spw, quick).items():
            model.fit(X.iloc[tr], y[tr]); rows.append({"split":seed,"model":name,
                "auc":roc_auc_score(y[te],model.predict_proba(X.iloc[te])[:,1])})
    pd.DataFrame(rows).to_csv(out/"table3_repeated_holdout.csv",index=False)


def xgb_grid_search(df, y, out):
    """Re-run and record the 108-configuration Table S5 search on training data only."""
    X,pre=design(df); tr,_=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=RNG)
    spw=np.sum(y[tr]==0)/np.sum(y[tr]==1)
    pipe=Pipeline([("pre",pre),("model",XGBClassifier(objective="binary:logistic",eval_metric="logloss",
        tree_method="hist",n_jobs=1,random_state=RNG,scale_pos_weight=spw))])
    grid={"model__max_depth":[3,5,7],"model__learning_rate":[.01,.05,.1],
          "model__n_estimators":[300,500,700],"model__subsample":[.8,1.0],
          "model__colsample_bytree":[.8,1.0]}
    gs=GridSearchCV(pipe,grid,scoring="roc_auc",cv=StratifiedKFold(5,shuffle=True,random_state=RNG),
                    n_jobs=1,refit=False,return_train_score=False).fit(X.iloc[tr],y[tr])
    cols=[c for c in gs.cv_results_ if c.startswith("param_")]+["mean_test_score","std_test_score","rank_test_score"]
    pd.DataFrame(gs.cv_results_)[cols].sort_values("rank_test_score").to_csv(out/"table_s5_xgb_grid_search.csv",index=False)


def missingness_sensitivity(df, outcomes, out, quick):
    """Table S4A/B: change only r1shlt handling; pool MICE metrics over m=20."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    import shap

    def fit_once(work, y, train, test, features):
        x, pre = design(work, features); spw = np.sum(y[train] == 0) / np.sum(y[train] == 1)
        model = estimators(pre, spw, quick)["XGBoost"]
        model.fit(x.iloc[train], y[train]); p = model.predict_proba(x.iloc[test])[:, 1]
        xt = model.named_steps["pre"].transform(x.iloc[test])
        vals = shap.TreeExplainer(model.named_steps["model"]).shap_values(xt)
        names = model.named_steps["pre"].get_feature_names_out()
        return p, pd.Series(np.abs(vals).mean(0), index=names)

    rows = []; base_features = FEATS_NUM + FEATS_CAT; boots = 100 if quick else 2000
    for panel, y in outcomes.items():
        y = np.asarray(y, dtype=int)
        tr, te = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=RNG)
        for scheme in ("mode", "missing_indicator", "complete_case", "mice_m20"):
            work = df.copy(); train = tr.copy(); test = te.copy(); features = base_features.copy()
            aucs, imps, final_p = [], [], None
            if scheme == "complete_case":
                complete = np.flatnonzero(work.r1shlt.notna().to_numpy())
                train, test = train_test_split(complete, test_size=.3, stratify=y[complete], random_state=RNG)
            if scheme == "missing_indicator":
                work["r1shlt_miss"] = work.r1shlt.isna().astype(int); features.append("r1shlt_miss")
            if scheme == "mice_m20":
                # The chained equation uses all predictors, but only its r1shlt column replaces
                # the original data; every other variable retains main-analysis handling. Each
                # completed dataset is then passed through the identical locked 70/30 pipeline.
                mice_cols = base_features
                for j in range(20):
                    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                                           max_iter=10, random_state=RNG+j)
                    completed = imp.fit_transform(work[mice_cols])
                    shlt_col = mice_cols.index("r1shlt")
                    imputed = work.copy()
                    imputed["r1shlt"] = np.clip(np.rint(completed[:, shlt_col]), 1, 5)
                    p, importance = fit_once(imputed, y, train, test, features)
                    aucs.append(roc_auc_score(y[test], p)); imps.append(importance); final_p = p
                importance = pd.concat(imps, axis=1).mean(axis=1)
                auc = float(np.mean(aucs)); lo, hi = min(aucs), max(aucs); interval = "across-imputation range"
            else:
                final_p, importance = fit_once(work, y, train, test, features)
                auc = roc_auc_score(y[test], final_p); lo, hi = bootstrap_auc(y[test], final_p, boots)
                interval = "95% bootstrap percentile CI"
            ordered = importance.sort_values(ascending=False)
            rows.append({"panel": panel, "method": scheme, "analytic_n": len(train)+len(test),
                         "positive_n": int(y[np.r_[train,test]].sum()), "auc": auc,
                         "auc_interval_lo": lo, "auc_interval_hi": hi, "interval_type": interval,
                         "r1shlt_mean_abs_shap": importance.get("r1shlt", np.nan),
                         "r1shlt_rank": int(ordered.index.get_loc("r1shlt")+1),
                         "r1arthre_mean_abs_shap": importance.get("r1arthre", np.nan),
                         "r1arthre_rank": int(ordered.index.get_loc("r1arthre")+1),
                         "r1shlt_missing_rank": (int(ordered.index.get_loc("r1shlt_miss")+1)
                                                   if "r1shlt_miss" in ordered.index else np.nan)})
    pd.DataFrame(rows).to_csv(out/"table_s4_missingness_sensitivity.csv", index=False)


def sex_stratified(df, y, out, quick):
    """Table 5 using the locked pooled split and sex-specific imbalance weights."""
    pooled_tr,pooled_te=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=RNG)
    rows=[]; features=[f for f in FEATS_NUM+FEATS_CAT if f!="ragender"]
    for code,label in ((1,"Male"),(2,"Female")):
        tr=pooled_tr[df.ragender.iloc[pooled_tr].to_numpy()==code]
        te=pooled_te[df.ragender.iloc[pooled_te].to_numpy()==code]
        x,pre=design(df,features); spw=np.sum(y[tr]==0)/np.sum(y[tr]==1)
        model=estimators(pre,spw,quick)["XGBoost"]; model.fit(x.iloc[tr],y[tr]); p=model.predict_proba(x.iloc[te])[:,1]
        cal=CalibratedClassifierCV(clone(model),method="sigmoid",cv=5).fit(x.iloc[tr],y[tr]); pc=cal.predict_proba(x.iloc[te])[:,1]
        lo,hi=bootstrap_auc(y[te],p,100 if quick else 2000)
        row={"sex":label,"train_n":len(tr),"test_n":len(te),"scale_pos_weight":spw,"auc_raw":roc_auc_score(y[te],p),
             "auc_lo":lo,"auc_hi":hi,"auc_calibrated":roc_auc_score(y[te],pc),"brier_raw":brier_score_loss(y[te],p),
             "brier_calibrated":brier_score_loss(y[te],pc)}
        try:
            import shap
            xt=model.named_steps["pre"].transform(x.iloc[te]); vals=shap.TreeExplainer(model.named_steps["model"]).shap_values(xt)
            names=model.named_steps["pre"].get_feature_names_out(); top=np.argsort(np.abs(vals).mean(0))[::-1][:5]
            row["top_five_shap"]="; ".join(names[top])
        except ImportError: pass
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"table5_sex_stratified.csv",index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=Path("final_analytic_dataset.csv"))
    ap.add_argument("--output-dir",type=Path,default=Path("results")); ap.add_argument("--quick",action="store_true")
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data); y=df.outcome_high_risk.to_numpy(int)
    if not args.quick: xgb_grid_search(df,y,args.output_dir)
    main_result=evaluate(df,y,args.output_dir,args.quick,"main")
    repeated_splits(df,y,args.output_dir,args.quick)
    sex_stratified(df,y,args.output_dir,args.quick)

    # Prospective W2-W4 outcome and its full class enumeration (Table S9).
    k,(labs,_),summary,classes=enumerate_gmm(df[CESD_WAVES[1:]].to_numpy(),CESD_WAVES[1:])
    summary.to_csv(args.output_dir/"table_s9_prospective_gmm_enumeration.csv",index=False)
    classes.to_csv(args.output_dir/"table_s9_prospective_gmm_classes.csv",index=False)
    prospective=(labs==k-1).astype(int)
    evaluate(df,prospective,args.output_dir,args.quick,"prospective",figures=False)
    missingness_sensitivity(df,{"A_primary_W1_W4":y,"B_prospective_W2_W4":prospective},
                            args.output_dir,args.quick)

    # Outcome-definition robustness (k=4, k=6, and merged top two classes).
    robust=[]
    values=df[CESD_WAVES].to_numpy()
    for kk in (4,6):
        gm=GaussianMixture(kk,covariance_type="full",n_init=30,max_iter=1000,random_state=RNG,reg_covar=1e-4).fit(values)
        raw=gm.predict(values); order=np.argsort([values[raw==c].mean() for c in range(kk)])
        remap={r:i for i,r in enumerate(order)}; labs2=np.array([remap[z] for z in raw]); yy=(labs2==kk-1).astype(int)
        res=evaluate(df,yy,args.output_dir,args.quick,f"robust_k{kk}",figures=False)
        robust.append({"definition":f"k={kk}","positive_n":int(yy.sum()),"xgb_auc":next(x["auc"] for x in res["rows"] if x["model"]=="XGBoost")})
    merged=np.isin(df.traj_class,[df.traj_class.max()-1,df.traj_class.max()]).astype(int)
    res=evaluate(df,merged,args.output_dir,args.quick,"robust_merged",figures=False)
    robust.append({"definition":"merged top two","positive_n":int(merged.sum()),"xgb_auc":next(x["auc"] for x in res["rows"] if x["model"]=="XGBoost")})
    pd.DataFrame(robust).to_csv(args.output_dir/"table_s2_outcome_robustness.csv",index=False)

    manifest={"seed":RNG,"quick_mode":args.quick,"analytic_n":len(df),"main_positive_n":int(y.sum()),
              "train_n":main_result["train_n"],"test_n":main_result["test_n"],"prospective_k":k}
    (args.output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
