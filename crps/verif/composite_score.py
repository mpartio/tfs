# composite_score.py
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

COMP_COLS = ["S_mae", "S_fss", "S_coh", "S_hk", "S_var", "S_change"]
COMP_LABELS = {
    "S_mae": "MAE",
    "S_fss": "FSS",
    "S_coh": "Coherence",
    "S_hk": "High-k",
    "S_var": "Variance",
    "S_change": "Change",
}


def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def composite_score(
    run_name: list[str],
    values: dict,
    preset: str = "sharpness",
    mae_ref: float | None = None,
    tol_var: float = 0.10,
    tol_hk: float = 0.15,
    tol_stat: float = 0.25,
    lead_weights: dict[int, float] | None = None,  # e.g., {0:1,1:1,2:1,3:1,4:1}
    save_path: str = "runs/verification",
):
    df_mae = values["mae"]
    df_fss = values["fss"]
    df_var = values["variance_ratio"]
    df_hk = values["highk_power_ratio"]
    df_coh = values["spectral_coherence"]
    df_chg = values["change_metrics"]

    # Optional lead weighting
    def avg_over_leads(group, col):
        if not lead_weights:
            return _safe_mean(group[col])
        # weighted mean
        g = group.copy()
        g["w"] = g["timestep"].map(lead_weights).fillna(1.0)
        vals = pd.to_numeric(g[col], errors="coerce")
        w = g["w"]
        ok = vals.notna() & w.notna()
        if ok.any():
            return float((vals[ok] * w[ok]).sum() / w[ok].sum())
        return np.nan

    # Establish mae_ref if not provided: median model-wise mean MAE
    if mae_ref is None and not df_mae.empty:
        tmp = df_mae.groupby("model").apply(lambda g: _safe_mean(g["mae"]))
        mae_ref = float(tmp.median()) if len(tmp) else 1.0
    if not mae_ref or mae_ref <= 0:
        mae_ref = 1.0

    # Build a per-model table of normalized scores
    models = (
        set(df_mae["model"])
        | set(df_fss.get("model", []))
        | set(df_var["model"])
        | set(df_hk["model"])
        | set(df_coh["model"])
        | set(df_chg["model"])
    )
    models = sorted(list(models))

    rows = []
    for m in models:
        # MAE -> S_mae
        g = df_mae[df_mae["model"] == m]
        mae_mean = avg_over_leads(g, "mae") if not g.empty else np.nan
        S_mae = (
            float(math.exp(-((mae_mean / mae_ref) ** 2)))
            if not np.isnan(mae_mean)
            else np.nan
        )

        # FSS -> S_fss
        if not df_fss.empty:
            g = df_fss[df_fss["model"] == m]
            S_fss = avg_over_leads(g, "fss")
        else:
            S_fss = np.nan

        # Variance ratio -> S_var (ideal 1)
        g = df_var[df_var["model"] == m]
        var_mean = avg_over_leads(g, "variance_ratio")
        S_var = (
            max(0.0, 1.0 - abs((var_mean or 1.0) - 1.0) / tol_var)
            if not np.isnan(var_mean)
            else np.nan
        )

        # High-k power ratio -> S_hk (ideal 1)
        g = df_hk[df_hk["model"] == m]
        hk_mean = avg_over_leads(g, "highk_power_ratio")
        S_hk = (
            max(0.0, 1.0 - abs((hk_mean or 1.0) - 1.0) / tol_hk)
            if not np.isnan(hk_mean)
            else np.nan
        )

        # Spectral coherence -> S_coh (avg mid & high)
        g = df_coh[df_coh["model"] == m]
        coh_mid = avg_over_leads(g, "coherence_mid") if not g.empty else np.nan
        coh_high = avg_over_leads(g, "coherence_high") if not g.empty else np.nan
        # Most of your coherence values are [0,1]; if not, clip.
        if not np.isnan(coh_mid) and not np.isnan(coh_high):
            S_coh = float(np.clip(0.5 * (coh_mid + coh_high), 0.0, 1.0))
        else:
            S_coh = np.nan

        # Change metrics -> S_change = 0.5*F1 + 0.3*Tcorr + 0.2*Stationarity
        g = df_chg[df_chg["model"] == m]
        S_f1 = avg_over_leads(g, "f1") if not g.empty else np.nan
        S_tcorr = avg_over_leads(g, "tendency_corr") if not g.empty else np.nan
        stat_mean = avg_over_leads(g, "stationarity_ratio") if not g.empty else np.nan
        S_stat = (
            max(0.0, 1.0 - abs((stat_mean or 1.0) - 1.0) / tol_stat)
            if not np.isnan(stat_mean)
            else np.nan
        )
        parts = [p for p in [S_f1, S_tcorr, S_stat] if not np.isnan(p)]
        S_change = (
            0.5 * (S_f1 if not np.isnan(S_f1) else 0)
            + 0.3 * (S_tcorr if not np.isnan(S_tcorr) else 0)
            + 0.2 * (S_stat if not np.isnan(S_stat) else 0)
        )
        if len(parts) == 0:
            S_change = np.nan

        # Choose weights
        if preset == "balanced":
            w = dict(mae=0.30, fss=0.15, coh=0.15, hk=0.10, var=0.10, change=0.20)
        elif preset == "sharpness":
            w = dict(mae=0.20, fss=0.20, coh=0.20, hk=0.15, var=0.10, change=0.15)
        else:
            raise ValueError("Unknown preset")

        # Weighted average ignoring NaNs (re-normalize weights)
        components = dict(
            S_mae=S_mae,
            S_fss=S_fss,
            S_coh=S_coh,
            S_hk=S_hk,
            S_var=S_var,
            S_change=S_change,
        )
        map_to_w = dict(
            S_mae="mae",
            S_fss="fss",
            S_coh="coh",
            S_hk="hk",
            S_var="var",
            S_change="change",
        )
        num = 0.0
        den = 0.0
        for k, v in components.items():
            if not np.isnan(v):
                wk = w[map_to_w[k]]
                num += wk * v
                den += wk
        score = num / den if den > 0 else np.nan

        rows.append(
            {
                "model": m,
                "S_mae": S_mae,
                "S_fss": S_fss,
                "S_coh": S_coh,
                "S_hk": S_hk,
                "S_var": S_var,
                "S_change": S_change,
                "score": score,
                "preset": preset,
                "mae_ref": mae_ref,
                "tol_var": tol_var,
                "tol_hk": tol_hk,
                "tol_stat": tol_stat,
            }
        )

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out.to_csv(f"{save_path}/results/composite_scores_{preset}.csv", index=False)
    return out


def _ensure_cols(df: pd.DataFrame):
    need = set(["model", "score"]) | set(COMP_COLS)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in composite CSV: {missing}")


def plot_composite_bars(
    df: pd.DataFrame,
    save_path: str = "runs/verification/figures",
    top_n: int | None = None,
):
    """
    Horizontal bar chart of composite score per model (sorted descending).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # df = pd.read_csv(csv_path)
    _ensure_cols(df)

    # Sort by score and keep top_n if requested
    df_sorted = df.sort_values("score", ascending=False)
    if top_n is not None:
        df_sorted = df_sorted.head(top_n)

    plt.figure(figsize=(10, max(4, 0.5 * len(df_sorted))))
    sns.barplot(
        data=df_sorted,
        y="model",
        x="score",
        palette="viridis",
        orient="h",
        edgecolor="black",
    )
    plt.xlabel("Composite Score (0–1, higher is better)")
    plt.ylabel("Model")
    title = "Composite Score by Model"
    # add preset if present
    if "preset" in df_sorted.columns:
        title += f"  [{df_sorted['preset'].iloc[0]}]"
    plt.title(title)
    plt.xlim(0, 1)
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    # annotate bars with value
    for i, (score) in enumerate(df_sorted["score"]):
        plt.text(score + 0.01, i, f"{score:.3f}", va="center")

    plt.tight_layout()
    filename = f"{save_path}/figures/composite_bars.png"
    plt.savefig(filename, dpi=200)
    print(f"Saved {filename}")
    plt.close()


def plot_component_contributions(
    df: pd.DataFrame,
    save_path: str = "runs/verification/figures",
    preset_weights: dict[str, float] | None = None,
    top_n: int | None = None,
):
    """
    Stacked bars showing how each component contributes to the final score per model.
    We weight each component by the preset weights and re-normalize if some components are NaN.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # df = pd.read_csv(csv_path)
    _ensure_cols(df)

    # Default weights (must match what you used in compute_composite_scores)
    if preset_weights is None:
        if (
            "preset" in df.columns
            and df["preset"].nunique() == 1
            and df["preset"].iloc[0] == "balanced"
        ):
            preset_weights = {
                "S_mae": 0.30,
                "S_fss": 0.15,
                "S_coh": 0.15,
                "S_hk": 0.10,
                "S_var": 0.10,
                "S_change": 0.20,
            }
        elif (
            "preset" in df.columns
            and df["preset"].nunique() == 1
            and df["preset"].iloc[0] == "sharpness"
        ):
            preset_weights = {
                "S_mae": 0.20,
                "S_fss": 0.20,
                "S_coh": 0.20,
                "S_hk": 0.15,
                "S_var": 0.10,
                "S_change": 0.15,
            }

    # Sort models by final score
    df_sorted = df.sort_values("score", ascending=False)
    if top_n is not None:
        df_sorted = df_sorted.head(top_n)

    # Compute weighted contributions per model (respecting NaNs)
    contrib_rows = []
    for _, row in df_sorted.iterrows():
        model = row["model"]
        parts = {}
        # re-normalize weights over available (non-NaN) components
        avail = {k: preset_weights[k] for k in COMP_COLS if pd.notna(row[k])}
        wsum = sum(avail.values()) if avail else 1.0
        for k in COMP_COLS:
            if pd.notna(row[k]):
                w = preset_weights[k] / wsum
                parts[k] = float(w * row[k])
            else:
                parts[k] = 0.0
        # numerical guard: rescale so parts sum to score (keeps consistency)
        total_parts = sum(parts.values())
        target = float(row["score"])
        scale = (target / total_parts) if total_parts > 0 else 0.0
        parts = {k: v * scale for k, v in parts.items()}
        parts["model"] = model
        contrib_rows.append(parts)

    contrib_df = pd.DataFrame(contrib_rows)

    # Melt to long for stacked bars
    long_df = contrib_df.melt(
        id_vars=["model"],
        value_vars=COMP_COLS,
        var_name="component",
        value_name="value",
    )
    # Map nicer labels
    long_df["component"] = long_df["component"].map(COMP_LABELS)

    # Keep original model order
    long_df["model"] = pd.Categorical(
        long_df["model"], categories=df_sorted["model"], ordered=True
    )

    plt.figure(figsize=(10, max(4, 0.6 * len(df_sorted))))
    # Stacked horizontal bars
    bottom = np.zeros(len(df_sorted))
    models = df_sorted["model"].tolist()
    palette = sns.color_palette("Set2", n_colors=len(COMP_COLS))
    comp_order = [COMP_LABELS[c] for c in COMP_COLS]

    for idx, comp in enumerate(comp_order):
        vals = []
        for m in models:
            v = long_df[(long_df["model"] == m) & (long_df["component"] == comp)][
                "value"
            ]
            vals.append(float(v.iloc[0]) if len(v) else 0.0)
        plt.barh(
            models, vals, left=bottom, color=palette[idx], edgecolor="black", label=comp
        )
        bottom += np.array(vals)

    plt.xlabel("Contribution to Composite Score")
    plt.ylabel("Model")
    plt.title("Composite Score Contributions by Component")
    plt.xlim(0, max(1.0, bottom.max() * 1.05))
    plt.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{save_path}/figures/composite_contributions.png"
    plt.savefig(filename, dpi=200)
    print(f"Saved {filename}")
    plt.close()
