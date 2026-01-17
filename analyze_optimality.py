import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean scientific plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=2.2)

def load_and_clean_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Drop rows where Omega could not be calculated (NaN)
    initial_count = len(df)
    df = df.dropna(subset=['omega'])
    print(f"Loaded {initial_count} rows. Dropped {initial_count - len(df)} rows with NaN values.")

    return df

def generate_summary_stats(df, output_dir):
    print("Generating summary statistics...")

    global_stats = {
        "Total Sentences": len(df),
        "Global Mean Omega": df['omega'].mean(),
        "Global Median Omega": df['omega'].median(),
        "Percent Optimized (>0)": (df['omega'] > 0).mean() * 100,
        "Correlation (Omega vs N)": df['omega'].corr(df['N'])
    }

    with open(os.path.join(output_dir, "global_statistics.txt"), "w") as f:
        for k, v in global_stats.items():
            f.write(f"{k}: {v:.4f}\n")

    lang_stats = df.groupby('language').agg({
        'omega': ['mean', 'std', 'count'],
        'N': 'mean',
        'd_real': 'mean'
    }).reset_index()

    lang_stats.columns = ['Language', 'Omega_Mean', 'Omega_Std', 'Count', 'Avg_Length_N', 'Avg_Distance_d']
    lang_stats = lang_stats.sort_values(by="Omega_Mean", ascending=False)

    stats_path = os.path.join(output_dir, "language_summary_stats.csv")
    lang_stats.to_csv(stats_path, index=False)
    print(f"Saved summary tables to {output_dir}")

def plot_omega_by_language(df, output_dir):
    print("Plotting Omega distribution by language...")
    plt.figure(figsize=(12, 6))

    order = df.groupby('language')['omega'].median().sort_values(ascending=False).index
    sns.boxplot(x="language", y="omega", data=df, order=order, palette="viridis", linewidth=1)

    plt.axhline(0, color='red', linestyle='--', alpha=0.7, label="Random Baseline (0)")
    plt.axhline(1, color='green', linestyle='--', alpha=0.7, label="Theoretical Max (1)")

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Optimality Score (Ω)")
    plt.xlabel("")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "plot_omega_by_language.pdf"), format='pdf')
    plt.close()

def plot_omega_vs_length(df, output_dir):
    print("Plotting Omega vs Sentence Length...")
    plt.figure(figsize=(10, 6))

    plt.hexbin(df['N'], df['omega'], gridsize=25, cmap='Blues', mincnt=1)
    plt.colorbar(label='Count of Sentences')

    sns.regplot(x="N", y="omega", data=df, scatter=False, color="red", line_kws={"label": "Trend"})

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Sentence Length (N nodes)")
    plt.ylabel("Optimality Score (Ω)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "plot_omega_vs_length.pdf"), format='pdf')
    plt.close()

def plot_omega_histogram(df, output_dir):
    print("Plotting Omega histogram...")

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["omega"],
        bins=40,
        kde=True,
        color="steelblue",
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Random baseline (Ω = 0)")

    plt.xlabel("Optimality Score (Ω)")
    plt.ylabel("Number of Sentences")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "plot_omega_histogram.pdf"), format="pdf")
    plt.close()


def plot_distance_comparison(df, output_dir):
    print("Plotting Distance comparisons (Real vs Random vs Min)...")

    df_long = pd.melt(df,
                      id_vars=['sent_id'],
                      value_vars=['d_min', 'd_real', 'd_rand'],
                      var_name='Metric',
                      value_name='Distance')

    metric_map = {'d_min': 'Minimum (Theoretical)', 'd_real': 'Real (Observed)', 'd_rand': 'Random (Baseline)'}
    df_long['Metric'] = df_long['Metric'].map(metric_map)

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Metric", y="Distance", data=df_long, palette="muted", inner="quartile")

    plt.ylabel("Mean Topological Distance (hops)")
    plt.xlabel("")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "plot_distance_comparison.pdf"), format='pdf')
    plt.close()

def plot_sentence_length_histogram(df, output_dir):
    """
    Sentence length histogram with fixed bins:
      <=5, 6–10, 11–15, 16–20, 21–30, 31–40, 41–60, >60
    """
    print("Plotting sentence length histogram...")

    if 'N' not in df.columns:
        print("Skipping histogram: required column 'N' not found.")
        return

    # Fixed bins
    bins = [-np.inf, 5, 10, 15, 20, 30, 40, 60, np.inf]
    labels = ["≤5", "6–10", "11–15", "16–20", "21–30", "31–40", "41–60", ">60"]

    d = df[['N']].copy()
    d['len_bin'] = pd.cut(d['N'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)

    # Count per bin
    counts = d['len_bin'].value_counts().reindex(labels).fillna(0)

    x = np.arange(len(labels))
    y = counts.values.astype(float)

    plt.figure(figsize=(10, 6))

    plt.bar(
        x, y,
        color=sns.color_palette("muted")[0],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85
    )

    plt.xlabel("Sentence length (N)")
    plt.ylabel("Number of sentences")
    plt.xticks(x, labels)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_sentence_length_histogram.pdf"), format='pdf')
    plt.close()

def plot_omega_by_length_bins(df, output_dir):
    print("Plotting Omega by sentence length bins...")

    bins = [0, 5, 10, 15, 20, 30, 40, 60, 100]
    labels = ["≤5", "6–10", "11–15", "16–20", "21–30", "31–40", "41–60", ">60"]

    d = df.copy()
    d["N_bin"] = pd.cut(d["N"], bins=bins, labels=labels, right=True)

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        x="N_bin",
        y="omega",
        data=d,
        palette="Blues",
        showfliers=False,
        linewidth=1
    )

    plt.axhline(0, color="red", linestyle="--", linewidth=1.5,
                label="Random baseline (Ω = 0)")

    plt.xlabel("Sentence length (number of nodes)")
    plt.ylabel("Optimality score (Ω)")

    plt.xticks()
    plt.yticks()

    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig2_omega_by_length_bins.pdf"),
                format="pdf")
    plt.close()

def plot_omega_length_heatmap(df, output_dir):
    """
    Heatmap of Omega distribution by sentence-length bin.

    X-axis: sentence length bins
    Y-axis: Omega bins (including a special 0.99–1.0 bin)
    Cell value: proportion of sentences in that length bin that fall into that Omega bin
    """
    print("Plotting Omega-by-length heatmap...")

    required = {'N', 'omega'}
    if not required.issubset(df.columns):
        print("Skipping heatmap: required columns missing.")
        return

    # Sentence length bins (fixed, linguistic)
    len_bins = [-np.inf, 5, 10, 15, 20, 30, 40, 60, np.inf]
    len_labels = ["≤5", "6–10", "11–15", "16–20", "21–30", "31–40", "41–60", ">60"]

    # Omega bins with special near-1 resolution
    omega_bins = [-0.4, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.000001]
    omega_labels = [
        "-0.4-0", "0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5",
        "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–0.99", "0.99–1.0"
    ]

    d = df[['N', 'omega']].copy()
    d['len_bin'] = pd.cut(d['N'], bins=len_bins, labels=len_labels, right=True, include_lowest=True, ordered=True)
    d['omega_bin'] = pd.cut(d['omega'], bins=omega_bins, labels=omega_labels, right=False, include_lowest=True)

    counts = (
        d.groupby(['omega_bin', 'len_bin'])
         .size()
         .unstack(fill_value=0)
         .reindex(index=omega_labels, columns=len_labels, fill_value=0)
    )

    props = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1)

    plt.figure(figsize=(12, 7))
    sns.heatmap(
        props,
        cmap="viridis",
        vmin=0,
        vmax=props.max().max(),
        cbar_kws={"label": "Proportion within length bin"}
    )

    plt.xlabel("Sentence length (N)")
    plt.ylabel("Optimality score Ω")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_omega_length_heatmap.pdf"), format="pdf")
    plt.close()

# --------------------------
# Hypothesis test utilities
# --------------------------

def bootstrap_ci(x, stat_fn=np.median, n_boot=5000, alpha=0.05, seed=0):
    """
    Basic percentile bootstrap CI for a statistic.
    Returns: (stat, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        stats[i] = stat_fn(samp)

    lo = np.quantile(stats, alpha / 2)
    hi = np.quantile(stats, 1 - alpha / 2)
    return (stat_fn(x), lo, hi)

def sign_test_greater_than_zero(values):
    """
    Exact one-sided sign test for median > 0.
    H0: P(value > 0) = 0.5 (symmetry around 0)
    H1: median(value) > 0

    We discard ties (values == 0), as is standard for the sign test.

    Returns dict with n_eff, n_pos, p_value (exact binomial tail).
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    v = v[v != 0]  # discard ties
    n_eff = len(v)
    if n_eff == 0:
        return {"n_eff": 0, "n_pos": 0, "p_value": np.nan}

    n_pos = int(np.sum(v > 0))

    # Exact binomial tail: P(X >= n_pos) with X ~ Bin(n_eff, 0.5)
    # Compute in log-space for stability
    from math import comb
    denom = 2 ** n_eff
    num = 0
    for k in range(n_pos, n_eff + 1):
        num += comb(n_eff, k)
    p_value = num / denom
    return {"n_eff": n_eff, "n_pos": n_pos, "p_value": p_value}

def run_nonrandomness_test(df, output_dir, seed=0):
    """
    Tests whether Ω is significantly > 0 in a way that respects language clustering.

    Approach:
      - Compute one robust summary per language: median Ω within that language.
      - Test whether these language medians are > 0 using an exact one-sided sign test.
      - Report effect size: median of language medians + bootstrap CI.

    Saves results to: omega_nonrandomness_test.txt
    """
    print("Running hypothesis test: is Ω > 0 across languages?")

    if 'language' not in df.columns or 'omega' not in df.columns:
        print("Skipping hypothesis test: required columns missing.")
        return

    # One value per language (robust to outliers and sentence-level dependence)
    lang_medians = (
        df.groupby('language')['omega']
          .median()
          .dropna()
          .to_numpy(dtype=float)
    )

    L = len(lang_medians)
    out_path = os.path.join(output_dir, "omega_nonrandomness_test.txt")

    if L == 0:
        with open(out_path, "w") as f:
            f.write("No languages available after filtering.\n")
        print(f"Saved hypothesis test (empty) to {out_path}")
        return

    # Effect size: median across language medians + bootstrap CI
    med, lo, hi = bootstrap_ci(lang_medians, stat_fn=np.median, n_boot=5000, alpha=0.05, seed=seed)

    # Exact sign test for median > 0
    st = sign_test_greater_than_zero(lang_medians)

    # Useful descriptive: how many languages have median Ω > 0 (including ties)
    n_pos_including_ties = int(np.sum(lang_medians > 0))
    n_zero = int(np.sum(lang_medians == 0))
    n_neg = int(np.sum(lang_medians < 0))
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Non-randomness test for Ω (language-level)\n")
        f.write("=========================================\n\n")
        f.write("We compute one robust summary per language: median Ω over sentences.\n")
        f.write("We then test whether the median of these language medians is > 0.\n\n")

        f.write(f"Number of languages (L): {L}\n")
        f.write(f"Languages with median Ω > 0: {n_pos_including_ties}\n")
        f.write(f"Languages with median Ω = 0: {n_zero}\n")
        f.write(f"Languages with median Ω < 0: {n_neg}\n\n")

        f.write(f"Median of language medians: {med:.4f}\n")
        f.write(f"Bootstrap 95% CI (median): [{lo:.4f}, {hi:.4f}]\n\n")

        f.write("Exact one-sided sign test (discarding ties at 0)\n")
        f.write(f"Effective sample size (non-ties): {st['n_eff']}\n")
        f.write(f"Positive non-ties: {st['n_pos']}\n")
        f.write(f"p-value (H1: median > 0): {st['p_value']:.3e}\n")

    print(f"Saved hypothesis test results to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Syntactic Optimality Results")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the results CSV file")
    parser.add_argument("-o", "--output", type=str, default="analysis_output", help="Directory to save plots and tables")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap confidence intervals")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_and_clean_data(args.input)

    generate_summary_stats(df, args.output)
    plot_omega_by_language(df, args.output)
    plot_omega_vs_length(df, args.output)
    plot_distance_comparison(df, args.output)
    plot_sentence_length_histogram(df, args.output)

    # Diagnostic plot
    plot_omega_length_heatmap(df, args.output)

    # Hypothesis test: Ω significantly > 0?
    run_nonrandomness_test(df, args.output, seed=args.seed)

    print("\n" + "="*50)
    print(f"Analysis complete. Results saved in: {args.output}/")
    print("="*50)

if __name__ == "__main__":
    main()
