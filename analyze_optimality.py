import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean scientific plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

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
    
    # 1. Global Stats
    global_stats = {
        "Total Sentences": len(df),
        "Global Mean Omega": df['omega'].mean(),
        "Global Median Omega": df['omega'].median(),
        "Percent Optimized (>0)": (df['omega'] > 0).mean() * 100,
        "Correlation (Omega vs N)": df['omega'].corr(df['N'])
    }
    
    # Save Global Stats
    with open(os.path.join(output_dir, "global_statistics.txt"), "w") as f:
        for k, v in global_stats.items():
            f.write(f"{k}: {v:.4f}\n")

    # 2. Per-Language Stats
    lang_stats = df.groupby('language').agg({
        'omega': ['mean', 'std', 'count'],
        'N': 'mean',
        'd_real': 'mean'
    }).reset_index()
    
    # Flatten columns
    lang_stats.columns = ['Language', 'Omega_Mean', 'Omega_Std', 'Count', 'Avg_Length_N', 'Avg_Distance_d']
    lang_stats = lang_stats.sort_values(by="Omega_Mean", ascending=False)
    
    # Save to CSV
    stats_path = os.path.join(output_dir, "language_summary_stats.csv")
    lang_stats.to_csv(stats_path, index=False)
    print(f"Saved summary tables to {output_dir}")

def plot_omega_by_language(df, output_dir):
    print("Plotting Omega distribution by language...")
    plt.figure(figsize=(12, 6))
    
    # Sort languages by median Omega for better readability
    order = df.groupby('language')['omega'].median().sort_values(ascending=False).index
    
    sns.boxplot(x="language", y="omega", data=df, order=order, palette="viridis", linewidth=1)
    
    plt.axhline(0, color='red', linestyle='--', alpha=0.7, label="Random Baseline (0)")
    plt.axhline(1, color='green', linestyle='--', alpha=0.7, label="Theoretical Max (1)")
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribution of Optimality Score (Ω) by Language")
    plt.ylabel("Optimality Score (Ω)")
    plt.xlabel("")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "plot_omega_by_language.pdf"), format='pdf')
    plt.close()

def plot_omega_vs_length(df, output_dir):
    print("Plotting Omega vs Sentence Length...")
    plt.figure(figsize=(10, 6))
    
    # Use a hexbin or scatter with low alpha to handle dense data
    plt.hexbin(df['N'], df['omega'], gridsize=25, cmap='Blues', mincnt=1)
    cb = plt.colorbar(label='Count of Sentences')
    
    # Add a trend line (Lowess smoothing)
    sns.regplot(x="N", y="omega", data=df, scatter=False, color="red", line_kws={"label": "Trend"})
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Does Optimality degrade with Sentence Length?")
    plt.xlabel("Sentence Length (N nodes)")
    plt.ylabel("Optimality Score (Ω)")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "plot_omega_vs_length.pdf"), format='pdf')
    plt.close()

def plot_distance_comparison(df, output_dir):
    print("Plotting Distance comparisons (Real vs Random vs Min)...")
    
    # Melt the dataframe to long format for plotting
    df_long = pd.melt(df, 
                      id_vars=['sent_id'], 
                      value_vars=['d_min', 'd_real', 'd_rand'], 
                      var_name='Metric', 
                      value_name='Distance')
    
    # Rename for clearer legend
    metric_map = {'d_min': 'Minimum (Theoretical)', 'd_real': 'Real (Observed)', 'd_rand': 'Random (Baseline)'}
    df_long['Metric'] = df_long['Metric'].map(metric_map)
    
    plt.figure(figsize=(10, 6))
    
    # Violin plot to show density
    sns.violinplot(x="Metric", y="Distance", data=df_long, palette="muted", inner="quartile")
    
    plt.title("Comparison of Topological Distances")
    plt.ylabel("Mean Topological Distance (hops)")
    plt.xlabel("")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "plot_distance_comparison.pdf"), format='pdf')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze Syntactic Optimality Results")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the results CSV file")
    parser.add_argument("-o", "--output", type=str, default="analysis_output", help="Directory to save plots and tables")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run Analysis
    df = load_and_clean_data(args.input)
    
    generate_summary_stats(df, args.output)
    plot_omega_by_language(df, args.output)
    plot_omega_vs_length(df, args.output)
    plot_distance_comparison(df, args.output)
    
    print("\n" + "="*50)
    print(f"Analysis complete. Results saved in: {args.output}/")
    print("="*50)

if __name__ == "__main__":
    main()
