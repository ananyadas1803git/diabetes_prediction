"""
Ablation Study Results Visualization

This script visualizes and compares the results from the ablation study.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_ablation_results(results_path: str) -> pd.DataFrame:
    """Load ablation study results from CSV."""
    return pd.read_csv(results_path)


def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: str = "results") -> None:
    """Plot comparison of all metrics across configurations."""
    # Filter out status column and convert to numeric
    metrics_df = results_df.drop(columns=['Status', 'Configuration'])
    config_names = results_df['Configuration']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Ablation Study: Metrics Comparison Across Configurations', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = results_df[metric].values
        bars = ax.bar(config_names, values, color=colors[:len(config_names)], alpha=0.7)
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12)
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if pd.notna(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "ablation_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to {output_path}")


def plot_recall_f1_tradeoff(results_df: pd.DataFrame, output_dir: str = "results") -> None:
    """Plot Recall vs F1 score trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    config_names = results_df['Configuration']
    recall = results_df['Recall'].values
    f1 = results_df['F1'].values
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    scatter = ax.scatter(recall, f1, c=colors[:len(config_names)], s=200, alpha=0.7)
    
    # Add labels for each point
    for i, config in enumerate(config_names):
        ax.annotate(config, (recall[i], f1[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Recall vs F1 Score Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    min_val = min(min(recall[pd.notna(recall)]), min(f1[pd.notna(f1)]))
    max_val = max(max(recall[pd.notna(recall)]), max(f1[pd.notna(f1)]))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect Balance')
    ax.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "ablation_recall_f1_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Recall-F1 trade-off plot saved to {output_path}")


def plot_radar_chart(results_df: pd.DataFrame, output_dir: str = "results") -> None:
    """Plot radar chart for comprehensive comparison."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    config_names = results_df['Configuration']
    
    # Filter data
    data = results_df[metrics].values
    
    # Create angles for radar chart
    angles = [n / len(metrics) * 2 * 3.14159 for n in range(len(metrics))]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, config in enumerate(config_names):
        values = data[i].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=config, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Ablation Study: Comprehensive Performance Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "ablation_radar_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved to {output_path}")


def plot_improvement_analysis(results_df: pd.DataFrame, output_dir: str = "results") -> None:
    """Plot improvement analysis showing percentage gains."""
    # Use XGBoost Only as baseline
    baseline_idx = results_df[results_df['Configuration'] == 'XGBoost Only'].index[0]
    baseline_values = results_df.iloc[baseline_idx][['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']].values
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    
    # Calculate improvements
    improvements = []
    config_names = []
    
    for idx, row in results_df.iterrows():
        if idx == baseline_idx:
            continue
        
        config = row['Configuration']
        values = row[metrics].values
        
        # Calculate percentage improvement
        pct_improvement = [(val - base) / base * 100 if pd.notna(val) and pd.notna(base) and base != 0 
                          else 0 for val, base in zip(values, baseline_values)]
        
        improvements.append(pct_improvement)
        config_names.append(config)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(metrics))
    width = 0.35
    
    colors = ['#ff7f0e', '#2ca02c']
    
    for i, (improvement, config) in enumerate(zip(improvements, config_names)):
        offset = i * width
        bars = ax.bar([xi + offset for xi in x], improvement, width, 
                     label=config, color=colors[i], alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, improvement):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement over XGBoost Only (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Performance Improvement Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks([xi + width/2 for xi in x])
    ax.set_xticklabels(metrics)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "ablation_improvement_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Improvement analysis plot saved to {output_path}")


def generate_summary_report(results_df: pd.DataFrame, output_dir: str = "results") -> None:
    """Generate a text summary report."""
    report_path = Path(output_dir) / "ablation_study_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDY SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configurations Tested:\n")
        for config in results_df['Configuration']:
            f.write(f"  - {config}\n")
        f.write("\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 80 + "\n")
        
        for _, row in results_df.iterrows():
            f.write(f"\n{row['Configuration']}:\n")
            f.write(f"  Status: {row['Status']}\n")
            f.write(f"  Accuracy:  {row['Accuracy']:.4f}\n" if pd.notna(row['Accuracy']) else "  Accuracy:  N/A\n")
            f.write(f"  Precision: {row['Precision']:.4f}\n" if pd.notna(row['Precision']) else "  Precision: N/A\n")
            f.write(f"  Recall:    {row['Recall']:.4f}\n" if pd.notna(row['Recall']) else "  Recall:    N/A\n")
            f.write(f"  F1:        {row['F1']:.4f}\n" if pd.notna(row['F1']) else "  F1:        N/A\n")
            f.write(f"  ROC-AUC:   {row['ROC-AUC']:.4f}\n" if pd.notna(row['ROC-AUC']) else "  ROC-AUC:   N/A\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Key Findings:\n")
        f.write("="*80 + "\n")
        
        # Find best performing configuration for each metric
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        for metric in metrics:
            valid_results = results_df[results_df[metric].notna()]
            if not valid_results.empty:
                best_config = valid_results.loc[valid_results[metric].idxmax(), 'Configuration']
                best_value = valid_results[metric].max()
                f.write(f"  Best {metric}: {best_config} ({best_value:.4f})\n")
        
    print(f"Summary report saved to {report_path}")


def main(results_path: str, output_dir: str = "results"):
    """Generate all visualizations for ablation study results."""
    
    # Load results
    results_df = load_ablation_results(results_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Generating ablation study visualizations...")
    
    # Generate all plots
    plot_metrics_comparison(results_df, output_dir)
    plot_recall_f1_tradeoff(results_df, output_dir)
    plot_radar_chart(results_df, output_dir)
    plot_improvement_analysis(results_df, output_dir)
    
    # Generate summary report
    generate_summary_report(results_df, output_dir)
    
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Ablation Study Results")
    parser.add_argument("--results", type=str, required=True, 
                       help="Path to ablation study results CSV file")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    main(args.results, args.output)
