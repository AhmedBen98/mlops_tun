"""
Script de comparaison des différentes versions du dataset Visiteurs
Compare les statistiques et (optionnellement) les performances des modèles sur les 3 versions
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

import utils
setup_logging = utils.setup_logging
ensure_dir = utils.ensure_dir

logger = setup_logging()

class DatasetComparator:
    """Compare les différentes versions du dataset"""

    def __init__(self):
        self.versions = {}
        self.metrics = {}

    def load_version(self, version_name, filepath):
        logger.info(f"Loading {version_name} from {filepath}")
        df = pd.read_csv(filepath)
        self.versions[version_name] = df
        return df

    def load_metrics(self, version_name, metrics_file):
        if Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                self.metrics[version_name] = json.load(f)
            logger.info(f"Loaded metrics for {version_name}")
        else:
            logger.warning(f"Metrics file not found: {metrics_file}")

    def compare_datasets(self, output_dir='reports'):
        ensure_dir(output_dir)
        logger.info("Comparing dataset statistics...")
        comparison = {}
        for version_name, df in self.versions.items():
            comparison[version_name] = {
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,  # Excluding target
                'target_mean': float(df['visiteurs'].mean()),
                'target_std': float(df['visiteurs'].std()),
                'target_min': float(df['visiteurs'].min()),
                'target_max': float(df['visiteurs'].max()),
                'features': {}
            }
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col != 'visiteurs':
                    comparison[version_name]['features'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
        comparison_file = f"{output_dir}/dataset_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Dataset comparison saved to {comparison_file}")
        return comparison

    def plot_target_distributions(self, output_dir='reports'):
        ensure_dir(output_dir)
        logger.info("Plotting target distributions...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, (version_name, df) in enumerate(self.versions.items()):
            ax = axes[idx]
            ax.hist(df['visiteurs'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(df['visiteurs'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {df['visiteurs'].mean():.2f}")
            ax.set_xlabel('Visiteurs')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{version_name}\n({len(df)} samples)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = f"{output_dir}/target_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Target distributions saved to {output_file}")

    def generate_report(self, output_dir='reports'):
        ensure_dir(output_dir)
        logger.info("Generating comparison report...")
        dataset_stats = self.compare_datasets(output_dir)
        self.plot_target_distributions(output_dir)
        report_lines = [
            "# Rapport de Comparaison - Versions du Dataset Visiteurs",
            "",
            "## Statistiques des Datasets",
            ""
        ]
        report_lines.append("| Métrique | v1 | v2 | v3 |")
        report_lines.append("|----------|----|----|----|")
        metrics_names = [
            ('n_samples', 'Nombre d\'échantillons'),
            ('target_mean', 'Visiteurs - Moyenne'),
            ('target_std', 'Visiteurs - Écart-type'),
            ('target_min', 'Visiteurs - Min'),
            ('target_max', 'Visiteurs - Max')
        ]
        for key, label in metrics_names:
            values = [f"{dataset_stats[v][key]:.2f}" if isinstance(dataset_stats[v][key], float)
                      else str(dataset_stats[v][key])
                      for v in ['v1', 'v2', 'v3']]
            report_lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
        report_lines.append("")
        report_lines.append("## Visualisations")
        report_lines.append("")
        report_lines.append("### Distributions de la cible (visiteurs)")
        report_lines.append("![Target Distributions](target_distributions.png)")
        report_file = f"{output_dir}/comparison_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"Comparison report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare dataset versions')
    parser.add_argument('--v1-data', type=str, default='data/processed/v1/train.csv')
    parser.add_argument('--v2-data', type=str, default='data/processed/v2/train.csv')
    parser.add_argument('--v3-data', type=str, default='data/processed/v3/train.csv')
    parser.add_argument('--output', type=str, default='reports')
    args = parser.parse_args()
    comparator = DatasetComparator()
    comparator.load_version('v1', args.v1_data)
    comparator.load_version('v2', args.v2_data)
    comparator.load_version('v3', args.v3_data)
    comparator.generate_report(args.output)
    print("\n" + "=" * 80)
    print("Comparison report generated successfully!")
    print(f"Check the '{args.output}' directory for results")
    print("=" * 80)

if __name__ == "__main__":
    main()
