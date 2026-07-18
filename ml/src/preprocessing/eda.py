"""Exploratory Data Analysis module for FraudLens AI preprocessing pipeline.

This module provides the ExploratoryAnalyzer class, which generates descriptive
statistics, visualisations, and a markdown EDA report for the credit card fraud
dataset without opening interactive windows (headless/CI-safe).
"""

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from config.config import TARGET_COLUMN, settings
from src.utils.logging import get_logger

# Use non-interactive backend — safe for headless servers and CI pipelines
matplotlib.use("Agg")

logger = get_logger(__name__)

# Plot style constants
FIGURE_DPI: int = 150
FIGURE_SIZE_WIDE: tuple[int, int] = (14, 6)
FIGURE_SIZE_SQUARE: tuple[int, int] = (10, 8)
PALETTE: dict[int, str] = {0: "#3498db", 1: "#e74c3c"}
PALETTE_LIST: list[str] = ["#3498db", "#e74c3c"]


class ExploratoryAnalyzer:
    """Generates EDA statistics and visualisations for the fraud dataset.

    All plots are saved as PNG files under ml/reports/figures/. A markdown
    summary report is written to ml/reports/eda_report.md. No windows open.

    Attributes:
        df: The DataFrame to analyse.
        figures_dir: Directory path where PNG plots will be saved.
        report_path: Path for the generated eda_report.md file.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        figures_dir: Path | None = None,
        report_path: Path | None = None,
    ) -> None:
        """Initialises the ExploratoryAnalyzer.

        Args:
            df: The DataFrame to analyse.
            figures_dir: Optional override for the figures output directory.
            report_path: Optional override for the EDA markdown report path.
        """
        self.df = df
        self.figures_dir = figures_dir or settings.figures_dir
        self.report_path = report_path or (settings.reports_dir / "eda_report.md")
        self._saved_figures: list[str] = []

    def run(self) -> None:
        """Runs all EDA steps and saves plots and the markdown report.

        Generates class imbalance plot, amount/time distributions, fraud vs
        amount comparison, correlation heatmap, PCA scatter, feature statistics,
        and outlier summary. Writes eda_report.md on completion.
        """
        logger.info("Starting Exploratory Data Analysis...")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self._plot_class_imbalance()
        self._plot_amount_distribution()
        self._plot_time_distribution()
        self._plot_fraud_vs_amount()
        self._plot_correlation_heatmap()
        self._plot_pca_scatter()
        self._plot_feature_statistics()
        self._plot_outlier_summary()
        self._save_eda_report()

        logger.info(
            f"EDA complete — {len(self._saved_figures)} plots saved to: {self.figures_dir}"
        )

    # -------------------------------------------------------------------------
    # Plot generators
    # -------------------------------------------------------------------------

    def _plot_class_imbalance(self) -> None:
        """Generates and saves a bar chart showing class distribution.

        The chart displays raw counts and percentage labels for both classes.
        """
        counts = self.df[TARGET_COLUMN].value_counts().sort_index()
        labels = ["Normal (0)", "Fraud (1)"]
        colors = [PALETTE[0], PALETTE[1]]
        total = len(self.df)

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE, dpi=FIGURE_DPI)

        # Bar chart
        bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
        axes[0].set_title("Class Distribution — Count", fontsize=14, pad=12)
        axes[0].set_ylabel("Count")
        for bar, count in zip(bars, counts.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{count:,}",
                ha="center", fontsize=11, fontweight="bold",
            )

        # Pie chart
        pcts = [counts[0] / total * 100, counts[1] / total * 100]
        axes[1].pie(
            counts.values,
            labels=[f"Normal\n{pcts[0]:.3f}%", f"Fraud\n{pcts[1]:.3f}%"],
            colors=colors,
            autopct="%1.3f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        axes[1].set_title("Class Distribution — Proportion", fontsize=14, pad=12)

        plt.suptitle("Credit Card Fraud — Class Imbalance", fontsize=16, y=1.02)
        plt.tight_layout()
        self._save_figure("class_imbalance.png")
        plt.close()

    def _plot_amount_distribution(self) -> None:
        """Generates and saves Amount distribution histograms split by class."""
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE, dpi=FIGURE_DPI)
        for ax, cls, label in zip(axes, [0, 1], ["Normal", "Fraud"]):
            subset = self.df[self.df[TARGET_COLUMN] == cls]["Amount"]
            ax.hist(subset, bins=60, color=PALETTE[cls], alpha=0.8, edgecolor="white", linewidth=0.4)
            ax.set_title(f"Transaction Amount — {label}", fontsize=13)
            ax.set_xlabel("Amount (USD)")
            ax.set_ylabel("Frequency")
            ax.axvline(subset.median(), color="black", linestyle="--", linewidth=1.2, label=f"Median: ${subset.median():.2f}")
            ax.legend(fontsize=10)
        plt.suptitle("Transaction Amount Distribution by Class", fontsize=15, y=1.01)
        plt.tight_layout()
        self._save_figure("amount_distribution.png")
        plt.close()

    def _plot_time_distribution(self) -> None:
        """Generates and saves Time distribution density plots split by class."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=FIGURE_DPI)
        for cls, label in [(0, "Normal"), (1, "Fraud")]:
            subset = self.df[self.df[TARGET_COLUMN] == cls]["Time"]
            ax.hist(subset, bins=80, alpha=0.6, color=PALETTE[cls], label=label, edgecolor="none", density=True)
        ax.set_title("Transaction Time Distribution by Class", fontsize=14)
        ax.set_xlabel("Time (seconds from first transaction)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=11)
        plt.tight_layout()
        self._save_figure("time_distribution.png")
        plt.close()

    def _plot_fraud_vs_amount(self) -> None:
        """Generates and saves a boxplot comparing transaction amounts by class."""
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE, dpi=FIGURE_DPI)

        # Box plot
        groups = [
            self.df[self.df[TARGET_COLUMN] == 0]["Amount"].values,
            self.df[self.df[TARGET_COLUMN] == 1]["Amount"].values,
        ]
        bp = axes[0].boxplot(groups, labels=["Normal (0)", "Fraud (1)"], patch_artist=True, notch=True)
        for patch, color in zip(bp["boxes"], PALETTE_LIST):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_title("Amount Distribution by Class (Boxplot)", fontsize=13)
        axes[0].set_ylabel("Amount (USD)")

        # Log-scale box plot to better show Fraud distribution
        axes[1].boxplot(groups, labels=["Normal (0)", "Fraud (1)"], patch_artist=True)
        for patch, color in zip(axes[1].findobj(plt.matplotlib.patches.PathPatch), PALETTE_LIST):
            patch.set_facecolor(color)
        axes[1].set_yscale("log")
        axes[1].set_title("Amount by Class (Log Scale)", fontsize=13)
        axes[1].set_ylabel("Amount (USD, log scale)")

        plt.suptitle("Fraud vs Transaction Amount", fontsize=15, y=1.01)
        plt.tight_layout()
        self._save_figure("fraud_vs_amount.png")
        plt.close()

    def _plot_correlation_heatmap(self) -> None:
        """Generates and saves a correlation heatmap for all numeric columns."""
        fig, ax = plt.subplots(figsize=(16, 13), dpi=FIGURE_DPI)
        corr = self.df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.3,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=15, pad=14)
        plt.tight_layout()
        self._save_figure("correlation_heatmap.png")
        plt.close()

    def _plot_pca_scatter(self) -> None:
        """Generates and saves a 2D PCA scatter plot coloured by class.

        Uses the V1–V28 PCA features plus Amount and Time. Samples 10k rows for
        speed if the dataset is large.
        """
        feature_cols = [c for c in self.df.columns if c != TARGET_COLUMN]
        sample = self.df.sample(n=min(10_000, len(self.df)), random_state=42)
        X = sample[feature_cols].values
        y = sample[TARGET_COLUMN].values

        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE, dpi=FIGURE_DPI)
        for cls, label in [(0, "Normal"), (1, "Fraud")]:
            mask = y == cls
            ax.scatter(
                components[mask, 0],
                components[mask, 1],
                c=PALETTE[cls],
                label=f"{label} (n={mask.sum():,})",
                alpha=0.5,
                s=10,
                edgecolors="none",
            )
        ax.set_title(
            f"PCA Scatter — PC1 ({explained[0]:.1%}) vs PC2 ({explained[1]:.1%})",
            fontsize=13,
        )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(markerscale=2, fontsize=11)
        plt.tight_layout()
        self._save_figure("pca_scatter.png")
        plt.close()

    def _plot_feature_statistics(self) -> None:
        """Generates and saves a heatmap of per-feature descriptive statistics."""
        feature_cols = [c for c in self.df.columns if c != TARGET_COLUMN]
        stats = self.df[feature_cols].describe().T[["mean", "std", "min", "max"]]
        stats_norm = (stats - stats.min()) / (stats.max() - stats.min() + 1e-9)

        fig, ax = plt.subplots(figsize=(10, 14), dpi=FIGURE_DPI)
        sns.heatmap(
            stats_norm,
            annot=False,
            cmap="viridis",
            linewidths=0.3,
            ax=ax,
            cbar_kws={"label": "Normalised Value"},
        )
        ax.set_title("Normalised Feature Statistics (mean, std, min, max)", fontsize=13)
        ax.set_xlabel("Statistic")
        plt.tight_layout()
        self._save_figure("feature_statistics.png")
        plt.close()

    def _plot_outlier_summary(self) -> None:
        """Generates and saves a bar chart of IQR-based outlier counts per feature."""
        feature_cols = [c for c in self.df.columns if c != TARGET_COLUMN]
        outlier_counts: dict[str, int] = {}
        for col in feature_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_counts[col] = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())

        sorted_items = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
        cols, counts = zip(*sorted_items)

        fig, ax = plt.subplots(figsize=(14, 7), dpi=FIGURE_DPI)
        ax.bar(cols, counts, color="#9b59b6", edgecolor="white", linewidth=0.5)
        ax.set_title("IQR-Based Outlier Counts per Feature", fontsize=14)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Outlier Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        self._save_figure("outlier_summary.png")
        plt.close()

    # -------------------------------------------------------------------------
    # Report generation
    # -------------------------------------------------------------------------

    def _save_eda_report(self) -> None:
        """Assembles and writes the eda_report.md markdown file."""
        counts = self.df[TARGET_COLUMN].value_counts().sort_index()
        total = len(self.df)
        fraud_pct = counts.get(1, 0) / total * 100
        desc = self.df.describe().T

        figures_section = "\n".join(
            f"- `reports/figures/{fig}`" for fig in self._saved_figures
        )

        report = textwrap.dedent(f"""\
            # EDA Report — Credit Card Fraud Detection

            **Generated by FraudLens AI preprocessing pipeline**

            ---

            ## Dataset Summary

            | Metric | Value |
            |--------|-------|
            | Total Rows | {total:,} |
            | Total Columns | {self.df.shape[1]} |
            | Feature Columns | {self.df.shape[1] - 1} |
            | Target Column | `{TARGET_COLUMN}` |
            | Memory Usage | {self.df.memory_usage(deep=True).sum() / 1_048_576:.2f} MB |

            ---

            ## Class Distribution

            | Class | Count | Percentage |
            |-------|-------|------------|
            | Normal (0) | {counts.get(0, 0):,} | {100 - fraud_pct:.4f}% |
            | Fraud (1) | {counts.get(1, 0):,} | {fraud_pct:.4f}% |

            > **Class Imbalance Ratio**: {counts.get(0, 0) / max(counts.get(1, 0), 1):.1f}:1 (Normal:Fraud)
            >
            > SMOTE will be applied to the training split only to address this imbalance.

            ---

            ## Feature Statistics

            | Feature | Mean | Std | Min | Max |
            |---------|------|-----|-----|-----|
            {self._build_stats_table(desc)}

            ---

            ## Generated Figures

            {figures_section}

            ---

            ## Preprocessing Notes

            - **Amount** and **Time** will be scaled using `StandardScaler`.
            - **V1–V28** are already PCA-transformed and will NOT be scaled.
            - SMOTE oversampling is applied to training data only (never validation or test).
            - Stratified splitting is used to preserve class ratios across all splits.
        """)

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"EDA report saved to: {self.report_path}")

    def _build_stats_table(self, desc: pd.DataFrame) -> str:
        """Builds a markdown table string for feature statistics.

        Args:
            desc: Transposed describe() DataFrame with mean, std, min, max.

        Returns:
            str: Markdown table rows as a single string.
        """
        rows = []
        for col in desc.index:
            row = desc.loc[col]
            rows.append(
                f"| {col} | {row.get('mean', 0):.4f} | {row.get('std', 0):.4f} "
                f"| {row.get('min', 0):.4f} | {row.get('max', 0):.4f} |"
            )
        return "\n            ".join(rows)

    def _save_figure(self, filename: str) -> None:
        """Saves the current matplotlib figure to the figures directory.

        Args:
            filename: The output filename (e.g., 'class_imbalance.png').
        """
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        self._saved_figures.append(filename)
        logger.debug(f"Saved figure: {output_path}")
