# eval_script.py
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu, ttest_ind
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

warnings.filterwarnings("ignore")


def comprehensive_distribution_analysis(
    honest_scores,
    sneaky_scores,
    figsize=(24, 20),
    save_path=None,
    honest_label="Honest",
    sneaky_label="Sneaky",
    threshold_range=None,
    n_thresholds=100,
    higher_is_sneaky=None,
):
    """
    Comprehensive statistical analysis and visualization of two probability distributions
    with enhanced ROC/AUC analysis, precision-recall curves, and tail behavior analysis.

    Parameters:
    -----------
    honest_scores : list/array
        Classifier scores for honest condition
    sneaky_scores : list/array
        Classifier scores for sneaky condition
    figsize : tuple
        Figure size for the plot
    save_path : str
        Path to save the figure (optional)
    honest_label : str
        Label for honest class
    sneaky_label : str
        Label for sneaky class
    threshold_range : tuple
        Range for threshold analysis (min, max). If None, auto-determined from data.
    n_thresholds : int
        Number of thresholds to analyze
    higher_is_sneaky : bool
        If True, higher scores indicate sneaky behavior. If False, higher scores indicate honest behavior.
        If None, automatically determined based on mean differences.
    """

    # Convert to numpy arrays and validate
    honest = np.array(honest_scores)
    sneaky = np.array(sneaky_scores)

    if len(honest) == 0 or len(sneaky) == 0:
        raise ValueError("Both honest and sneaky scores must contain data")

    # Auto-determine score interpretation if not specified
    if higher_is_sneaky is None:
        higher_is_sneaky = np.mean(sneaky) > np.mean(honest)
        print(f"Auto-detected: Higher scores indicate {'sneaky' if higher_is_sneaky else 'honest'} behavior")
        print(f"  Mean honest score: {np.mean(honest):.4f}")
        print(f"  Mean sneaky score: {np.mean(sneaky):.4f}")

    # Auto-determine threshold range if not provided
    if threshold_range is None:
        all_scores = np.concatenate([honest, sneaky])
        score_min, score_max = np.min(all_scores), np.max(all_scores)
        score_range = score_max - score_min
        threshold_range = (score_min + 0.1 * score_range, score_max - 0.1 * score_range)

    # Prepare data for ROC/PR analysis
    # y_true defines the true labels, with the "positive class" being 1.
    # y_scores_roc should be scores where a higher value indicates a higher likelihood of belonging to the positive class.
    if higher_is_sneaky:
        # Sneaky is positive class (1), honest is negative class (0).
        # Higher original scores already indicate sneaky.
        y_true = np.concatenate([np.zeros(len(honest)), np.ones(len(sneaky))])
        y_scores_roc = np.concatenate([honest, sneaky])  # Original scores are correct for this.
    else:
        # Honest is positive class (1), sneaky is negative class (0).
        # Higher original scores already indicate honest.
        y_true = np.concatenate([np.ones(len(honest)), np.zeros(len(sneaky))])
        y_scores_roc = np.concatenate(
            [honest, sneaky]
        )  # Original scores are correct. The previous negation was an error.
        # The line "y_scores = -y_scores" has been removed.

    # Create comprehensive figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 5, hspace=0.35, wspace=0.3)

    # Enhanced color scheme
    honest_color = "#2E86AB"
    sneaky_color = "#F24236"
    overlap_color = "#A23B72"
    roc_color = "#FF6B35"
    pr_color = "#004E89"

    # 1. Main histogram with KDE overlay (top left, spanning 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    # Determine optimal binning
    n_bins_honest = max(10, min(50, int(np.sqrt(len(honest)))))
    n_bins_sneaky = max(10, min(50, int(np.sqrt(len(sneaky)))))
    n_bins = max(n_bins_honest, n_bins_sneaky)

    # Create histograms
    alpha = 0.6
    ax1.hist(
        honest,
        bins=n_bins,
        alpha=alpha,
        density=True,
        color=honest_color,
        label=f"{honest_label} (n={len(honest)})",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.hist(
        sneaky,
        bins=n_bins,
        alpha=alpha,
        density=True,
        color=sneaky_color,
        label=f"{sneaky_label} (n={len(sneaky)})",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add KDE curves
    x_range = np.linspace(min(np.min(honest), np.min(sneaky)), max(np.max(honest), np.max(sneaky)), 200)

    if len(honest) > 1:
        kde_honest = gaussian_kde(honest)
        ax1.plot(
            x_range,
            kde_honest(x_range),
            color=honest_color,
            linewidth=3,
            alpha=0.8,
            linestyle="--",
        )

    if len(sneaky) > 1:
        kde_sneaky = gaussian_kde(sneaky)
        ax1.plot(
            x_range,
            kde_sneaky(x_range),
            color=sneaky_color,
            linewidth=3,
            alpha=0.8,
            linestyle="--",
        )

    ax1.set_xlabel("Classifier Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax1.set_title("Distribution Comparison with KDE", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. ROC Curve (top middle)
    ax2 = fig.add_subplot(gs[0, 2])
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores_roc)
    roc_auc = auc(fpr, tpr)

    ax2.plot(fpr, tpr, color=roc_color, lw=3, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", alpha=0.5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax2.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    ax2.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add optimal threshold point (Youden's J)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    ax2.plot(
        fpr[optimal_idx],
        tpr[optimal_idx],
        "o",
        color="red",
        markersize=8,
        label=f"Optimal (θ={optimal_threshold:.3f})",
    )
    ax2.legend(loc="lower right", fontsize=8)

    # 3. Precision-Recall Curve (top right)
    ax3 = fig.add_subplot(gs[0, 3])
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores_roc)
    avg_precision = average_precision_score(y_true, y_scores_roc)

    ax3.plot(
        recall,
        precision,
        color=pr_color,
        lw=3,
        label=f"PR Curve (AP = {avg_precision:.4f})",
    )

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax3.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline ({baseline:.3f})",
    )

    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel("Recall", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax3.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Threshold Analysis (top far right)
    ax4 = fig.add_subplot(gs[0, 4])

    # Calculate metrics across thresholds
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    # Convert thresholds back to original score space if needed
    # This display_thresholds logic might need review if threshold_range is always original scale.
    # For now, assume threshold_range was derived from y_scores which might have been negated.
    # With the fix, y_scores_roc is always original scale, so threshold_range should be too.
    # Let's simplify: thresholds for analysis should be on the original score scale.

    metrics = {
        "threshold": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fpr": [],
        "fnr": [],
        "accuracy": [],
        "specificity": [],
    }

    for thresh in thresholds:
        # Prediction rule: if score >= thresh, predict positive class.
        # y_scores_roc are original scores (higher = more likely positive defined by y_true).
        y_pred = (y_scores_roc >= thresh).astype(int)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
            # Handle case where only one class is predicted
            continue

        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = (
            2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        )
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
        accuracy_val = (tp + tn) / (tp + tn + fp + fn)

        metrics["threshold"].append(thresh)
        metrics["precision"].append(precision_val)
        metrics["recall"].append(recall_val)
        metrics["f1"].append(f1_val)
        metrics["fpr"].append(fpr_val)
        metrics["fnr"].append(fnr_val)
        metrics["accuracy"].append(accuracy_val)
        metrics["specificity"].append(specificity_val)

    if len(metrics["threshold"]) > 0:
        ax4.plot(
            metrics["threshold"],
            metrics["precision"],
            label="Precision",
            color="blue",
            linewidth=2,
        )
        ax4.plot(
            metrics["threshold"],
            metrics["recall"],
            label="Recall",
            color="green",
            linewidth=2,
        )
        ax4.plot(
            metrics["threshold"],
            metrics["f1"],
            label="F1-Score",
            color="red",
            linewidth=2,
        )
        ax4.plot(
            metrics["threshold"],
            metrics["specificity"],
            label="Specificity",
            color="purple",
            linewidth=2,
        )

        # Mark optimal F1 threshold
        optimal_f1_idx = np.argmax(metrics["f1"])
        optimal_f1_thresh = metrics["threshold"][optimal_f1_idx]
        ax4.axvline(
            x=optimal_f1_thresh,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Optimal F1 (θ={optimal_f1_thresh:.3f})",
        )
    else:
        optimal_f1_thresh = np.median(thresholds)

    ax4.set_xlabel("Threshold", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
    ax4.set_title("Threshold Analysis", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Box plots with violin plots (second row, left)
    ax5 = fig.add_subplot(gs[1, 2])

    # Violin plot
    parts = ax5.violinplot([honest, sneaky], positions=[1, 2], showmeans=True, showmedians=True, widths=0.6)
    parts["bodies"][0].set_facecolor(honest_color)
    parts["bodies"][1].set_facecolor(sneaky_color)
    parts["bodies"][0].set_alpha(0.7)
    parts["bodies"][1].set_alpha(0.7)

    # Box plot overlay
    bp = ax5.boxplot(
        [honest, sneaky],
        positions=[1, 2],
        widths=0.3,
        patch_artist=True,
        showfliers=True,
    )
    bp["boxes"][0].set_facecolor(honest_color)
    bp["boxes"][1].set_facecolor(sneaky_color)
    bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_alpha(0.8)

    ax5.set_xticks([1, 2])
    ax5.set_xticklabels([honest_label, sneaky_label], fontweight="bold")
    ax5.set_ylabel("Classifier Score", fontsize=12, fontweight="bold")
    ax5.set_title("Distribution Shapes", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # 6. Tail Analysis (second row, middle-right)
    ax6 = fig.add_subplot(gs[1, 3:])

    # Calculate tail statistics
    percentiles = [1, 5, 10, 90, 95, 99]
    tail_stats = pd.DataFrame(
        {
            "Percentile": percentiles,
            honest_label: [np.percentile(honest, p) for p in percentiles],
            sneaky_label: [np.percentile(sneaky, p) for p in percentiles],
        }
    )

    # Plot tail comparison
    x_pos = np.arange(len(percentiles))
    width = 0.35

    bars1 = ax6.bar(
        x_pos - width / 2,
        tail_stats[honest_label],
        width,
        label=honest_label,
        color=honest_color,
        alpha=0.7,
    )
    bars2 = ax6.bar(
        x_pos + width / 2,
        tail_stats[sneaky_label],
        width,
        label=sneaky_label,
        color=sneaky_color,
        alpha=0.7,
    )

    ax6.set_xlabel("Percentile", fontsize=12, fontweight="bold")
    ax6.set_ylabel("Score Value", fontsize=12, fontweight="bold")
    ax6.set_title("Tail Behavior Analysis", fontsize=12, fontweight="bold")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f"{p}%" for p in percentiles])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 7. Statistical summary table (third row, left)
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.axis("off")

    # Calculate statistics
    stats_data = {
        "Statistic": [
            "Count",
            "Mean",
            "Std Dev",
            "Median",
            "IQR",
            "Skewness",
            "Kurtosis",
            "Min",
            "Max",
            "1st Pct",
            "99th Pct",
        ],
        honest_label: [
            len(honest),
            np.mean(honest),
            np.std(honest, ddof=1) if len(honest) > 1 else 0,
            np.median(honest),
            np.percentile(honest, 75) - np.percentile(honest, 25),
            stats.skew(honest) if len(honest) > 3 else 0,
            stats.kurtosis(honest) if len(honest) > 3 else 0,
            np.min(honest),
            np.max(honest),
            np.percentile(honest, 1),
            np.percentile(honest, 99),
        ],
        sneaky_label: [
            len(sneaky),
            np.mean(sneaky),
            np.std(sneaky, ddof=1) if len(sneaky) > 1 else 0,
            np.median(sneaky),
            np.percentile(sneaky, 75) - np.percentile(sneaky, 25),
            stats.skew(sneaky) if len(sneaky) > 3 else 0,
            stats.kurtosis(sneaky) if len(sneaky) > 3 else 0,
            np.min(sneaky),
            np.max(sneaky),
            np.percentile(sneaky, 1),
            np.percentile(sneaky, 99),
        ],
    }

    stats_df = pd.DataFrame(stats_data)

    # Format numbers for display
    for col in [honest_label, sneaky_label]:
        stats_df[col] = stats_df[col].apply(
            lambda x: (
                f"{x:.0f}"
                if isinstance(x, (int, float)) and x == int(x) and abs(x) > 10
                else (
                    f"{x:.4f}"
                    if isinstance(x, (int, float)) and abs(x) < 1
                    else f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
                )
            )
        )

    table = ax7.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Color code the table
    for i in range(len(stats_df)):
        table[(i + 1, 1)].set_facecolor("#E8F4FD")  # Light blue for honest
        table[(i + 1, 2)].set_facecolor("#FDE8E8")  # Light red for sneaky

    ax7.set_title("Descriptive Statistics", fontsize=14, fontweight="bold", pad=20)

    # 8. Classification Performance at Optimal Threshold (third row, middle)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")

    # Use optimal F1 threshold for detailed analysis
    # optimal_f1_thresh is from the 'threshold' column of 'metrics' table, which is original score scale.
    y_pred_optimal = (y_scores_roc >= optimal_f1_thresh).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    except ValueError:
        # Handle edge case
        tn = fp = fn = tp = 0

    # Calculate all metrics
    precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_opt = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_opt = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_opt = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy_opt = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    perf_data = [
        ["Threshold", f"{optimal_f1_thresh:.4f}", ""],
        ["True Positives", f"{tp}", f"{tp/(tp+fn):.1%}" if (tp + fn) > 0 else "0.0%"],
        ["False Positives", f"{fp}", f"{fp/(tn+fp):.1%}" if (tn + fp) > 0 else "0.0%"],
        ["True Negatives", f"{tn}", f"{tn/(tn+fp):.1%}" if (tn + fp) > 0 else "0.0%"],
        ["False Negatives", f"{fn}", f"{fn/(tp+fn):.1%}" if (tp + fn) > 0 else "0.0%"],
        ["False Positives Rate", f"{fpr_opt:.4f}", ""],
        ["False Negatives Rate", f"{fnr_opt:.4f}", ""],
        ["Precision", f"{precision_opt:.4f}", ""],
        ["Recall/Sensitivity", f"{recall_opt:.4f}", ""],
        ["Specificity", f"{specificity:.4f}", ""],
        ["F1-Score", f"{f1_opt:.4f}", ""],
        ["Accuracy", f"{accuracy_opt:.4f}", ""],
    ]

    perf_table = ax8.table(
        cellText=perf_data,
        colLabels=["Metric", "Value", "Rate"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    perf_table.auto_set_font_size(False)
    perf_table.set_fontsize(9)
    perf_table.scale(1, 1.5)

    ax8.set_title(
        "Classification Performance\n(Optimal F1 Threshold)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    # 9. Confusion Matrix Heatmap (third row, right)
    ax9 = fig.add_subplot(gs[2, 3:])

    cm = confusion_matrix(y_true, y_pred_optimal)

    # Create labels based on score interpretation
    if higher_is_sneaky:
        labels = [honest_label, sneaky_label]
    else:
        labels = [sneaky_label, honest_label]

    # Create custom colormap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax9,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )

    ax9.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax9.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax9.set_title("Confusion Matrix", fontsize=12, fontweight="bold")

    # 10. False Positive/Negative Analysis (fourth row, left)
    ax10 = fig.add_subplot(gs[3, 0:2])

    # Identify false positives and false negatives
    # y_pred_optimal is based on y_scores_roc and optimal_f1_thresh (original scale)
    fp_mask = (y_true == 0) & (y_pred_optimal == 1)
    fn_mask = (y_true == 1) & (y_pred_optimal == 0)

    fp_scores = y_scores_roc[fp_mask]
    fn_scores = y_scores_roc[fn_mask]
    tp_scores = y_scores_roc[(y_true == 1) & (y_pred_optimal == 1)]
    tn_scores = y_scores_roc[(y_true == 0) & (y_pred_optimal == 0)]

    # Plot score distributions by prediction outcome
    bins = np.linspace(min(y_scores_roc), max(y_scores_roc), 30)

    ax10.hist(
        tn_scores,
        bins=bins,
        alpha=0.7,
        color="lightblue",
        label=f"True Negatives (n={len(tn_scores)})",
        density=True,
    )
    ax10.hist(
        tp_scores,
        bins=bins,
        alpha=0.7,
        color="lightcoral",
        label=f"True Positives (n={len(tp_scores)})",
        density=True,
    )

    if len(fp_scores) > 0:
        ax10.hist(
            fp_scores,
            bins=bins,
            alpha=0.9,
            color="red",
            label=f"False Positives (n={len(fp_scores)})",
            density=True,
            edgecolor="black",
            linewidth=1,
        )

    if len(fn_scores) > 0:
        ax10.hist(
            fn_scores,
            bins=bins,
            alpha=0.9,
            color="blue",
            label=f"False Negatives (n={len(fn_scores)})",
            density=True,
            edgecolor="black",
            linewidth=1,
        )

    # Add threshold line (convert back to original scale if needed)
    # optimal_f1_thresh is already on the original score scale.
    # thresh_display = -optimal_f1_thresh if not higher_is_sneaky else optimal_f1_thresh # Old logic
    thresh_display = optimal_f1_thresh  # Simplified: it's always original scale now.
    ax10.axvline(
        x=thresh_display,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({thresh_display:.3f})",
    )

    ax10.set_xlabel("Classifier Score", fontsize=12, fontweight="bold")
    ax10.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax10.set_title("Error Analysis: False Positives & Negatives", fontsize=12, fontweight="bold")
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3)

    # 11. Distribution Overlap Analysis (fourth row, right)
    ax11 = fig.add_subplot(gs[3, 2:])

    # Calculate overlap coefficient
    overlap_coeff = np.nan  # Initialize to ensure it's always defined
    if len(honest) > 1 and len(sneaky) > 1:
        kde_honest = gaussian_kde(honest)
        kde_sneaky = gaussian_kde(sneaky)

        density_honest = kde_honest(x_range)
        density_sneaky = kde_sneaky(x_range)

        # Plot densities
        ax11.fill_between(x_range, density_honest, alpha=0.5, color=honest_color, label=honest_label)
        ax11.fill_between(x_range, density_sneaky, alpha=0.5, color=sneaky_color, label=sneaky_label)

        # Highlight overlap
        overlap = np.minimum(density_honest, density_sneaky)
        ax11.fill_between(x_range, overlap, alpha=0.8, color=overlap_color, label="Overlap")

        # Calculate overlap coefficient
        overlap_coeff = np.trapz(overlap, x_range)

        # Add threshold line (convert back to original scale if needed)
        thresh_display = -optimal_f1_thresh if not higher_is_sneaky else optimal_f1_thresh
        ax11.axvline(
            x=thresh_display,
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Optimal Threshold",
        )

        ax11.text(
            0.05,
            0.95,
            f"Overlap Coefficient: {overlap_coeff:.4f}",
            transform=ax11.transAxes,
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax11.set_xlabel("Classifier Score", fontsize=12, fontweight="bold")
    ax11.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax11.set_title("Distribution Overlap with Threshold", fontsize=12, fontweight="bold")
    ax11.legend(fontsize=10)
    ax11.grid(True, alpha=0.3)

    # 12. Extreme Values Analysis (bottom row)
    ax12 = fig.add_subplot(gs[4, :])

    # Identify extreme values (outliers) using IQR method
    def identify_outliers(data, method="iqr"):
        if method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data))
            return z_scores > 3
        return np.zeros(len(data), dtype=bool)

    honest_outliers = identify_outliers(honest)
    sneaky_outliers = identify_outliers(sneaky)

    # Create scatter plot of all values with outliers highlighted
    honest_normal = honest[~honest_outliers]
    honest_extreme = honest[honest_outliers]
    sneaky_normal = sneaky[~sneaky_outliers]
    sneaky_extreme = sneaky[sneaky_outliers]

    # Plot normal values
    ax12.scatter(
        honest_normal,
        np.random.normal(0, 0.05, len(honest_normal)),
        alpha=0.6,
        color=honest_color,
        s=20,
        label=f"{honest_label} Normal",
    )
    ax12.scatter(
        sneaky_normal,
        np.random.normal(1, 0.05, len(sneaky_normal)),
        alpha=0.6,
        color=sneaky_color,
        s=20,
        label=f"{sneaky_label} Normal",
    )

    # Plot extreme values
    if len(honest_extreme) > 0:
        ax12.scatter(
            honest_extreme,
            np.random.normal(0, 0.05, len(honest_extreme)),
            alpha=0.9,
            color="darkblue",
            s=60,
            marker="^",
            label=f"{honest_label} Outliers (n={len(honest_extreme)})",
        )

    if len(sneaky_extreme) > 0:
        ax12.scatter(
            sneaky_extreme,
            np.random.normal(1, 0.05, len(sneaky_extreme)),
            alpha=0.9,
            color="darkred",
            s=60,
            marker="v",
            label=f"{sneaky_label} Outliers (n={len(sneaky_extreme)})",
        )

    # Add threshold line
    # thresh_display = -optimal_f1_thresh if not higher_is_sneaky else optimal_f1_thresh # Old logic
    thresh_display_extreme = optimal_f1_thresh  # optimal_f1_thresh is on original scale
    ax12.axvline(
        x=thresh_display_extreme,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Optimal Threshold",
    )

    ax12.set_xlabel("Classifier Score", fontsize=12, fontweight="bold")
    ax12.set_ylabel("Class (with jitter)", fontsize=12, fontweight="bold")
    ax12.set_title("Extreme Values and Outliers Analysis", fontsize=14, fontweight="bold")
    ax12.set_yticks([0, 1])
    ax12.set_yticklabels([honest_label, sneaky_label])
    ax12.legend(fontsize=10, loc="upper right")
    ax12.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(
        f"Enhanced Distribution Analysis: {honest_label} vs {sneaky_label} Scores\n"
        f"Classifier Score Analysis with ROC/AUC, Precision-Recall, and Statistical Tests",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print comprehensive statistical insights
    print("\n" + "=" * 100)
    print("COMPREHENSIVE CLASSIFIER SCORE ANALYSIS")
    print("=" * 100)

    print("\n📊 SAMPLE SIZES:")
    print(f"   {honest_label}: {len(honest):,} observations")
    print(f"   {sneaky_label}: {len(sneaky):,} observations")
    print(f"   Total: {len(y_true):,} observations")
    print(f"   Class Balance: {np.mean(y_true):.1%} positive class, {1-np.mean(y_true):.1%} negative class")

    print("\n🎯 CLASSIFICATION PERFORMANCE:")
    print(
        f"   ROC-AUC: {roc_auc:.4f} ({'Excellent' if roc_auc > 0.9 else 'Good' if roc_auc > 0.8 else 'Fair' if roc_auc > 0.7 else 'Poor'})"
    )
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Optimal Threshold: {optimal_f1_thresh:.4f}")
    print(f"   F1-Score at Optimal: {f1_opt:.4f}")
    print(f"   Accuracy at Optimal: {accuracy_opt:.4f}")

    # Statistical tests
    print("\n📈 STATISTICAL TESTS:")

    # Mann-Whitney U test (non-parametric)
    try:
        mw_stat, mw_p = mannwhitneyu(honest, sneaky, alternative="two-sided")
        print(f"   Mann-Whitney U test: statistic={mw_stat:.3f}, p-value={mw_p:.6f}")
    except Exception as e:
        print(f"   Mann-Whitney U test: Failed ({e})")

    # Independent t-test
    try:
        t_stat, t_p = ttest_ind(honest, sneaky)
        print(f"   Independent t-test: t-statistic={t_stat:.3f}, p-value={t_p:.6f}")
    except Exception as e:
        print(f"   Independent t-test: Failed ({e})")

    # Kolmogorov-Smirnov test
    try:
        ks_stat, ks_p = ks_2samp(honest, sneaky)
        print(f"   Kolmogorov-Smirnov test: statistic={ks_stat:.3f}, p-value={ks_p:.6f}")
    except Exception as e:
        print(f"   Kolmogorov-Smirnov test: Failed ({e})")

    print("\n📊 DISTRIBUTION CHARACTERISTICS:")
    print(f"   {honest_label} - Mean: {np.mean(honest):.4f}, Std: {np.std(honest):.4f}")
    print(f"   {sneaky_label} - Mean: {np.mean(sneaky):.4f}, Std: {np.std(sneaky):.4f}")
    print(f"   Mean Difference: {np.mean(sneaky) - np.mean(honest):.4f}")
    print(
        f"   Effect Size (Cohen's d): {(np.mean(sneaky) - np.mean(honest)) / np.sqrt((np.var(sneaky) + np.var(honest)) / 2):.4f}"
    )

    if len(honest) > 1 and len(sneaky) > 1:
        print(f"   Overlap Coefficient: {overlap_coeff:.4f}")

    print("\n✅ Analysis complete!")

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "optimal_threshold": optimal_f1_thresh,
        "f1_score": f1_opt,
        "accuracy": accuracy_opt,
        "stats_df": stats_df,
        "higher_is_sneaky": higher_is_sneaky,
    }
