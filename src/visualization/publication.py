"""Publication-ready figure generation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class PublicationFigures:
    """Generate publication-ready figures."""

    def __init__(self, style: str = 'publication',
                 font_family: str = 'Arial',
                 font_size: int = 12):
        self.font_family = font_family
        self.font_size = font_size

        # Set publication style
        plt.rcParams.update({
            'font.family': font_family,
            'font.size': font_size,
            'axes.titlesize': font_size + 2,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 1,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

    def figure_1_baseline_comparison(self,
                                     data: pd.DataFrame,
                                     continuous_vars: List[str],
                                     group_col: str,
                                     save_path: str = None) -> plt.Figure:
        """Create Figure 1: Baseline characteristics comparison.

        Args:
            data: DataFrame with the data
            continuous_vars: List of continuous variables
            group_col: Grouping variable
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_vars = len(continuous_vars)
        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten() if n_vars > 1 else [axes]

        colors = ['#4878D0', '#EE854A']  # Blue and orange

        for i, var in enumerate(continuous_vars):
            ax = axes[i]

            groups = data[group_col].unique()
            positions = np.arange(len(groups))
            width = 0.6

            for j, group in enumerate(groups):
                group_data = data[data[group_col] == group][var].dropna()
                bp = ax.boxplot([group_data], positions=[j], widths=width,
                               patch_artist=True)
                bp['boxes'][0].set_facecolor(colors[j % len(colors)])
                bp['boxes'][0].set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(groups)
            ax.set_ylabel(var)
            ax.set_title(var, fontweight='bold')

        # Hide unused axes
        for i in range(len(continuous_vars), len(axes)):
            axes[i].set_visible(False)

        # Add legend
        legend_patches = [mpatches.Patch(color=colors[i], label=str(g), alpha=0.7)
                         for i, g in enumerate(groups)]
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')

        return fig

    def figure_primary_outcome(self,
                              data: pd.DataFrame,
                              outcome_col: str,
                              group_col: str,
                              test_result: Dict,
                              ylabel: str = None,
                              save_path: str = None) -> plt.Figure:
        """Create primary outcome comparison figure with statistics.

        Args:
            data: DataFrame with the data
            outcome_col: Outcome variable
            group_col: Grouping variable
            test_result: Test result dictionary
            ylabel: Y-axis label
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        groups = data[group_col].unique()
        colors = ['#4878D0', '#EE854A']

        # Bar plot with error bars
        means = []
        sems = []
        for group in groups:
            group_data = data[data[group_col] == group][outcome_col].dropna()
            means.append(group_data.mean())
            sems.append(group_data.sem())

        x = np.arange(len(groups))
        bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors[:len(groups)],
                     edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add significance indicator
        p_value = test_result.get('p_value', 1)
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'

        # Draw significance bracket
        y_max = max(means) + max(sems) * 1.5
        ax.plot([0, 0, 1, 1], [y_max, y_max + 0.02 * y_max, y_max + 0.02 * y_max, y_max],
               'k-', linewidth=1.5)
        ax.text(0.5, y_max + 0.04 * y_max, sig_text, ha='center', va='bottom',
               fontsize=self.font_size + 2)

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel or outcome_col)
        ax.set_ylim(0, y_max * 1.15)

        # Add sample sizes
        for i, (mean, group) in enumerate(zip(means, groups)):
            n = len(data[data[group_col] == group][outcome_col].dropna())
            ax.text(i, -0.05 * max(means), f'n={n}', ha='center', va='top',
                   fontsize=self.font_size - 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')

        return fig

    def figure_correlation(self,
                          data: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          correlation_result: Dict,
                          xlabel: str = None,
                          ylabel: str = None,
                          save_path: str = None) -> plt.Figure:
        """Create correlation figure with regression line.

        Args:
            data: DataFrame with the data
            x_col: X variable
            y_col: Y variable
            correlation_result: Correlation result dictionary
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        clean_data = data[[x_col, y_col]].dropna()

        # Scatter plot
        ax.scatter(clean_data[x_col], clean_data[y_col],
                  alpha=0.6, color='#4878D0', edgecolor='white', s=50)

        # Regression line
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(clean_data[x_col], clean_data[y_col])
        x_line = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

        # 95% CI for regression line
        n = len(clean_data)
        x_mean = clean_data[x_col].mean()
        ss_x = ((clean_data[x_col] - x_mean) ** 2).sum()
        mse = ((clean_data[y_col] - (slope * clean_data[x_col] + intercept)) ** 2).sum() / (n - 2)

        t_val = stats.t.ppf(0.975, n - 2)
        y_pred = slope * x_line + intercept

        se_y = np.sqrt(mse * (1/n + (x_line - x_mean)**2 / ss_x))
        ci_upper = y_pred + t_val * se_y
        ci_lower = y_pred - t_val * se_y

        ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='red')

        # Statistics annotation
        r_val = correlation_result.get('correlation', r)
        p_val = correlation_result.get('p_value', p)
        ax.annotate(f'r = {r_val:.3f}\np {"< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"}',
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=self.font_size, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

        ax.set_xlabel(xlabel or x_col)
        ax.set_ylabel(ylabel or y_col)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')

        return fig

    def figure_survival(self,
                       time: pd.Series,
                       event: pd.Series,
                       group: pd.Series = None,
                       xlabel: str = 'Time',
                       ylabel: str = 'Survival Probability',
                       title: str = None,
                       at_risk_table: bool = True,
                       save_path: str = None) -> plt.Figure:
        """Create publication-quality Kaplan-Meier plot.

        Args:
            time: Time to event
            event: Event indicator
            group: Grouping variable (optional)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            at_risk_table: Show number at risk table
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.plotting import add_at_risk_counts

            if at_risk_table:
                fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(8, 6),
                                                   gridspec_kw={'height_ratios': [4, 1]})
            else:
                fig, ax = plt.subplots(figsize=(8, 5))

            colors = ['#4878D0', '#EE854A', '#6ACC64', '#D65F5F']
            kmfs = []

            if group is not None:
                groups = group.dropna().unique()
                for i, g in enumerate(groups):
                    mask = group == g
                    kmf = KaplanMeierFitter()
                    kmf.fit(time[mask], event[mask], label=str(g))
                    kmf.plot_survival_function(ax=ax, ci_show=True,
                                              color=colors[i % len(colors)], linewidth=2)
                    kmfs.append(kmf)
            else:
                kmf = KaplanMeierFitter()
                kmf.fit(time, event, label='Overall')
                kmf.plot_survival_function(ax=ax, ci_show=True,
                                          color=colors[0], linewidth=2)
                kmfs.append(kmf)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title:
                ax.set_title(title)
            ax.set_ylim(0, 1.05)
            ax.legend(loc='lower left')

            if at_risk_table and len(kmfs) > 0:
                add_at_risk_counts(*kmfs, ax=ax)

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, bbox_inches='tight')

            return fig

        except ImportError:
            print("lifelines package required for survival plots")
            return None

    def figure_forest(self,
                     subgroups: List[Dict],
                     overall: Dict,
                     xlabel: str = 'Effect Size (95% CI)',
                     null_value: float = 0,
                     log_scale: bool = False,
                     save_path: str = None) -> plt.Figure:
        """Create publication-quality forest plot.

        Args:
            subgroups: List of dicts with 'name', 'effect', 'ci_lower', 'ci_upper', 'n'
            overall: Dict with overall effect
            xlabel: X-axis label
            null_value: Null effect value
            log_scale: Use log scale for x-axis
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_subgroups = len(subgroups)
        fig, ax = plt.subplots(figsize=(10, max(4, n_subgroups * 0.5 + 2)))

        y_positions = []
        y_current = n_subgroups + 1

        # Plot subgroups
        for sg in subgroups:
            y_positions.append(y_current)

            # Point estimate
            ax.plot(sg['effect'], y_current, 'ko', markersize=8)

            # CI line
            ax.hlines(y_current, sg['ci_lower'], sg['ci_upper'],
                     colors='black', linewidth=2)

            # Name and stats
            ax.text(-0.1, y_current, sg['name'], ha='right', va='center',
                   transform=ax.get_yaxis_transform())

            effect_text = f"{sg['effect']:.2f} [{sg['ci_lower']:.2f}, {sg['ci_upper']:.2f}]"
            ax.text(1.02, y_current, effect_text, ha='left', va='center',
                   transform=ax.get_yaxis_transform(), fontsize=self.font_size - 1)

            y_current -= 1

        # Overall effect (diamond)
        y_current -= 0.5
        diamond_height = 0.3

        diamond_x = [overall['ci_lower'], overall['effect'],
                    overall['ci_upper'], overall['effect']]
        diamond_y = [y_current, y_current + diamond_height,
                    y_current, y_current - diamond_height]
        ax.fill(diamond_x, diamond_y, 'black')

        ax.text(-0.1, y_current, 'Overall', ha='right', va='center',
               transform=ax.get_yaxis_transform(), fontweight='bold')

        effect_text = f"{overall['effect']:.2f} [{overall['ci_lower']:.2f}, {overall['ci_upper']:.2f}]"
        ax.text(1.02, y_current, effect_text, ha='left', va='center',
               transform=ax.get_yaxis_transform(), fontsize=self.font_size - 1,
               fontweight='bold')

        # Null effect line
        ax.axvline(x=null_value, color='gray', linestyle='--', linewidth=1)

        # Formatting
        ax.set_ylim(y_current - 1, n_subgroups + 2)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])

        if log_scale:
            ax.set_xscale('log')

        # Add separating line
        ax.axhline(y=y_current + 0.25, color='gray', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')

        return fig

    def multi_panel_figure(self,
                          panels: List[Dict],
                          layout: Tuple[int, int] = None,
                          figsize: Tuple[int, int] = None,
                          save_path: str = None) -> plt.Figure:
        """Create multi-panel figure with labels.

        Args:
            panels: List of dicts with 'type' and parameters
            layout: (rows, cols) layout
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_panels = len(panels)

        if layout is None:
            n_cols = min(3, n_panels)
            n_rows = int(np.ceil(n_panels / n_cols))
            layout = (n_rows, n_cols)

        if figsize is None:
            figsize = (5 * layout[1], 4 * layout[0])

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
        axes = np.array(axes).flatten()

        labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        for i, panel in enumerate(panels):
            ax = axes[i]

            # Add panel label
            ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes,
                   fontsize=self.font_size + 4, fontweight='bold', va='top')

            # Execute plot function based on type
            # This would call the appropriate plotting function

        # Hide unused axes
        for i in range(len(panels), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')

        return fig
