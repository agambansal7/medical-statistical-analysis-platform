"""Statistical visualization module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class StatisticalPlots:
    """Generate statistical visualizations."""

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 300,
                 palette: str = 'colorblind'):
        self.figsize = figsize
        self.dpi = dpi
        self.palette = palette

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')

        sns.set_palette(palette)

    def histogram(self, data: pd.Series,
                  title: str = None,
                  xlabel: str = None,
                  show_normal: bool = True,
                  bins: int = 'auto',
                  save_path: str = None) -> plt.Figure:
        """Create histogram with optional normal curve overlay.

        Args:
            data: Data to plot
            title: Plot title
            xlabel: X-axis label
            show_normal: Overlay normal distribution curve
            bins: Number of bins
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        clean_data = data.dropna()

        # Histogram
        ax.hist(clean_data, bins=bins, density=True, alpha=0.7,
                color=sns.color_palette(self.palette)[0], edgecolor='white')

        # Normal curve
        if show_normal:
            mu, std = clean_data.mean(), clean_data.std()
            x = np.linspace(clean_data.min(), clean_data.max(), 100)
            from scipy import stats
            ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2,
                   label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
            ax.legend()

        ax.set_xlabel(xlabel or str(data.name) or 'Value')
        ax.set_ylabel('Density')
        ax.set_title(title or f'Distribution of {data.name}')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def boxplot(self, data: pd.DataFrame,
                value_col: str,
                group_col: str = None,
                title: str = None,
                xlabel: str = None,
                ylabel: str = None,
                show_points: bool = True,
                save_path: str = None) -> plt.Figure:
        """Create boxplot, optionally grouped.

        Args:
            data: DataFrame with the data
            value_col: Column for values
            group_col: Column for grouping (optional)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_points: Show individual data points
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if group_col:
            if show_points:
                sns.boxplot(data=data, x=group_col, y=value_col, ax=ax,
                           palette=self.palette)
                sns.stripplot(data=data, x=group_col, y=value_col, ax=ax,
                             color='black', alpha=0.5, size=4)
            else:
                sns.boxplot(data=data, x=group_col, y=value_col, ax=ax,
                           palette=self.palette)
        else:
            ax.boxplot(data[value_col].dropna())
            if show_points:
                ax.scatter(np.ones(len(data[value_col].dropna())),
                          data[value_col].dropna(), alpha=0.5, color='black')

        ax.set_xlabel(xlabel or group_col or '')
        ax.set_ylabel(ylabel or value_col)
        ax.set_title(title or f'Boxplot of {value_col}')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def violin_plot(self, data: pd.DataFrame,
                   value_col: str,
                   group_col: str,
                   title: str = None,
                   split: bool = False,
                   hue: str = None,
                   save_path: str = None) -> plt.Figure:
        """Create violin plot.

        Args:
            data: DataFrame with the data
            value_col: Column for values
            group_col: Column for grouping
            title: Plot title
            split: Split violins for hue variable
            hue: Additional grouping variable
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.violinplot(data=data, x=group_col, y=value_col,
                      hue=hue, split=split, ax=ax, palette=self.palette)

        ax.set_title(title or f'{value_col} by {group_col}')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def scatter_plot(self, data: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    hue: str = None,
                    title: str = None,
                    show_regression: bool = True,
                    show_correlation: bool = True,
                    save_path: str = None) -> plt.Figure:
        """Create scatter plot with optional regression line.

        Args:
            data: DataFrame with the data
            x_col: X-axis column
            y_col: Y-axis column
            hue: Color by group
            title: Plot title
            show_regression: Show regression line
            show_correlation: Show correlation coefficient
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if hue:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue,
                           ax=ax, palette=self.palette, alpha=0.7)
        else:
            ax.scatter(data[x_col], data[y_col], alpha=0.7,
                      color=sns.color_palette(self.palette)[0])

        if show_regression and not hue:
            from scipy import stats
            mask = data[x_col].notna() & data[y_col].notna()
            x = data.loc[mask, x_col]
            y = data.loc[mask, y_col]
            slope, intercept, r, p, se = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(line_x, slope * line_x + intercept, 'r-', lw=2,
                   label=f'y = {slope:.3f}x + {intercept:.3f}')

            if show_correlation:
                ax.annotate(f'r = {r:.3f}, p = {p:.4f}',
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=12, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.legend()

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title or f'{y_col} vs {x_col}')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def correlation_heatmap(self, data: pd.DataFrame,
                           variables: List[str] = None,
                           method: str = 'pearson',
                           title: str = None,
                           annot: bool = True,
                           save_path: str = None) -> plt.Figure:
        """Create correlation heatmap.

        Args:
            data: DataFrame with the data
            variables: List of variables (uses all numeric if None)
            method: Correlation method
            title: Plot title
            annot: Show correlation values
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if variables:
            corr_data = data[variables]
        else:
            corr_data = data.select_dtypes(include=[np.number])

        corr_matrix = corr_data.corr(method=method)

        # Adjust figure size based on number of variables
        n_vars = len(corr_matrix)
        fig_size = (max(8, n_vars * 0.8), max(6, n_vars * 0.6))

        fig, ax = plt.subplots(figsize=fig_size)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Correlation'})

        ax.set_title(title or f'{method.capitalize()} Correlation Matrix')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def bar_chart(self, data: pd.DataFrame,
                 x_col: str,
                 y_col: str = None,
                 hue: str = None,
                 title: str = None,
                 error_bars: bool = True,
                 horizontal: bool = False,
                 save_path: str = None) -> plt.Figure:
        """Create bar chart.

        Args:
            data: DataFrame with the data
            x_col: Category column
            y_col: Value column (if None, uses counts)
            hue: Color by group
            title: Plot title
            error_bars: Show confidence intervals
            horizontal: Horizontal bars
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if y_col:
            if horizontal:
                sns.barplot(data=data, y=x_col, x=y_col, hue=hue,
                           ax=ax, palette=self.palette, ci=95 if error_bars else None)
            else:
                sns.barplot(data=data, x=x_col, y=y_col, hue=hue,
                           ax=ax, palette=self.palette, ci=95 if error_bars else None)
        else:
            counts = data[x_col].value_counts()
            if horizontal:
                ax.barh(counts.index, counts.values,
                       color=sns.color_palette(self.palette)[0])
            else:
                ax.bar(counts.index, counts.values,
                      color=sns.color_palette(self.palette)[0])

        ax.set_title(title or f'{y_col or "Count"} by {x_col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def qq_plot(self, data: pd.Series,
               title: str = None,
               save_path: str = None) -> plt.Figure:
        """Create Q-Q plot for normality assessment.

        Args:
            data: Data to plot
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        from scipy import stats

        fig, ax = plt.subplots(figsize=self.figsize)

        clean_data = data.dropna()
        stats.probplot(clean_data, dist="norm", plot=ax)

        ax.set_title(title or f'Q-Q Plot: {data.name}')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def paired_plot(self, before: pd.Series,
                   after: pd.Series,
                   title: str = None,
                   labels: Tuple[str, str] = ('Before', 'After'),
                   save_path: str = None) -> plt.Figure:
        """Create paired data visualization (spaghetti plot).

        Args:
            before: Before values
            after: After values
            title: Plot title
            labels: Labels for before/after
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        mask = before.notna() & after.notna()
        b = before[mask].values
        a = after[mask].values

        # Individual lines
        for i in range(len(b)):
            ax.plot([0, 1], [b[i], a[i]], 'o-', color='gray', alpha=0.3, lw=1)

        # Means
        ax.plot([0, 1], [b.mean(), a.mean()], 'o-', color='red',
               lw=3, markersize=10, label='Mean')

        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_ylabel('Value')
        ax.set_title(title or 'Paired Comparison')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def forest_plot(self, effects: List[Dict[str, Any]],
                   title: str = None,
                   xlabel: str = 'Effect Size',
                   null_value: float = 0,
                   save_path: str = None) -> plt.Figure:
        """Create forest plot for meta-analysis or subgroup effects.

        Args:
            effects: List of dicts with 'name', 'effect', 'ci_lower', 'ci_upper'
            title: Plot title
            xlabel: X-axis label
            null_value: Value for null effect line
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n = len(effects)
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))

        y_positions = range(n)

        for i, effect in enumerate(effects):
            # Point estimate
            ax.plot(effect['effect'], i, 'ko', markersize=8)

            # Confidence interval
            ax.hlines(i, effect['ci_lower'], effect['ci_upper'],
                     color='black', linewidth=2)

        # Null effect line
        ax.axvline(x=null_value, color='red', linestyle='--', alpha=0.7)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([e['name'] for e in effects])
        ax.set_xlabel(xlabel)
        ax.set_title(title or 'Forest Plot')

        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def kaplan_meier_plot(self, km_results: Dict,
                         title: str = None,
                         xlabel: str = 'Time',
                         ylabel: str = 'Survival Probability',
                         show_ci: bool = True,
                         save_path: str = None) -> plt.Figure:
        """Create Kaplan-Meier survival curve.

        Args:
            km_results: Dictionary with KM fitter objects by group
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_ci: Show confidence intervals
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = sns.color_palette(self.palette)

        for i, (group_name, kmf) in enumerate(km_results.items()):
            kmf.plot_survival_function(ax=ax, ci_show=show_ci,
                                       color=colors[i % len(colors)],
                                       label=group_name)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or 'Kaplan-Meier Survival Curves')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def survival_curve(self, data: pd.DataFrame,
                       time_col: str,
                       event_col: str,
                       group_col: str = None,
                       title: str = None,
                       xlabel: str = 'Time',
                       ylabel: str = 'Survival Probability',
                       show_ci: bool = True,
                       at_risk: bool = True,
                       show_censors: bool = True,
                       save_path: str = None) -> plt.Figure:
        """Create publication-quality Kaplan-Meier survival curve.

        Args:
            data: DataFrame with survival data
            time_col: Column name for time-to-event
            event_col: Column name for event indicator (1=event, 0=censored)
            group_col: Column name for grouping (optional)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_ci: Show confidence intervals
            at_risk: Show at-risk table below plot
            show_censors: Show censoring marks on curves
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        from lifelines import KaplanMeierFitter

        # Filter out rows with missing data
        cols = [time_col, event_col]
        if group_col:
            cols.append(group_col)
        clean_data = data[cols].dropna()

        # Publication-quality color palette
        km_colors = [
            '#2E86AB',  # Steel Blue
            '#E94F37',  # Vermillion Red
            '#1B998B',  # Teal
            '#F18F01',  # Orange
            '#8338EC',  # Purple
            '#FF006E',  # Pink
        ]

        # Adjust figure size based on whether we show at-risk table
        if at_risk and group_col:
            fig = plt.figure(figsize=(12, 9))
            gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
            ax = fig.add_subplot(gs[0])
            ax_risk = fig.add_subplot(gs[1])
        else:
            fig, ax = plt.subplots(figsize=(11, 8))
            ax_risk = None

        # Set white background for publication quality
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        at_risk_data = {}
        kmf_objects = {}

        if group_col and group_col in clean_data.columns:
            groups = sorted(clean_data[group_col].unique())
            n_groups = len(groups)

            for i, group in enumerate(groups):
                group_data = clean_data[clean_data[group_col] == group]
                kmf = KaplanMeierFitter()
                kmf.fit(group_data[time_col], group_data[event_col], label=str(group))
                kmf_objects[str(group)] = kmf

                color = km_colors[i % len(km_colors)]

                # Plot step function with enhanced styling
                times = kmf.survival_function_.index
                survival = kmf.survival_function_.values.flatten()

                # Main survival curve - thicker line
                ax.step(times, survival, where='post', color=color,
                       linewidth=2.5, label=f'{group} (n={len(group_data)})')

                # Confidence interval with gradient effect
                if show_ci:
                    ci = kmf.confidence_interval_survival_function_
                    ci_lower = ci.iloc[:, 0].values
                    ci_upper = ci.iloc[:, 1].values
                    ax.fill_between(times, ci_lower, ci_upper,
                                   step='post', alpha=0.15, color=color)

                # Censoring marks - small vertical ticks
                if show_censors:
                    censored_times = group_data[group_data[event_col] == 0][time_col]
                    for censor_time in censored_times:
                        try:
                            surv_at_censor = kmf.predict(censor_time)
                            ax.plot(censor_time, surv_at_censor, '|', color=color,
                                   markersize=8, markeredgewidth=1.5)
                        except:
                            pass

                # Store at-risk numbers
                if at_risk:
                    at_risk_data[str(group)] = {
                        'times': times,
                        'kmf': kmf,
                        'n_total': len(group_data),
                        'color': color
                    }
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(clean_data[time_col], clean_data[event_col], label='Overall')
            kmf_objects['Overall'] = kmf

            color = km_colors[0]
            times = kmf.survival_function_.index
            survival = kmf.survival_function_.values.flatten()

            ax.step(times, survival, where='post', color=color,
                   linewidth=2.5, label=f'Overall (n={len(clean_data)})')

            if show_ci:
                ci = kmf.confidence_interval_survival_function_
                ci_lower = ci.iloc[:, 0].values
                ci_upper = ci.iloc[:, 1].values
                ax.fill_between(times, ci_lower, ci_upper,
                               step='post', alpha=0.15, color=color)

            if show_censors:
                censored_times = clean_data[clean_data[event_col] == 0][time_col]
                for censor_time in censored_times:
                    try:
                        surv_at_censor = kmf.predict(censor_time)
                        ax.plot(censor_time, surv_at_censor, '|', color=color,
                               markersize=8, markeredgewidth=1.5)
                    except:
                        pass

        # Enhanced axis styling
        ax.set_xlabel(xlabel, fontsize=13, fontweight='medium', labelpad=10)
        ax.set_ylabel(ylabel, fontsize=13, fontweight='medium', labelpad=10)
        ax.set_title(title or 'Kaplan-Meier Survival Curves',
                    fontsize=16, fontweight='bold', pad=15)

        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(left=0)

        # Add gridlines for readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Style the spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
            ax.spines[spine].set_color('#333333')

        # Enhanced legend
        legend = ax.legend(loc='lower left', frameon=True, fancybox=True,
                          shadow=False, fontsize=11, framealpha=0.95,
                          edgecolor='#cccccc')
        legend.get_frame().set_linewidth(1)

        # Y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.tick_params(axis='both', labelsize=11)

        # At-risk table with enhanced styling
        if at_risk and ax_risk is not None and at_risk_data:
            ax_risk.set_facecolor('white')
            ax_risk.axis('off')

            max_time = ax.get_xlim()[1]
            n_timepoints = 6
            time_points = np.linspace(0, max_time * 0.95, n_timepoints).astype(int)

            row_labels = list(at_risk_data.keys())
            cell_text = []
            cell_colors = []

            for group_name in row_labels:
                row = []
                row_color = []
                kmf = at_risk_data[group_name]['kmf']
                group_color = at_risk_data[group_name]['color']

                for t in time_points:
                    try:
                        # Get at-risk count at time t
                        at_risk_count = int((kmf.event_table.index <= t).sum())
                        if at_risk_count > 0:
                            at_risk_count = int(kmf.event_table.loc[kmf.event_table.index <= t, 'at_risk'].iloc[-1])
                        else:
                            at_risk_count = at_risk_data[group_name]['n_total']
                        row.append(str(at_risk_count))
                    except:
                        row.append('-')
                    row_color.append('white')
                cell_text.append(row)
                cell_colors.append(row_color)

            # Create styled table
            table = ax_risk.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=[f'{int(t)}' for t in time_points],
                loc='upper center',
                cellLoc='center',
                cellColours=cell_colors
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)

            # Style the table
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('#e0e0e0')
                cell.set_linewidth(0.5)
                if row == 0:  # Header row
                    cell.set_text_props(fontweight='bold', color='#333333')
                    cell.set_facecolor('#f5f5f5')
                elif col == -1:  # Row labels
                    idx = row - 1
                    if idx < len(row_labels):
                        cell.set_text_props(fontweight='bold',
                                           color=at_risk_data[row_labels[idx]]['color'])
                    cell.set_facecolor('#fafafa')

            # Table title
            ax_risk.text(0.0, 0.95, 'Number at Risk', fontsize=11,
                        fontweight='bold', transform=ax_risk.transAxes,
                        color='#333333')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

        return fig

    def roc_curve_plot(self, roc_results: Union[Dict, List[Dict]],
                      title: str = None,
                      save_path: str = None) -> plt.Figure:
        """Create ROC curve plot.

        Args:
            roc_results: ROC result dict or list of dicts for comparison
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        if isinstance(roc_results, dict):
            roc_results = [roc_results]

        colors = sns.color_palette(self.palette)

        for i, roc in enumerate(roc_results):
            fpr = [1 - s for s in roc['specificities']]
            tpr = roc['sensitivities']
            auc = roc['auc']
            label = roc.get('name', f'Model {i+1}')

            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                   label=f'{label} (AUC = {auc:.3f})')

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('1 - Specificity (False Positive Rate)')
        ax.set_ylabel('Sensitivity (True Positive Rate)')
        ax.set_title(title or 'ROC Curve')
        ax.legend(loc='lower right')
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def bland_altman_plot(self, ba_results: Dict,
                         title: str = None,
                         save_path: str = None) -> plt.Figure:
        """Create Bland-Altman plot.

        Args:
            ba_results: Results from Bland-Altman analysis
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        means = ba_results['means']
        diffs = ba_results['differences']
        bias = ba_results['bias']
        loa_lower = ba_results['loa_lower']
        loa_upper = ba_results['loa_upper']

        # Scatter plot
        ax.scatter(means, diffs, alpha=0.6, color=sns.color_palette(self.palette)[0])

        # Bias line
        ax.axhline(y=bias, color='red', linestyle='-', lw=2, label=f'Bias: {bias:.3f}')

        # Limits of agreement
        ax.axhline(y=loa_upper, color='gray', linestyle='--', lw=1.5,
                  label=f'+1.96 SD: {loa_upper:.3f}')
        ax.axhline(y=loa_lower, color='gray', linestyle='--', lw=1.5,
                  label=f'-1.96 SD: {loa_lower:.3f}')

        ax.set_xlabel('Mean of Two Methods')
        ax.set_ylabel('Difference (Method 1 - Method 2)')
        ax.set_title(title or 'Bland-Altman Plot')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def residual_plots(self, model_results,
                      save_path: str = None) -> plt.Figure:
        """Create diagnostic residual plots for regression.

        Args:
            model_results: Fitted model with residuals
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        residuals = model_results.resid
        fitted = model_results.fittedvalues

        # 1. Residuals vs Fitted
        axes[0, 0].scatter(fitted, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')

        # 2. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q')

        # 3. Scale-Location
        std_resid = np.sqrt(np.abs(residuals / residuals.std()))
        axes[1, 0].scatter(fitted, std_resid, alpha=0.6)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location')

        # 4. Histogram of residuals
        axes[1, 1].hist(residuals, bins=30, edgecolor='white', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Histogram of Residuals')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def save_figure(self, fig: plt.Figure, path: str,
                   formats: List[str] = ['png', 'pdf']) -> List[str]:
        """Save figure in multiple formats.

        Args:
            fig: Matplotlib figure
            path: Base path (without extension)
            formats: List of formats to save

        Returns:
            List of saved file paths
        """
        saved_paths = []
        path = Path(path)

        for fmt in formats:
            save_path = path.with_suffix(f'.{fmt}')
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=fmt)
            saved_paths.append(str(save_path))

        return saved_paths

    def close_all(self):
        """Close all figures."""
        plt.close('all')
