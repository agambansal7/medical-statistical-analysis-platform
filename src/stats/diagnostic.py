"""Diagnostic test evaluation module."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field


@dataclass
class DiagnosticResult:
    """Result of diagnostic test evaluation."""
    n_total: int
    n_positive: int
    n_negative: int
    prevalence: float

    # Basic metrics
    sensitivity: float
    specificity: float
    ppv: float  # Positive predictive value
    npv: float  # Negative predictive value

    # Confidence intervals
    sensitivity_ci: Tuple[float, float]
    specificity_ci: Tuple[float, float]
    ppv_ci: Tuple[float, float]
    npv_ci: Tuple[float, float]

    # Advanced metrics
    accuracy: float
    lr_positive: float  # Positive likelihood ratio
    lr_negative: float  # Negative likelihood ratio
    dor: float  # Diagnostic odds ratio
    youden_index: float

    # Counts
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    summary_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ROCResult:
    """Result of ROC analysis."""
    auc: float
    auc_ci_lower: float
    auc_ci_upper: float
    optimal_threshold: float
    sensitivity_at_optimal: float
    specificity_at_optimal: float
    thresholds: List[float]
    sensitivities: List[float]
    specificities: List[float]
    summary_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiagnosticTests:
    """Diagnostic test evaluation methods."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def evaluate_test(self, y_true: pd.Series,
                      y_pred: pd.Series) -> DiagnosticResult:
        """Evaluate diagnostic test performance.

        Args:
            y_true: True condition (gold standard), binary 0/1
            y_pred: Test prediction, binary 0/1

        Returns:
            DiagnosticResult object
        """
        mask = y_true.notna() & y_pred.notna()
        true = y_true[mask].values.astype(int)
        pred = y_pred[mask].values.astype(int)

        n = len(true)
        if n == 0:
            return self._empty_result()

        # Confusion matrix
        tp = ((true == 1) & (pred == 1)).sum()
        tn = ((true == 0) & (pred == 0)).sum()
        fp = ((true == 0) & (pred == 1)).sum()
        fn = ((true == 1) & (pred == 0)).sum()

        # Basic metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / n
        prevalence = (tp + fn) / n

        # Likelihood ratios
        lr_pos = sensitivity / (1 - specificity) if specificity < 1 else np.inf
        lr_neg = (1 - sensitivity) / specificity if specificity > 0 else np.inf

        # Diagnostic odds ratio
        dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.inf

        # Youden's index
        youden = sensitivity + specificity - 1

        # Wilson confidence intervals
        sens_ci = self._wilson_ci(tp, tp + fn)
        spec_ci = self._wilson_ci(tn, tn + fp)
        ppv_ci = self._wilson_ci(tp, tp + fp)
        npv_ci = self._wilson_ci(tn, tn + fn)

        summary = self._generate_summary(sensitivity, specificity, ppv, npv,
                                         lr_pos, lr_neg, accuracy)

        return DiagnosticResult(
            n_total=n,
            n_positive=int(tp + fn),
            n_negative=int(tn + fp),
            prevalence=float(prevalence),
            sensitivity=float(sensitivity),
            specificity=float(specificity),
            ppv=float(ppv),
            npv=float(npv),
            sensitivity_ci=sens_ci,
            specificity_ci=spec_ci,
            ppv_ci=ppv_ci,
            npv_ci=npv_ci,
            accuracy=float(accuracy),
            lr_positive=float(lr_pos) if not np.isinf(lr_pos) else None,
            lr_negative=float(lr_neg) if not np.isinf(lr_neg) else None,
            dor=float(dor) if not np.isinf(dor) else None,
            youden_index=float(youden),
            true_positive=int(tp),
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            summary_text=summary
        )

    def roc_analysis(self, y_true: pd.Series,
                     y_scores: pd.Series) -> ROCResult:
        """ROC curve analysis.

        Args:
            y_true: True condition (gold standard), binary 0/1
            y_scores: Continuous test scores/probabilities

        Returns:
            ROCResult object
        """
        from sklearn.metrics import roc_curve, roc_auc_score

        mask = y_true.notna() & y_scores.notna()
        true = y_true[mask].values.astype(int)
        scores = y_scores[mask].values

        if len(true) < 10 or len(np.unique(true)) < 2:
            return self._empty_roc_result()

        # ROC curve
        fpr, tpr, thresholds = roc_curve(true, scores)

        # AUC with confidence interval (DeLong method approximation)
        auc = roc_auc_score(true, scores)
        auc_se = self._auc_se(true, scores, auc)
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        auc_ci_lower = max(0, auc - z * auc_se)
        auc_ci_upper = min(1, auc + z * auc_se)

        # Optimal threshold (Youden's index)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Interpretation
        if auc >= 0.9:
            interp = "excellent"
        elif auc >= 0.8:
            interp = "good"
        elif auc >= 0.7:
            interp = "fair"
        else:
            interp = "poor"

        summary = (f"AUC = {auc:.3f} (95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f}), "
                  f"{interp} discrimination. "
                  f"Optimal threshold = {optimal_threshold:.3f} "
                  f"(sensitivity = {tpr[optimal_idx]:.2f}, "
                  f"specificity = {1-fpr[optimal_idx]:.2f}).")

        return ROCResult(
            auc=float(auc),
            auc_ci_lower=float(auc_ci_lower),
            auc_ci_upper=float(auc_ci_upper),
            optimal_threshold=float(optimal_threshold),
            sensitivity_at_optimal=float(tpr[optimal_idx]),
            specificity_at_optimal=float(1 - fpr[optimal_idx]),
            thresholds=thresholds.tolist(),
            sensitivities=tpr.tolist(),
            specificities=(1 - fpr).tolist(),
            summary_text=summary
        )

    def compare_roc_curves(self, y_true: pd.Series,
                           scores1: pd.Series,
                           scores2: pd.Series,
                           names: Tuple[str, str] = ("Test 1", "Test 2")) -> Dict[str, Any]:
        """Compare two ROC curves using DeLong's test.

        Args:
            y_true: True condition
            scores1: Scores from first test
            scores2: Scores from second test
            names: Names of the two tests

        Returns:
            Dictionary with comparison results
        """
        from sklearn.metrics import roc_auc_score

        mask = y_true.notna() & scores1.notna() & scores2.notna()
        true = y_true[mask].values.astype(int)
        s1 = scores1[mask].values
        s2 = scores2[mask].values

        auc1 = roc_auc_score(true, s1)
        auc2 = roc_auc_score(true, s2)

        # Simplified comparison using bootstrap
        n_bootstrap = 1000
        auc_diffs = []
        n = len(true)

        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            if len(np.unique(true[idx])) < 2:
                continue
            try:
                auc1_boot = roc_auc_score(true[idx], s1[idx])
                auc2_boot = roc_auc_score(true[idx], s2[idx])
                auc_diffs.append(auc1_boot - auc2_boot)
            except:
                pass

        if len(auc_diffs) < 100:
            return {'error': 'Insufficient valid bootstrap samples'}

        auc_diff = auc1 - auc2
        se = np.std(auc_diffs)
        z = auc_diff / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        if p_value < 0.05:
            conclusion = (f"Significant difference between {names[0]} (AUC={auc1:.3f}) "
                         f"and {names[1]} (AUC={auc2:.3f}), p = {p_value:.4f}.")
        else:
            conclusion = (f"No significant difference between {names[0]} (AUC={auc1:.3f}) "
                         f"and {names[1]} (AUC={auc2:.3f}), p = {p_value:.4f}.")

        return {
            'test_name': "Bootstrap AUC comparison",
            f'auc_{names[0]}': float(auc1),
            f'auc_{names[1]}': float(auc2),
            'auc_difference': float(auc_diff),
            'z_statistic': float(z),
            'p_value': float(p_value),
            'conclusion': conclusion
        }

    def find_optimal_cutoff(self, y_true: pd.Series,
                           y_scores: pd.Series,
                           method: str = 'youden') -> Dict[str, Any]:
        """Find optimal cutoff for diagnostic test.

        Args:
            y_true: True condition
            y_scores: Continuous test scores
            method: 'youden', 'closest_topleft', 'sensitivity', 'specificity'

        Returns:
            Dictionary with optimal cutoff and performance
        """
        from sklearn.metrics import roc_curve

        mask = y_true.notna() & y_scores.notna()
        true = y_true[mask].values.astype(int)
        scores = y_scores[mask].values

        fpr, tpr, thresholds = roc_curve(true, scores)

        if method == 'youden':
            # Maximize sensitivity + specificity - 1
            optimal_idx = np.argmax(tpr - fpr)
        elif method == 'closest_topleft':
            # Minimize distance to top-left corner
            distances = np.sqrt(fpr**2 + (1-tpr)**2)
            optimal_idx = np.argmin(distances)
        elif method == 'sensitivity':
            # First threshold with sensitivity >= 0.9
            idx = np.where(tpr >= 0.9)[0]
            optimal_idx = idx[-1] if len(idx) > 0 else 0
        elif method == 'specificity':
            # First threshold with specificity >= 0.9
            idx = np.where((1-fpr) >= 0.9)[0]
            optimal_idx = idx[0] if len(idx) > 0 else len(fpr) - 1
        else:
            optimal_idx = np.argmax(tpr - fpr)

        optimal_threshold = thresholds[optimal_idx]
        sens = tpr[optimal_idx]
        spec = 1 - fpr[optimal_idx]

        # Calculate other metrics at optimal threshold
        pred = (scores >= optimal_threshold).astype(int)
        tp = ((true == 1) & (pred == 1)).sum()
        tn = ((true == 0) & (pred == 0)).sum()
        fp = ((true == 0) & (pred == 1)).sum()
        fn = ((true == 1) & (pred == 0)).sum()

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        return {
            'method': method,
            'optimal_threshold': float(optimal_threshold),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'ppv': float(ppv),
            'npv': float(npv),
            'youden_index': float(sens + spec - 1),
            'accuracy': float((tp + tn) / len(true))
        }

    def calibration_analysis(self, y_true: pd.Series,
                             y_prob: pd.Series,
                             n_bins: int = 10) -> Dict[str, Any]:
        """Assess calibration of predicted probabilities.

        Args:
            y_true: True outcomes
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            Dictionary with calibration results
        """
        mask = y_true.notna() & y_prob.notna()
        true = y_true[mask].values
        prob = y_prob[mask].values

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        calibration_data = []
        for i in range(n_bins):
            mask_bin = bin_indices == i
            if mask_bin.sum() > 0:
                mean_predicted = prob[mask_bin].mean()
                mean_observed = true[mask_bin].mean()
                calibration_data.append({
                    'bin': i + 1,
                    'predicted_probability': float(mean_predicted),
                    'observed_frequency': float(mean_observed),
                    'n': int(mask_bin.sum())
                })

        # Hosmer-Lemeshow statistic
        hl_stat = 0
        for item in calibration_data:
            n = item['n']
            obs = item['observed_frequency'] * n
            exp = item['predicted_probability'] * n
            if exp > 0 and (n - exp) > 0:
                hl_stat += (obs - exp)**2 / exp
                hl_stat += ((n - obs) - (n - exp))**2 / (n - exp)

        df = len(calibration_data) - 2
        p_value = 1 - stats.chi2.cdf(hl_stat, df) if df > 0 else np.nan

        # Brier score
        brier = np.mean((prob - true)**2)

        return {
            'calibration_data': calibration_data,
            'hosmer_lemeshow_statistic': float(hl_stat),
            'hosmer_lemeshow_df': df,
            'hosmer_lemeshow_p': float(p_value) if not np.isnan(p_value) else None,
            'brier_score': float(brier),
            'well_calibrated': p_value > 0.05 if not np.isnan(p_value) else None,
            'interpretation': ("Model is well calibrated" if p_value > 0.05
                             else "Model shows poor calibration")
        }

    def _wilson_ci(self, x: int, n: int) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for proportion."""
        if n == 0:
            return (0.0, 0.0)

        p = x / n
        z = stats.norm.ppf((1 + self.confidence_level) / 2)

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def _auc_se(self, y_true: np.ndarray, y_scores: np.ndarray,
                auc: float) -> float:
        """Estimate standard error of AUC."""
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()

        q1 = auc / (2 - auc)
        q2 = (2 * auc**2) / (1 + auc)

        se = np.sqrt((auc * (1 - auc) +
                     (n_pos - 1) * (q1 - auc**2) +
                     (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))
        return se

    def _generate_summary(self, sens: float, spec: float,
                         ppv: float, npv: float,
                         lr_pos: float, lr_neg: float,
                         accuracy: float) -> str:
        """Generate interpretive summary."""
        summary = (f"Sensitivity: {sens*100:.1f}%, Specificity: {spec*100:.1f}%. ")

        if sens >= 0.9:
            summary += "Excellent for ruling out disease (high sensitivity). "
        if spec >= 0.9:
            summary += "Excellent for confirming disease (high specificity). "

        if lr_pos > 10:
            summary += f"LR+ = {lr_pos:.1f} (strong positive evidence). "
        elif lr_pos > 5:
            summary += f"LR+ = {lr_pos:.1f} (moderate positive evidence). "

        if lr_neg < 0.1:
            summary += f"LR- = {lr_neg:.2f} (strong negative evidence). "
        elif lr_neg < 0.2:
            summary += f"LR- = {lr_neg:.2f} (moderate negative evidence). "

        return summary

    def _empty_result(self) -> DiagnosticResult:
        """Create empty result for insufficient data."""
        return DiagnosticResult(
            n_total=0, n_positive=0, n_negative=0, prevalence=0,
            sensitivity=0, specificity=0, ppv=0, npv=0,
            sensitivity_ci=(0, 0), specificity_ci=(0, 0),
            ppv_ci=(0, 0), npv_ci=(0, 0),
            accuracy=0, lr_positive=0, lr_negative=0, dor=0, youden_index=0,
            true_positive=0, true_negative=0, false_positive=0, false_negative=0,
            summary_text="Insufficient data"
        )

    def _empty_roc_result(self) -> ROCResult:
        """Create empty ROC result."""
        return ROCResult(
            auc=0.5, auc_ci_lower=0, auc_ci_upper=1,
            optimal_threshold=0.5, sensitivity_at_optimal=0, specificity_at_optimal=0,
            thresholds=[], sensitivities=[], specificities=[],
            summary_text="Insufficient data for ROC analysis"
        )
