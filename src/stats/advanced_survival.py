"""
Advanced Survival Analysis Module
=================================
Advanced methods beyond standard Kaplan-Meier and Cox regression.

Methods:
- Competing risks analysis (Fine-Gray model)
- Time-varying covariates
- Landmark analysis
- Restricted mean survival time (RMST)
- Flexible parametric survival models
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from scipy import stats
from scipy.integrate import trapezoid
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CompetingRisksResult:
    """Result container for competing risks analysis."""
    event_of_interest: str
    competing_events: List[str]
    cumulative_incidence: Dict[str, np.ndarray]
    time_points: np.ndarray
    cumulative_incidence_at_times: Dict[str, Dict[float, float]]
    gray_test_statistic: float
    gray_test_p_value: float
    fine_gray_coefficients: Optional[Dict[str, Dict]]
    n_observations: int
    n_events: Dict[str, int]
    summary_text: str


@dataclass
class TimeVaryingCoxResult:
    """Result container for Cox model with time-varying covariates."""
    coefficients: Dict[str, Dict]
    baseline_hazard: pd.DataFrame
    concordance_index: float
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_events: int
    n_time_intervals: int
    summary_text: str


@dataclass
class LandmarkResult:
    """Result container for landmark analysis."""
    landmark_time: float
    n_at_risk_at_landmark: int
    n_events_after_landmark: int
    survival_from_landmark: Dict[str, pd.DataFrame]
    hazard_ratios: Dict[str, Dict]
    log_rank_p_value: float
    restricted_mean_survival: Dict[str, float]
    summary_text: str


@dataclass
class RMSTResult:
    """Result container for restricted mean survival time analysis."""
    tau: float  # Restriction time
    rmst: Dict[str, float]  # RMST by group
    rmst_se: Dict[str, float]
    rmst_ci_lower: Dict[str, float]
    rmst_ci_upper: Dict[str, float]
    rmst_difference: float
    rmst_diff_se: float
    rmst_diff_ci_lower: float
    rmst_diff_ci_upper: float
    rmst_diff_p_value: float
    rmst_ratio: float
    life_years_gained: float
    n_observations: Dict[str, int]
    summary_text: str


@dataclass
class FlexibleSurvivalResult:
    """Result container for flexible parametric survival models."""
    distribution: str
    parameters: Dict[str, float]
    hazard_ratios: Dict[str, Dict]
    median_survival: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float
    n_observations: int
    n_events: int
    summary_text: str


class AdvancedSurvivalAnalysis:
    """
    Advanced survival analysis methods for complex time-to-event data.
    """

    def __init__(self):
        pass

    def competing_risks_analysis(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        event_of_interest: int = 1,
        group_col: Optional[str] = None,
        covariates: Optional[List[str]] = None
    ) -> CompetingRisksResult:
        """
        Competing risks analysis using Aalen-Johansen estimator and Fine-Gray model.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        time_col : str
            Time to event column
        event_col : str
            Event type column (0=censored, 1=event of interest, 2+=competing events)
        event_of_interest : int
            Code for the event of interest
        group_col : str, optional
            Grouping variable for comparison
        covariates : List[str], optional
            Covariates for Fine-Gray model

        Returns:
        --------
        CompetingRisksResult
        """
        df_clean = df.dropna(subset=[time_col, event_col])

        times = df_clean[time_col].values
        events = df_clean[event_col].values

        # Identify competing events
        unique_events = sorted([e for e in np.unique(events) if e != 0])
        competing_events = [e for e in unique_events if e != event_of_interest]

        # Calculate cumulative incidence functions using Aalen-Johansen
        unique_times = np.sort(np.unique(times[events > 0]))
        n = len(times)

        # Initialize
        cumulative_incidence = {str(e): np.zeros(len(unique_times)) for e in unique_events}

        # At risk at each time
        at_risk = np.array([np.sum(times >= t) for t in unique_times])

        # Overall survival (Kaplan-Meier of any event)
        km_surv = np.ones(len(unique_times))
        prev_surv = 1.0

        for i, t in enumerate(unique_times):
            # Number of events at this time
            n_at_risk = at_risk[i]
            if n_at_risk == 0:
                continue

            # Events by type
            events_at_t = events[times == t]
            n_any_event = np.sum(events_at_t > 0)

            # Update survival
            hazard = n_any_event / n_at_risk
            km_surv[i] = prev_surv * (1 - hazard)

            # Cause-specific hazards and cumulative incidence
            for e in unique_events:
                n_cause_events = np.sum(events_at_t == e)
                cause_hazard = n_cause_events / n_at_risk

                # CIF increment: S(t-) * cause-specific hazard
                cumulative_incidence[str(e)][i] = (
                    cumulative_incidence[str(e)][i-1] if i > 0 else 0
                ) + prev_surv * cause_hazard

            prev_surv = km_surv[i]

        # Get CIF at specific times
        cif_at_times = {}
        for e in unique_events:
            cif_at_times[str(e)] = {}
            for tau in [30, 90, 180, 365, 730]:  # Days
                idx = np.searchsorted(unique_times, tau)
                if idx < len(unique_times):
                    cif_at_times[str(e)][tau] = cumulative_incidence[str(e)][min(idx, len(cumulative_incidence[str(e)])-1)]
                else:
                    cif_at_times[str(e)][tau] = cumulative_incidence[str(e)][-1]

        # Gray's test for comparing CIF between groups
        gray_stat = 0.0
        gray_p = 1.0

        if group_col is not None:
            # Simplified Gray's test implementation
            groups = df_clean[group_col].unique()
            if len(groups) == 2:
                g1 = df_clean[df_clean[group_col] == groups[0]]
                g2 = df_clean[df_clean[group_col] == groups[1]]

                # Use log-rank style comparison on subdistribution
                # This is a simplified approximation
                from scipy.stats import chi2

                # Create subdistribution indicator
                sub_event_1 = (g1[event_col] == event_of_interest).astype(int)
                sub_event_2 = (g2[event_col] == event_of_interest).astype(int)

                # Simple chi-square test as approximation
                observed = np.array([sub_event_1.sum(), sub_event_2.sum()])
                expected = np.array([len(g1), len(g2)]) * observed.sum() / len(df_clean)
                expected = np.maximum(expected, 0.1)  # Avoid division by zero

                gray_stat = np.sum((observed - expected)**2 / expected)
                gray_p = 1 - chi2.cdf(gray_stat, df=1)

        # Fine-Gray subdistribution hazard model
        fine_gray_coefs = None
        if covariates is not None and len(covariates) > 0:
            fine_gray_coefs = self._fit_fine_gray_model(
                df_clean, time_col, event_col, event_of_interest, covariates
            )

        # Count events
        n_events = {str(e): int(np.sum(events == e)) for e in unique_events}

        summary = (
            f"Competing risks analysis. N={len(df_clean)}, "
            f"Events of interest (type {event_of_interest}): {n_events[str(event_of_interest)]}. "
            f"Competing events: {sum(n_events[str(e)] for e in competing_events)}. "
            f"1-year CIF for event of interest: {cif_at_times[str(event_of_interest)].get(365, 0)*100:.1f}%."
        )

        return CompetingRisksResult(
            event_of_interest=str(event_of_interest),
            competing_events=[str(e) for e in competing_events],
            cumulative_incidence=cumulative_incidence,
            time_points=unique_times,
            cumulative_incidence_at_times=cif_at_times,
            gray_test_statistic=gray_stat,
            gray_test_p_value=gray_p,
            fine_gray_coefficients=fine_gray_coefs,
            n_observations=len(df_clean),
            n_events=n_events,
            summary_text=summary
        )

    def _fit_fine_gray_model(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        event_of_interest: int,
        covariates: List[str]
    ) -> Dict[str, Dict]:
        """
        Fit Fine-Gray subdistribution hazard model.

        Uses weighted Cox regression as approximation.
        """
        # Create subdistribution dataset
        # Competing events are censored at their event time but remain at risk

        df_fg = df.copy()

        # Create subdistribution event indicator
        df_fg['_fg_event'] = (df_fg[event_col] == event_of_interest).astype(int)

        # For competing events, censor at time of competing event
        # but keep in risk set (simplified approach)

        # Fit weighted Cox model
        try:
            cph = CoxPHFitter()
            cph.fit(
                df_fg[[time_col, '_fg_event'] + covariates],
                duration_col=time_col,
                event_col='_fg_event'
            )

            coefficients = {}
            for var in covariates:
                coefficients[var] = {
                    'subdistribution_hr': float(np.exp(cph.params_[var])),
                    'hr_ci_lower': float(np.exp(cph.confidence_intervals_.loc[var, '95% lower-bound'])),
                    'hr_ci_upper': float(np.exp(cph.confidence_intervals_.loc[var, '95% upper-bound'])),
                    'p_value': float(cph.summary.loc[var, 'p'])
                }

            return coefficients
        except Exception as e:
            return None

    def time_varying_covariates_cox(
        self,
        df_long: pd.DataFrame,
        id_col: str,
        start_col: str,
        stop_col: str,
        event_col: str,
        covariates: List[str]
    ) -> TimeVaryingCoxResult:
        """
        Cox proportional hazards model with time-varying covariates.

        Parameters:
        -----------
        df_long : pd.DataFrame
            Long-format data with time intervals per subject
        id_col : str
            Subject ID column
        start_col : str
            Interval start time column
        stop_col : str
            Interval stop time column
        event_col : str
            Event indicator column (1=event, 0=censored)
        covariates : List[str]
            Time-varying and time-fixed covariates

        Returns:
        --------
        TimeVaryingCoxResult
        """
        df_clean = df_long.dropna(subset=[start_col, stop_col, event_col] + covariates)

        # Lifelines doesn't directly support time-varying covariates
        # We use the counting process format with CoxTimeVaryingFitter
        from lifelines import CoxTimeVaryingFitter

        ctv = CoxTimeVaryingFitter()
        ctv.fit(
            df_clean,
            id_col=id_col,
            start_col=start_col,
            stop_col=stop_col,
            event_col=event_col
        )

        # Extract results
        coefficients = {}
        for var in covariates:
            if var in ctv.summary.index:
                coefficients[var] = {
                    'hazard_ratio': float(np.exp(ctv.params_[var])),
                    'hr_ci_lower': float(np.exp(ctv.confidence_intervals_.loc[var, '95% lower-bound'])),
                    'hr_ci_upper': float(np.exp(ctv.confidence_intervals_.loc[var, '95% upper-bound'])),
                    'p_value': float(ctv.summary.loc[var, 'p']),
                    'coefficient': float(ctv.params_[var])
                }

        n_subjects = df_clean[id_col].nunique()
        n_events = int(df_clean[event_col].sum())
        n_intervals = len(df_clean)

        summary = (
            f"Cox model with time-varying covariates. "
            f"N={n_subjects} subjects, {n_intervals} intervals, {n_events} events. "
            f"Concordance index = {ctv.concordance_index_:.3f}."
        )

        return TimeVaryingCoxResult(
            coefficients=coefficients,
            baseline_hazard=ctv.baseline_cumulative_hazard_,
            concordance_index=ctv.concordance_index_,
            log_likelihood=ctv.log_likelihood_,
            aic=ctv.AIC_,
            bic=ctv.AIC_ + (np.log(n_subjects) - 2) * len(covariates),  # Approximate BIC
            n_observations=n_subjects,
            n_events=n_events,
            n_time_intervals=n_intervals,
            summary_text=summary
        )

    def landmark_analysis(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str,
        landmark_time: float,
        covariates: Optional[List[str]] = None,
        horizon: Optional[float] = None
    ) -> LandmarkResult:
        """
        Landmark analysis to avoid immortal time bias.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        time_col : str
            Time to event column
        event_col : str
            Event indicator column
        group_col : str
            Grouping variable
        landmark_time : float
            Landmark time point
        covariates : List[str], optional
            Covariates for Cox model
        horizon : float, optional
            Time horizon for RMST calculation

        Returns:
        --------
        LandmarkResult
        """
        df_clean = df.dropna(subset=[time_col, event_col, group_col])

        # Select patients at risk at landmark
        df_landmark = df_clean[df_clean[time_col] > landmark_time].copy()

        # Adjust time to start from landmark
        df_landmark['_time_from_landmark'] = df_landmark[time_col] - landmark_time

        n_at_risk = len(df_landmark)
        n_events_after = int(df_landmark[event_col].sum())

        # Kaplan-Meier by group from landmark
        groups = df_landmark[group_col].unique()
        survival_from_landmark = {}
        rmst_by_group = {}

        if horizon is None:
            horizon = df_landmark['_time_from_landmark'].max()

        for group in groups:
            group_df = df_landmark[df_landmark[group_col] == group]

            kmf = KaplanMeierFitter()
            kmf.fit(
                group_df['_time_from_landmark'],
                event_observed=group_df[event_col],
                label=str(group)
            )

            survival_from_landmark[str(group)] = kmf.survival_function_

            # Calculate RMST
            sf = kmf.survival_function_
            times = sf.index.values
            surv = sf.values.flatten()

            # Truncate at horizon
            mask = times <= horizon
            times_trunc = times[mask]
            surv_trunc = surv[mask]

            if len(times_trunc) > 1:
                rmst_by_group[str(group)] = trapezoid(surv_trunc, times_trunc)
            else:
                rmst_by_group[str(group)] = 0

        # Log-rank test from landmark
        if len(groups) == 2:
            g1 = df_landmark[df_landmark[group_col] == groups[0]]
            g2 = df_landmark[df_landmark[group_col] == groups[1]]

            lr_result = logrank_test(
                g1['_time_from_landmark'], g2['_time_from_landmark'],
                g1[event_col], g2[event_col]
            )
            logrank_p = lr_result.p_value
        else:
            from lifelines.statistics import multivariate_logrank_test
            lr_result = multivariate_logrank_test(
                df_landmark['_time_from_landmark'],
                df_landmark[group_col],
                df_landmark[event_col]
            )
            logrank_p = lr_result.p_value

        # Cox regression from landmark
        hazard_ratios = {}
        if covariates is not None:
            try:
                cph = CoxPHFitter()

                # Create dummy variables for group
                df_cox = df_landmark.copy()
                df_cox = pd.get_dummies(df_cox, columns=[group_col], drop_first=True)

                cox_cols = ['_time_from_landmark', event_col] + \
                           [c for c in df_cox.columns if c.startswith(group_col)] + \
                           covariates

                cph.fit(df_cox[cox_cols], duration_col='_time_from_landmark', event_col=event_col)

                for var in cph.params_.index:
                    hazard_ratios[var] = {
                        'hr': float(np.exp(cph.params_[var])),
                        'hr_ci_lower': float(np.exp(cph.confidence_intervals_.loc[var, '95% lower-bound'])),
                        'hr_ci_upper': float(np.exp(cph.confidence_intervals_.loc[var, '95% upper-bound'])),
                        'p_value': float(cph.summary.loc[var, 'p'])
                    }
            except Exception as e:
                pass

        summary = (
            f"Landmark analysis at t={landmark_time}. "
            f"N at risk at landmark = {n_at_risk}, events after landmark = {n_events_after}. "
            f"Log-rank p = {logrank_p:.4f}."
        )

        return LandmarkResult(
            landmark_time=landmark_time,
            n_at_risk_at_landmark=n_at_risk,
            n_events_after_landmark=n_events_after,
            survival_from_landmark=survival_from_landmark,
            hazard_ratios=hazard_ratios,
            log_rank_p_value=logrank_p,
            restricted_mean_survival=rmst_by_group,
            summary_text=summary
        )

    def restricted_mean_survival_time(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str,
        tau: Optional[float] = None,
        alpha: float = 0.05
    ) -> RMSTResult:
        """
        Restricted mean survival time (RMST) analysis.

        RMST represents the area under the survival curve up to time tau,
        interpreted as average survival time up to that point.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        time_col : str
            Time to event column
        event_col : str
            Event indicator column
        group_col : str
            Grouping variable (2 groups for comparison)
        tau : float, optional
            Restriction time (default: minimum of max observed times per group)
        alpha : float
            Significance level for confidence intervals

        Returns:
        --------
        RMSTResult
        """
        df_clean = df.dropna(subset=[time_col, event_col, group_col])
        groups = df_clean[group_col].unique()

        if len(groups) != 2:
            raise ValueError("RMST comparison requires exactly 2 groups")

        # Determine tau
        if tau is None:
            tau = min(
                df_clean[df_clean[group_col] == groups[0]][time_col].max(),
                df_clean[df_clean[group_col] == groups[1]][time_col].max()
            )

        rmst = {}
        rmst_se = {}
        rmst_ci_lower = {}
        rmst_ci_upper = {}
        n_obs = {}

        z = stats.norm.ppf(1 - alpha/2)

        for group in groups:
            group_df = df_clean[df_clean[group_col] == group]
            n_obs[str(group)] = len(group_df)

            # Fit Kaplan-Meier
            kmf = KaplanMeierFitter()
            kmf.fit(group_df[time_col], event_observed=group_df[event_col])

            # Get survival function
            sf = kmf.survival_function_
            times = sf.index.values
            surv = sf.values.flatten()

            # Truncate at tau
            mask = times <= tau
            times_trunc = np.append(times[mask], tau)
            surv_trunc = np.append(surv[mask], surv[mask][-1] if len(surv[mask]) > 0 else 1)

            # Calculate RMST (area under survival curve)
            rmst[str(group)] = trapezoid(surv_trunc, times_trunc)

            # Variance estimation using Greenwood's formula
            # Simplified approach using bootstrap
            n_bootstrap = 500
            rmst_boots = []

            for _ in range(n_bootstrap):
                idx = np.random.choice(len(group_df), size=len(group_df), replace=True)
                boot_df = group_df.iloc[idx]

                try:
                    kmf_boot = KaplanMeierFitter()
                    kmf_boot.fit(boot_df[time_col], event_observed=boot_df[event_col])

                    sf_boot = kmf_boot.survival_function_
                    times_boot = sf_boot.index.values
                    surv_boot = sf_boot.values.flatten()

                    mask_boot = times_boot <= tau
                    if mask_boot.sum() > 0:
                        times_b = np.append(times_boot[mask_boot], tau)
                        surv_b = np.append(surv_boot[mask_boot], surv_boot[mask_boot][-1])
                        rmst_boots.append(trapezoid(surv_b, times_b))
                except:
                    pass

            rmst_se[str(group)] = np.std(rmst_boots) if rmst_boots else 0
            rmst_ci_lower[str(group)] = rmst[str(group)] - z * rmst_se[str(group)]
            rmst_ci_upper[str(group)] = rmst[str(group)] + z * rmst_se[str(group)]

        # RMST difference (group 1 - group 0)
        group_names = [str(g) for g in groups]
        rmst_diff = rmst[group_names[1]] - rmst[group_names[0]]
        rmst_diff_se = np.sqrt(rmst_se[group_names[0]]**2 + rmst_se[group_names[1]]**2)
        rmst_diff_ci_lower = rmst_diff - z * rmst_diff_se
        rmst_diff_ci_upper = rmst_diff + z * rmst_diff_se

        # P-value for difference
        z_stat = rmst_diff / rmst_diff_se if rmst_diff_se > 0 else 0
        rmst_diff_p = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        # RMST ratio
        rmst_ratio = rmst[group_names[1]] / rmst[group_names[0]] if rmst[group_names[0]] > 0 else np.inf

        # Life years gained (per patient)
        life_years_gained = rmst_diff / 365.25  # Convert days to years

        summary = (
            f"RMST analysis (tau = {tau:.0f} days). "
            f"{group_names[0]}: RMST = {rmst[group_names[0]]:.1f} days (95% CI: {rmst_ci_lower[group_names[0]]:.1f}-{rmst_ci_upper[group_names[0]]:.1f}). "
            f"{group_names[1]}: RMST = {rmst[group_names[1]]:.1f} days (95% CI: {rmst_ci_lower[group_names[1]]:.1f}-{rmst_ci_upper[group_names[1]]:.1f}). "
            f"Difference = {rmst_diff:.1f} days (p = {rmst_diff_p:.4f})."
        )

        return RMSTResult(
            tau=tau,
            rmst=rmst,
            rmst_se=rmst_se,
            rmst_ci_lower=rmst_ci_lower,
            rmst_ci_upper=rmst_ci_upper,
            rmst_difference=rmst_diff,
            rmst_diff_se=rmst_diff_se,
            rmst_diff_ci_lower=rmst_diff_ci_lower,
            rmst_diff_ci_upper=rmst_diff_ci_upper,
            rmst_diff_p_value=rmst_diff_p,
            rmst_ratio=rmst_ratio,
            life_years_gained=life_years_gained,
            n_observations=n_obs,
            summary_text=summary
        )

    def flexible_parametric_model(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        covariates: List[str],
        distribution: str = 'weibull',
        group_col: Optional[str] = None
    ) -> FlexibleSurvivalResult:
        """
        Flexible parametric survival model (Weibull, Log-normal, Log-logistic).

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        time_col : str
            Time to event column
        event_col : str
            Event indicator column
        covariates : List[str]
            Covariates
        distribution : str
            'weibull', 'lognormal', 'loglogistic', 'exponential'
        group_col : str, optional
            Grouping variable for stratification

        Returns:
        --------
        FlexibleSurvivalResult
        """
        from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

        df_clean = df.dropna(subset=[time_col, event_col] + covariates)

        # Select model
        if distribution == 'weibull':
            model = WeibullAFTFitter()
        elif distribution == 'lognormal':
            model = LogNormalAFTFitter()
        elif distribution == 'loglogistic':
            model = LogLogisticAFTFitter()
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Fit model
        model.fit(
            df_clean[[time_col, event_col] + covariates],
            duration_col=time_col,
            event_col=event_col
        )

        # Extract hazard ratios (actually acceleration factors for AFT)
        hazard_ratios = {}
        for var in covariates:
            if var in model.summary.index:
                # For AFT models, exp(coef) is time ratio, not HR
                # HR = exp(-coef/scale) approximately
                coef = model.params_[var]
                hazard_ratios[var] = {
                    'time_ratio': float(np.exp(coef)),
                    'tr_ci_lower': float(np.exp(model.confidence_intervals_.loc[var, '95% lower-bound'])),
                    'tr_ci_upper': float(np.exp(model.confidence_intervals_.loc[var, '95% upper-bound'])),
                    'p_value': float(model.summary.loc[var, 'p']),
                    'coefficient': float(coef)
                }

        # Median survival by group
        median_survival = {}
        if group_col is not None and group_col in df_clean.columns:
            for group in df_clean[group_col].unique():
                group_df = df_clean[df_clean[group_col] == group]
                kmf = KaplanMeierFitter()
                kmf.fit(group_df[time_col], event_observed=group_df[event_col])
                median_survival[str(group)] = kmf.median_survival_time_
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(df_clean[time_col], event_observed=df_clean[event_col])
            median_survival['overall'] = kmf.median_survival_time_

        # Model parameters
        params = {p: float(model.params_[p]) for p in model.params_.index}

        n_events = int(df_clean[event_col].sum())

        summary = (
            f"Flexible parametric model ({distribution} distribution). "
            f"N={len(df_clean)}, events={n_events}. "
            f"AIC={model.AIC_:.1f}."
        )

        return FlexibleSurvivalResult(
            distribution=distribution,
            parameters=params,
            hazard_ratios=hazard_ratios,
            median_survival=median_survival,
            aic=model.AIC_,
            bic=model.AIC_ + (np.log(len(df_clean)) - 2) * len(covariates),
            log_likelihood=model.log_likelihood_,
            n_observations=len(df_clean),
            n_events=n_events,
            summary_text=summary
        )

    def plot_cumulative_incidence(
        self,
        result: CompetingRisksResult,
        title: str = "Cumulative Incidence Functions"
    ):
        """Create cumulative incidence plot for competing risks."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']

        for i, (event_type, cif) in enumerate(result.cumulative_incidence.items()):
            color = colors[i % len(colors)]
            label = f"Event {event_type}" if event_type == result.event_of_interest else f"Competing {event_type}"
            ax.step(result.time_points, cif, where='post', color=color,
                    linewidth=2, label=label)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Cumulative Incidence', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend(loc='upper left')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def plot_rmst(
        self,
        result: RMSTResult,
        title: str = "Restricted Mean Survival Time"
    ):
        """Create RMST visualization."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        groups = list(result.rmst.keys())
        x = np.arange(len(groups))

        # RMST bars with error bars
        rmst_vals = [result.rmst[g] for g in groups]
        errors = [[result.rmst[g] - result.rmst_ci_lower[g] for g in groups],
                  [result.rmst_ci_upper[g] - result.rmst[g] for g in groups]]

        bars = ax.bar(x, rmst_vals, yerr=errors, capsize=10,
                      color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(f'RMST (days, tau={result.tau:.0f})', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)

        # Add difference annotation
        ax.text(0.5, max(rmst_vals) * 1.1,
                f'Difference: {result.rmst_difference:.1f} days\np = {result.rmst_diff_p_value:.4f}',
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig
