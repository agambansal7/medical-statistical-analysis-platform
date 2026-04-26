#!/usr/bin/env python3
"""
Demonstration of Advanced Statistical Features
==============================================

This script demonstrates the newly implemented advanced statistical methods:
1. Propensity Score Analysis (PSM, IPW, Doubly Robust)
2. Advanced Survival Analysis (Competing Risks, RMST, Landmark)
3. Meta-Analysis (Fixed/Random Effects, Publication Bias)
4. Missing Data Handling (MICE, Sensitivity Analysis)
5. Subgroup Analysis (Interaction Testing, Credibility Assessment)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stats.propensity_score import PropensityScoreAnalysis
from stats.advanced_survival import AdvancedSurvivalAnalysis
from stats.meta_analysis import MetaAnalysis, StudyData
from stats.missing_data import MissingDataHandler
from stats.subgroup_analysis import SubgroupAnalyzer

# Create output directory
output_dir = Path(__file__).parent / "advanced_analysis_output"
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("DEMONSTRATION OF ADVANCED STATISTICAL FEATURES")
print("=" * 80)

# =============================================================================
# GENERATE SAMPLE DATA
# =============================================================================
print("\n" + "=" * 80)
print("GENERATING SAMPLE DATA")
print("=" * 80)

np.random.seed(42)
n = 500

# Create sample clinical trial data
df = pd.DataFrame({
    'patient_id': range(1, n + 1),
    'age': np.random.normal(65, 10, n),
    'sex': np.random.choice(['Male', 'Female'], n),
    'bmi': np.random.normal(28, 5, n),
    'diabetes': np.random.binomial(1, 0.3, n),
    'hypertension': np.random.binomial(1, 0.5, n),
    'smoking': np.random.choice(['Never', 'Former', 'Current'], n, p=[0.4, 0.4, 0.2]),
    'treatment': np.random.binomial(1, 0.5, n),
})

# Add confounded treatment assignment (for propensity score demo)
df['treatment'] = (
    0.3 * (df['age'] > 65).astype(int) +
    0.2 * df['diabetes'] +
    0.1 * (df['bmi'] > 30).astype(int) +
    np.random.normal(0, 0.3, n)
) > 0.3
df['treatment'] = df['treatment'].astype(int)

# Generate outcomes (treatment has true effect + confounding)
true_effect = -0.5  # Treatment reduces outcome
df['continuous_outcome'] = (
    50 +
    0.3 * df['age'] +
    5 * df['diabetes'] +
    0.2 * df['bmi'] +
    true_effect * 10 * df['treatment'] +
    np.random.normal(0, 10, n)
)

df['binary_outcome'] = (
    -2 +
    0.03 * df['age'] +
    0.5 * df['diabetes'] +
    true_effect * df['treatment'] +
    np.random.normal(0, 0.5, n)
) > 0
df['binary_outcome'] = df['binary_outcome'].astype(int)

# Survival data
df['survival_time'] = np.random.exponential(365, n) * np.exp(-0.5 * df['treatment'])
df['death_event'] = np.random.binomial(1, 0.3, n)
df['competing_event'] = np.where(df['death_event'] == 0, np.random.binomial(1, 0.1, n), 0)
df['event_type'] = np.where(df['death_event'] == 1, 1, np.where(df['competing_event'] == 1, 2, 0))

# Add some missing data
missing_idx = np.random.choice(n, size=int(n * 0.15), replace=False)
df.loc[missing_idx[:50], 'bmi'] = np.nan
df.loc[missing_idx[50:], 'continuous_outcome'] = np.nan

print(f"Generated dataset with {n} patients")
print(f"Treatment: {df['treatment'].sum()} treated, {(1-df['treatment']).sum()} control")
print(f"Missing values: {df.isna().sum().sum()}")

# =============================================================================
# 1. PROPENSITY SCORE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("1. PROPENSITY SCORE ANALYSIS")
print("=" * 80)

ps_analyzer = PropensityScoreAnalysis()

# Prepare data (remove missing for PS analysis)
df_ps = df.dropna(subset=['age', 'diabetes', 'hypertension', 'bmi'])
df_ps['female'] = (df_ps['sex'] == 'Female').astype(int)

covariates = ['age', 'female', 'bmi', 'diabetes', 'hypertension']

# 1a. Estimate propensity scores
print("\n--- Propensity Score Estimation ---")
ps_result = ps_analyzer.estimate_propensity_scores(
    df_ps, 'treatment', covariates, method='logistic'
)
print(ps_result.summary_text)
print(f"  AUC: {ps_result.auc:.3f}")
print(f"  Brier Score: {ps_result.brier_score:.3f}")

# 1b. Propensity Score Matching
print("\n--- Propensity Score Matching ---")
match_result = ps_analyzer.match(
    df_ps, 'treatment', ps_result.propensity_scores, covariates,
    method='nearest', caliper=0.2, ratio=1
)
print(match_result.summary_text)
print(f"  Matched: {match_result.n_treated_matched} treated, {match_result.n_control_matched} controls")
print(f"  Unmatched: {match_result.n_unmatched}")

# 1c. Inverse Probability Weighting
print("\n--- Inverse Probability Weighting ---")
ipw_result = ps_analyzer.inverse_probability_weighting(
    df_ps, 'treatment', 'continuous_outcome', ps_result.propensity_scores,
    estimand='ATE', stabilized=True
)
print(ipw_result.summary_text)
print(f"  ATE: {ipw_result.ate:.4f} (95% CI: {ipw_result.ate_ci_lower:.4f}, {ipw_result.ate_ci_upper:.4f})")
print(f"  ATT: {ipw_result.att:.4f}")

# 1d. Doubly Robust Estimation
print("\n--- Doubly Robust Estimation ---")
dr_result = ps_analyzer.doubly_robust_estimation(
    df_ps, 'treatment', 'continuous_outcome', covariates
)
print(dr_result.summary_text)
print(f"  ATE: {dr_result.ate:.4f} (p = {dr_result.ate_p_value:.4f})")

# 1e. Balance Assessment
print("\n--- Covariate Balance ---")
balance_result = ps_analyzer.assess_balance(df_ps, 'treatment', covariates)
print(balance_result.summary_text)
for var, smd in balance_result.smd_before.items():
    print(f"  {var}: SMD = {smd:.3f}")

# Create Love plot
fig = ps_analyzer.plot_balance(
    match_result.standardized_mean_diff_before,
    match_result.standardized_mean_diff_after,
    title="Covariate Balance Before and After Matching"
)
fig.savefig(output_dir / 'propensity_balance_plot.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'propensity_balance_plot.png'}")
plt.close()

# =============================================================================
# 2. ADVANCED SURVIVAL ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("2. ADVANCED SURVIVAL ANALYSIS")
print("=" * 80)

adv_surv = AdvancedSurvivalAnalysis()

# 2a. Competing Risks Analysis
print("\n--- Competing Risks Analysis ---")
cr_result = adv_surv.competing_risks_analysis(
    df, 'survival_time', 'event_type', event_of_interest=1
)
print(cr_result.summary_text)
print(f"  Event of interest: {cr_result.n_events['1']} events")
print(f"  Competing events: {cr_result.n_events.get('2', 0)} events")

# Plot cumulative incidence
fig = adv_surv.plot_cumulative_incidence(cr_result, "Cumulative Incidence Functions")
fig.savefig(output_dir / 'competing_risks_cif.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'competing_risks_cif.png'}")
plt.close()

# 2b. Landmark Analysis
print("\n--- Landmark Analysis ---")
df['treatment_group'] = df['treatment'].map({0: 'Control', 1: 'Treatment'})
landmark_result = adv_surv.landmark_analysis(
    df, 'survival_time', 'death_event', 'treatment_group',
    landmark_time=90, horizon=365
)
print(landmark_result.summary_text)
print(f"  N at risk at landmark: {landmark_result.n_at_risk_at_landmark}")
print(f"  Events after landmark: {landmark_result.n_events_after_landmark}")

# 2c. Restricted Mean Survival Time
print("\n--- Restricted Mean Survival Time ---")
rmst_result = adv_surv.restricted_mean_survival_time(
    df, 'survival_time', 'death_event', 'treatment_group', tau=365
)
print(rmst_result.summary_text)
for group, rmst in rmst_result.rmst.items():
    print(f"  {group}: RMST = {rmst:.1f} days")
print(f"  Difference: {rmst_result.rmst_difference:.1f} days (p = {rmst_result.rmst_diff_p_value:.4f})")

# Plot RMST
fig = adv_surv.plot_rmst(rmst_result, "Restricted Mean Survival Time by Treatment")
fig.savefig(output_dir / 'rmst_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'rmst_plot.png'}")
plt.close()

# =============================================================================
# 3. META-ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("3. META-ANALYSIS")
print("=" * 80)

meta = MetaAnalysis()

# Create sample study data (simulating multiple RCTs)
print("\n--- Creating Sample Studies ---")
study_data = [
    {'study_id': 'Smith 2018', 'events_t': 15, 'n_t': 100, 'events_c': 25, 'n_c': 100},
    {'study_id': 'Jones 2019', 'events_t': 22, 'n_t': 150, 'events_c': 35, 'n_c': 150},
    {'study_id': 'Chen 2020', 'events_t': 8, 'n_t': 80, 'events_c': 18, 'n_c': 80},
    {'study_id': 'Brown 2021', 'events_t': 30, 'n_t': 200, 'events_c': 45, 'n_c': 200},
    {'study_id': 'Garcia 2022', 'events_t': 12, 'n_t': 120, 'events_c': 20, 'n_c': 120},
]

studies = meta.prepare_binary_data(study_data, measure='OR')

print("Studies prepared:")
for s in studies:
    print(f"  {s.study_id}: OR = {np.exp(s.effect_size):.2f}, SE = {s.se:.3f}")

# 3a. Fixed Effects Meta-Analysis
print("\n--- Fixed Effects Meta-Analysis ---")
fe_result = meta.fixed_effects(studies)
print(fe_result.summary_text)
print(f"  Pooled OR: {np.exp(fe_result.pooled_effect):.3f}")
print(f"  95% CI: ({np.exp(fe_result.pooled_ci_lower):.3f}, {np.exp(fe_result.pooled_ci_upper):.3f})")
print(f"  I²: {fe_result.heterogeneity.i_squared:.1f}%")

# 3b. Random Effects Meta-Analysis
print("\n--- Random Effects Meta-Analysis ---")
re_result = meta.random_effects(studies, method='dersimonian_laird')
print(re_result.summary_text)
print(f"  Pooled OR: {np.exp(re_result.pooled_effect):.3f}")
print(f"  95% CI: ({np.exp(re_result.pooled_ci_lower):.3f}, {np.exp(re_result.pooled_ci_upper):.3f})")
print(f"  τ²: {re_result.heterogeneity.tau_squared:.4f}")
print(f"  I²: {re_result.heterogeneity.i_squared:.1f}% ({re_result.heterogeneity.interpretation})")

# 3c. Publication Bias
print("\n--- Publication Bias Assessment ---")
pub_bias = meta.publication_bias(studies, re_result.pooled_effect)
print(pub_bias.summary_text)
print(f"  Egger's test: intercept = {pub_bias.egger_intercept:.3f}, p = {pub_bias.egger_p_value:.4f}")
print(f"  Begg's test: p = {pub_bias.begg_p_value:.4f}")
print(f"  Trim-and-fill: {pub_bias.trim_fill_added} studies imputed")

# Create Forest Plot
fig = meta.forest_plot(re_result, studies, title="Forest Plot: Treatment vs Control",
                       measure_label="Odds Ratio (95% CI)", exponentiate=True)
fig.savefig(output_dir / 'meta_forest_plot.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'meta_forest_plot.png'}")
plt.close()

# Create Funnel Plot
fig = meta.funnel_plot(studies, re_result.pooled_effect, title="Funnel Plot")
fig.savefig(output_dir / 'meta_funnel_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'meta_funnel_plot.png'}")
plt.close()

# =============================================================================
# 4. MISSING DATA HANDLING
# =============================================================================
print("\n" + "=" * 80)
print("4. MISSING DATA HANDLING")
print("=" * 80)

missing_handler = MissingDataHandler()

# 4a. Missing Pattern Analysis
print("\n--- Missing Data Pattern Analysis ---")
pattern_result = missing_handler.analyze_missing_patterns(
    df, variables=['age', 'bmi', 'diabetes', 'continuous_outcome']
)
print(pattern_result.summary_text)
print(f"  Complete cases: {pattern_result.n_complete_cases} ({pattern_result.pct_complete_cases:.1f}%)")
print(f"  Missing patterns: {pattern_result.n_patterns}")

print("\nMissing by variable:")
for var, info in pattern_result.missing_by_variable.items():
    print(f"  {var}: {info['n_missing']} ({info['pct_missing']:.1f}%)")

print("\nMost common patterns:")
for pattern, count, pct in pattern_result.most_common_patterns[:3]:
    print(f"  {pattern}: n={count} ({pct:.1f}%)")

# 4b. Multiple Imputation
print("\n--- Multiple Imputation (MICE) ---")
mi_result = missing_handler.multiple_imputation(
    df,
    variables_to_impute=['bmi', 'continuous_outcome'],
    predictor_variables=['age', 'diabetes', 'treatment'],
    n_imputations=5,
    method='mice'
)
print(mi_result.summary_text)
print(f"  Created {mi_result.n_imputations} imputed datasets")

# 4c. Compare Complete Case vs Imputed
print("\n--- Complete Case vs Multiple Imputation ---")
comparison = missing_handler.compare_complete_case_vs_imputed(
    df, mi_result.imputed_datasets, 'continuous_outcome', 'treatment'
)
print(comparison.summary_text)
print(f"  Complete case estimate: {comparison.complete_case_estimate:.4f}")
print(f"  MI estimate: {comparison.imputed_estimate:.4f}")
print(f"  Relative efficiency: {comparison.relative_efficiency:.2f}")

# Plot missing pattern
fig = missing_handler.plot_missing_pattern(pattern_result, "Missing Data Pattern Analysis")
fig.savefig(output_dir / 'missing_pattern_plot.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'missing_pattern_plot.png'}")
plt.close()

# =============================================================================
# 5. SUBGROUP ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("5. SUBGROUP ANALYSIS")
print("=" * 80)

subgroup = SubgroupAnalyzer()

# Use complete cases for subgroup analysis
df_complete = df.dropna()
df_complete['age_group'] = pd.cut(df_complete['age'], bins=[0, 60, 70, 100],
                                   labels=['<60', '60-70', '>70'])
df_complete['bmi_group'] = pd.cut(df_complete['bmi'], bins=[0, 25, 30, 50],
                                   labels=['Normal', 'Overweight', 'Obese'])

# 5a. Full Subgroup Analysis
print("\n--- Subgroup Analysis with Interaction Testing ---")
subgroup_result = subgroup.analyze_subgroups(
    df_complete,
    outcome_var='continuous_outcome',
    treatment_var='treatment',
    subgroup_vars=['sex', 'diabetes', 'age_group', 'bmi_group'],
    outcome_type='continuous',
    multiplicity_method='fdr_bh'
)
print(subgroup_result.summary_text)

print("\nInteraction Tests:")
for var, interaction in subgroup_result.interaction_tests.items():
    sig = "*" if interaction.interaction_p_value < 0.05 else ""
    print(f"  {var}: p = {interaction.interaction_p_value:.4f} {sig}")
    print(f"    {interaction.conclusion}")

print("\nSubgroup Effects:")
for var, effects in subgroup_result.subgroup_effects.items():
    print(f"\n  {var}:")
    for effect in effects:
        print(f"    {effect.subgroup_value}: {effect.effect_estimate:.2f} "
              f"({effect.effect_ci_lower:.2f}, {effect.effect_ci_upper:.2f}), n={effect.n_observations}")

# 5b. Credibility Assessment
print("\n--- Credibility Assessment ---")
for var, cred in subgroup_result.credibility_assessment.items():
    print(f"\n  {var}:")
    print(f"    Credibility: {cred['level']} (score: {cred['score']})")
    print(f"    {cred['recommendation']}")

# 5c. Consistency Analysis
print("\n--- Consistency of Treatment Effects ---")
consistency = subgroup.consistency_analysis(
    df_complete,
    'continuous_outcome',
    'treatment',
    ['sex', 'diabetes', 'age_group', 'bmi_group'],
    outcome_type='continuous'
)
print(f"  Consistency Level: {consistency['consistency_level']}")
print(f"  Same direction across subgroups: {consistency['same_direction']}")
print(f"  Significant interactions: {consistency['n_significant_interactions']}")
print(f"  {consistency['conclusion']}")

# Create Subgroup Forest Plot
fig = subgroup.create_subgroup_forest_plot(
    subgroup_result,
    exponentiate=False,
    title="Subgroup Analysis: Treatment Effect on Continuous Outcome"
)
fig.savefig(output_dir / 'subgroup_forest_plot.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'subgroup_forest_plot.png'}")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)

print(f"\nAll outputs saved to: {output_dir}")
print("\nGenerated files:")
for f in sorted(output_dir.glob('*.png')):
    print(f"  - {f.name}")

print("""
SUMMARY OF IMPLEMENTED METHODS:

1. PROPENSITY SCORE ANALYSIS
   - Propensity score estimation (logistic regression, GBM)
   - Nearest neighbor matching with caliper
   - Inverse probability weighting (IPW)
   - Doubly robust (AIPW) estimation
   - Covariate balance assessment (Love plots)

2. ADVANCED SURVIVAL ANALYSIS
   - Competing risks (Aalen-Johansen, Fine-Gray)
   - Landmark analysis
   - Restricted Mean Survival Time (RMST)
   - Flexible parametric models

3. META-ANALYSIS
   - Fixed effects (inverse variance)
   - Random effects (DerSimonian-Laird, REML)
   - Heterogeneity assessment (I², Q-test, τ²)
   - Publication bias (Egger's, Begg's, trim-and-fill)
   - Forest plots and funnel plots
   - Subgroup analysis and meta-regression

4. MISSING DATA HANDLING
   - Missing pattern analysis
   - Multiple imputation (MICE)
   - Rubin's rules for pooling
   - Sensitivity analysis (MNAR)
   - Complete case vs MI comparison

5. SUBGROUP ANALYSIS
   - Treatment-subgroup interaction testing
   - Multiplicity adjustment (Bonferroni, FDR)
   - Credibility assessment
   - Consistency analysis
   - Subgroup forest plots
""")
