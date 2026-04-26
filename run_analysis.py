#!/usr/bin/env python3
"""Comprehensive analysis of TAVR racial disparities dataset."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.profiler import DataProfiler
from stats.descriptive import DescriptiveStats
from stats.comparative import ComparativeTests
from stats.correlation import CorrelationAnalysis
from stats.regression import RegressionAnalysis
from stats.survival import SurvivalAnalysis
from stats.diagnostic import DiagnosticTests

# Load data
print("=" * 70)
print("TAVR RACIAL DISPARITIES ANALYSIS")
print("=" * 70)

df = pd.read_csv("/Users/agam/Downloads/tavr_racial_disparities_1000patients.csv")
print(f"\nDataset: {len(df)} patients, {len(df.columns)} variables")

# Initialize analyzers
profiler = DataProfiler()
descriptive = DescriptiveStats()
comparative = ComparativeTests()
correlation = CorrelationAnalysis()
regression = RegressionAnalysis()
survival = SurvivalAnalysis()
diagnostic = DiagnosticTests()

# ============================================================================
# 1. DATA PROFILING
# ============================================================================
print("\n" + "=" * 70)
print("1. DATA PROFILE")
print("=" * 70)

profile = profiler.profile_dataset(df)
print(f"\nRows: {profile.n_rows}")
print(f"Continuous variables: {profile.n_continuous}")
print(f"Categorical variables: {profile.n_categorical}")
print(f"Binary variables: {profile.n_binary}")
print(f"Missing data: {profile.total_missing} values ({profile.total_missing_pct:.1f}%)")

if profile.warnings:
    print("\nWarnings:")
    for w in profile.warnings[:5]:
        print(f"  - {w}")

# ============================================================================
# 2. RACIAL DEMOGRAPHICS
# ============================================================================
print("\n" + "=" * 70)
print("2. RACIAL DEMOGRAPHICS")
print("=" * 70)

race_stats = descriptive.categorical_stats(df['race'], 'Race')
print(f"\nRacial Distribution (n={race_stats.n}):")
for cat, data in race_stats.categories.items():
    print(f"  {cat}: {data['count']} ({data['percentage']:.1f}%)")

# ============================================================================
# 3. BASELINE CHARACTERISTICS BY RACE
# ============================================================================
print("\n" + "=" * 70)
print("3. BASELINE CHARACTERISTICS BY RACE")
print("=" * 70)

continuous_vars = ['age', 'bmi', 'sts_prom_score', 'lvef_baseline', 'median_household_income_zip',
                   'distance_to_center_miles', 'hospital_los_days']
categorical_vars = ['sex', 'diabetes', 'hypertension', 'ckd_stage_3_plus', 'atrial_fibrillation']

table1 = descriptive.generate_table1(
    df,
    group_col='race',
    continuous_vars=continuous_vars,
    categorical_vars=categorical_vars
)
print("\nTable 1: Baseline Characteristics by Race")
print(table1.to_string(index=False))

# ============================================================================
# 4. PRIMARY OUTCOME: 30-DAY MORTALITY BY RACE
# ============================================================================
print("\n" + "=" * 70)
print("4. PRIMARY OUTCOME: 30-DAY MORTALITY BY RACE")
print("=" * 70)

mortality_by_race = df.groupby('race')['mortality_30day'].agg(['sum', 'count', 'mean'])
mortality_by_race.columns = ['Deaths', 'Total', 'Rate']
mortality_by_race['Rate'] = mortality_by_race['Rate'] * 100
print("\n30-Day Mortality by Race:")
print(mortality_by_race.round(2))

# Chi-square test
chi_result = comparative.chi_square_independence(df, 'race', 'mortality_30day')
print(f"\nChi-square test: χ²={chi_result.statistic:.3f}, p={chi_result.p_value:.4f}")
print(f"Effect size (Cramér's V): {chi_result.effect_size:.3f}")
print(f"Conclusion: {chi_result.conclusion}")

# ============================================================================
# 5. SECONDARY OUTCOMES BY RACE
# ============================================================================
print("\n" + "=" * 70)
print("5. SECONDARY OUTCOMES BY RACE")
print("=" * 70)

outcomes = ['mortality_1year', 'stroke_30day', 'vascular_complication',
            'new_pacemaker', 'acute_kidney_injury', 'readmission_30day']

print("\nOutcome rates by race (%):")
print("-" * 60)
outcome_table = df.groupby('race')[outcomes].mean() * 100
print(outcome_table.round(2).to_string())

print("\nStatistical comparisons (Chi-square):")
for outcome in outcomes:
    result = comparative.chi_square_independence(df, 'race', outcome)
    sig = "*" if result.p_value < 0.05 else ""
    print(f"  {outcome}: χ²={result.statistic:.2f}, p={result.p_value:.4f} {sig}")

# ============================================================================
# 6. SOCIOECONOMIC DISPARITIES
# ============================================================================
print("\n" + "=" * 70)
print("6. SOCIOECONOMIC DISPARITIES BY RACE")
print("=" * 70)

# Income by race (ANOVA)
income_result = comparative.one_way_anova(df.dropna(subset=['median_household_income_zip']),
                                          'median_household_income_zip', 'race')
print(f"\nMedian Household Income by Race:")
for race in df['race'].unique():
    subset = df[df['race'] == race]['median_household_income_zip'].dropna()
    print(f"  {race}: ${subset.mean():,.0f} ± ${subset.std():,.0f}")
print(f"\nANOVA: F={income_result.statistic:.2f}, p={income_result.p_value:.4f}")
print(f"Effect size (η²): {income_result.effect_size:.3f} ({income_result.effect_size_interpretation})")

# Distance to center by race
dist_result = comparative.kruskal_wallis(df.dropna(subset=['distance_to_center_miles']),
                                         'distance_to_center_miles', 'race')
print(f"\nDistance to Center by Race:")
for race in df['race'].unique():
    subset = df[df['race'] == race]['distance_to_center_miles'].dropna()
    print(f"  {race}: {subset.median():.1f} miles (IQR: {subset.quantile(0.25):.1f}-{subset.quantile(0.75):.1f})")
print(f"\nKruskal-Wallis: H={dist_result.statistic:.2f}, p={dist_result.p_value:.4f}")

# Insurance type
print("\nInsurance Distribution by Race:")
ins_crosstab = pd.crosstab(df['race'], df['insurance_type'], normalize='index') * 100
print(ins_crosstab.round(1).to_string())

# ============================================================================
# 7. PROCEDURAL DIFFERENCES
# ============================================================================
print("\n" + "=" * 70)
print("7. PROCEDURAL CHARACTERISTICS BY RACE")
print("=" * 70)

proc_vars = ['procedure_duration_min', 'contrast_volume_ml', 'fluoroscopy_time_min']
print("\nProcedural Variables by Race (mean ± SD):")
for var in proc_vars:
    print(f"\n{var}:")
    for race in df['race'].unique():
        subset = df[df['race'] == race][var].dropna()
        print(f"  {race}: {subset.mean():.1f} ± {subset.std():.1f}")
    result = comparative.kruskal_wallis(df.dropna(subset=[var]), var, 'race')
    print(f"  Kruskal-Wallis: p={result.p_value:.4f}")

# Access site by race
print("\nAccess Site by Race:")
access_crosstab = pd.crosstab(df['race'], df['access_site'], normalize='index') * 100
print(access_crosstab.round(1).to_string())

# ============================================================================
# 8. SURVIVAL ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("8. SURVIVAL ANALYSIS")
print("=" * 70)

# Kaplan-Meier by race
km_results = survival.kaplan_meier_by_group(df, 'survival_days', 'death_event', 'race')
print("\nKaplan-Meier Survival by Race:")
for race, result in km_results.items():
    median = result.median_survival
    if median:
        print(f"  {race}: Median survival = {median:.0f} days, Events = {result.n_events}/{result.n_observations}")
    else:
        print(f"  {race}: Median survival not reached, Events = {result.n_events}/{result.n_observations}")

# Log-rank test
logrank_result = survival.log_rank_test(df, 'survival_days', 'death_event', 'race')
print(f"\nLog-rank test: χ²={logrank_result.test_statistic:.2f}, p={logrank_result.p_value:.4f}")
print(f"Conclusion: {logrank_result.summary_text}")

# ============================================================================
# 9. MULTIVARIATE ANALYSIS - LOGISTIC REGRESSION
# ============================================================================
print("\n" + "=" * 70)
print("9. MULTIVARIATE ANALYSIS: 30-DAY MORTALITY")
print("=" * 70)

# Prepare data for regression
df_reg = df.copy()

# Create dummy variables for race (White as reference)
df_reg['race_Black'] = (df_reg['race'] == 'Black').astype(int)
df_reg['race_Hispanic'] = (df_reg['race'] == 'Hispanic').astype(int)
df_reg['race_Asian'] = (df_reg['race'] == 'Asian').astype(int)
df_reg['female'] = (df_reg['sex'] == 'Female').astype(int)

predictors = ['age', 'female', 'race_Black', 'race_Hispanic', 'race_Asian',
              'sts_prom_score', 'diabetes', 'ckd_stage_3_plus', 'frailty_index']

# Drop rows with missing values in predictors
df_reg_clean = df_reg.dropna(subset=predictors + ['mortality_30day'])

logit_result = regression.logistic_regression(df_reg_clean, 'mortality_30day', predictors)

print(f"\nLogistic Regression (n={logit_result.n_observations})")
print(f"Model: {logit_result.summary_text}")
print(f"\nCoefficients:")
print("-" * 70)
print(f"{'Variable':<20} {'OR':>10} {'95% CI':>20} {'p-value':>12}")
print("-" * 70)
for var, coef in logit_result.coefficients.items():
    if var == 'Intercept':
        continue
    or_val = coef['odds_ratio']
    ci_low = coef['or_ci_lower']
    ci_high = coef['or_ci_upper']
    p = coef['p_value']
    sig = "*" if p < 0.05 else ""
    print(f"{var:<20} {or_val:>10.2f} [{ci_low:>7.2f}, {ci_high:>7.2f}] {p:>10.4f} {sig}")

# ============================================================================
# 10. COX REGRESSION FOR SURVIVAL
# ============================================================================
print("\n" + "=" * 70)
print("10. COX PROPORTIONAL HAZARDS REGRESSION")
print("=" * 70)

cox_predictors = ['age', 'female', 'race_Black', 'race_Hispanic', 'race_Asian',
                  'sts_prom_score', 'diabetes', 'ckd_stage_3_plus']

df_cox = df_reg.dropna(subset=cox_predictors + ['survival_days', 'death_event'])

cox_result = survival.cox_regression(df_cox, 'survival_days', 'death_event', cox_predictors)

print(f"\nCox Regression (n={cox_result.n_observations}, events={cox_result.n_events})")
print(f"\nHazard Ratios:")
print("-" * 70)
print(f"{'Variable':<20} {'HR':>10} {'95% CI':>20} {'p-value':>12}")
print("-" * 70)
for var, coef in cox_result.coefficients.items():
    hr = coef['hazard_ratio']
    ci_low = coef['hr_ci_lower']
    ci_high = coef['hr_ci_upper']
    p = coef['p_value']
    sig = "*" if p < 0.05 else ""
    print(f"{var:<20} {hr:>10.2f} [{ci_low:>7.2f}, {ci_high:>7.2f}] {p:>10.4f} {sig}")

# ============================================================================
# 11. KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("11. KEY FINDINGS SUMMARY")
print("=" * 70)

print("""
RACIAL DISPARITIES IN TAVR OUTCOMES - KEY FINDINGS:

1. DEMOGRAPHICS:
   - Dataset: 1000 TAVR patients""")

for cat, data in race_stats.categories.items():
    print(f"   - {cat}: {data['count']} ({data['percentage']:.1f}%)")

print(f"""
2. PRIMARY OUTCOME (30-Day Mortality):
   - Chi-square p-value: {chi_result.p_value:.4f}
   - {'SIGNIFICANT' if chi_result.p_value < 0.05 else 'NOT SIGNIFICANT'} differences by race

3. SOCIOECONOMIC DISPARITIES:
   - Income differences by race: {'SIGNIFICANT' if income_result.p_value < 0.05 else 'NOT SIGNIFICANT'} (p={income_result.p_value:.4f})
   - Effect size (η²): {income_result.effect_size:.3f} ({income_result.effect_size_interpretation})

4. SURVIVAL ANALYSIS:
   - Log-rank test: p={logrank_result.p_value:.4f}
   - {'SIGNIFICANT' if logrank_result.p_value < 0.05 else 'NOT SIGNIFICANT'} survival differences by race

5. MULTIVARIATE ANALYSIS:
   - After adjustment for clinical factors, racial disparities in mortality:""")

for race_var in ['race_Black', 'race_Hispanic', 'race_Asian']:
    if race_var in logit_result.coefficients:
        coef = logit_result.coefficients[race_var]
        or_val = coef['odds_ratio']
        p = coef['p_value']
        print(f"     {race_var}: OR={or_val:.2f}, p={p:.4f} ({'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'})")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
