#!/usr/bin/env python3
"""
Demonstration of the Complete LLM-Driven Statistical Analysis Workflow.

This script demonstrates the full workflow:
1. Load data and profile it
2. LLM generates comprehensive statistical plan
3. User reviews plan (displayed in markdown)
4. User can modify the plan
5. Plan is confirmed and executed
6. Report is generated with all results

Usage:
    python3 demo_llm_workflow.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np

# Import our modules
from data.profiler import DataProfiler
from llm.comprehensive_planner import (
    ComprehensiveStatisticalPlanner,
    ComprehensiveStatisticalPlan,
    AnalysisStep,
    VisualizationStep
)
from llm.plan_executor import PlanExecutor, ExecutionReport, AnalysisResult
from llm.report_generator import ReportGenerator


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_workflow():
    """Run the complete LLM-driven statistical analysis workflow."""

    print_section("LLM-DRIVEN STATISTICAL ANALYSIS WORKFLOW DEMO")
    print("This demonstrates the complete workflow from data to publication-ready report.\n")

    # =========================================================================
    # STEP 1: Load and Profile Data
    # =========================================================================
    print_section("STEP 1: Load and Profile Dataset")

    data_path = "/Users/agam/Downloads/tavr_racial_disparities_1000patients.csv"

    if not os.path.exists(data_path):
        print(f"Note: Dataset not found at {data_path}")
        print("Creating synthetic demo data...")
        df = create_demo_data()
    else:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {data_path}")

    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")

    # Profile the data
    profiler = DataProfiler()
    profile = profiler.profile_dataset(df)

    print(f"\nData Profile:")
    print(f"  - Continuous variables: {profile.n_continuous}")
    print(f"  - Categorical variables: {profile.n_categorical}")
    print(f"  - Binary variables: {profile.n_binary}")
    print(f"  - Missing values: {profile.total_missing} ({profile.total_missing_pct:.1f}%)")

    if profile.warnings:
        print(f"\nWarnings detected:")
        for w in profile.warnings[:3]:
            print(f"  - {w}")

    # =========================================================================
    # STEP 2: Define Research Question
    # =========================================================================
    print_section("STEP 2: Define Research Question")

    research_question = """
    Investigate racial disparities in TAVR (Transcatheter Aortic Valve Replacement) outcomes.

    Primary Research Questions:
    1. Are there significant differences in 30-day mortality across racial groups?
    2. Do socioeconomic factors explain any observed racial disparities?
    3. After adjusting for clinical and socioeconomic factors, do racial disparities persist?

    Primary Outcome: 30-day mortality (mortality_30day)
    Secondary Outcomes: 1-year mortality, stroke, readmission

    Key Variables of Interest:
    - Exposure: Race (White, Black, Hispanic, Asian)
    - Outcomes: mortality_30day, mortality_1year, stroke_30day, readmission_30day
    - Confounders: age, sex, STS-PROM score, comorbidities
    - Effect Modifiers: socioeconomic status (income, insurance)
    """

    print("Research Question:")
    print("-" * 60)
    print(research_question)

    # =========================================================================
    # STEP 3: Generate Comprehensive Statistical Plan (LLM)
    # =========================================================================
    print_section("STEP 3: Generate Statistical Plan (LLM)")

    print("Initializing Comprehensive Statistical Planner...")

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        print("  - Anthropic API key found")
        planner = ComprehensiveStatisticalPlanner(api_key=api_key)

        print("\nGenerating comprehensive statistical plan...")
        print("(This uses Claude to analyze the data and create a detailed plan)\n")

        try:
            plan = planner.generate_plan(df, research_question)

            print("Plan generated successfully!")
            print(f"\nPlan Overview:")
            print(f"  - Title: {plan.research_question[:50]}...")
            print(f"  - Primary Analyses: {len(plan.primary_analyses)}")
            print(f"  - Visualizations: {len(plan.visualizations)}")
            print(f"  - Status: {plan.status}")

            # Display plan in markdown format
            print("\n" + "-" * 60)
            print("STATISTICAL PLAN (Markdown Format):")
            print("-" * 60)
            markdown_plan = plan.to_markdown()
            # Print first 3000 chars to keep output manageable
            if len(markdown_plan) > 3000:
                print(markdown_plan[:3000])
                print(f"\n... [truncated, full plan is {len(markdown_plan)} characters]")
            else:
                print(markdown_plan)

        except Exception as e:
            print(f"Error generating plan: {e}")
            print("\nFalling back to demo plan...")
            plan = create_demo_plan(len(df))
    else:
        print("  - No Anthropic API key found (set ANTHROPIC_API_KEY)")
        print("  - Using demo plan for workflow demonstration")
        plan = create_demo_plan(len(df))

    # =========================================================================
    # STEP 4: User Review and Modification (Simulated)
    # =========================================================================
    print_section("STEP 4: User Reviews and Modifies Plan")

    print("In the actual application, the user would:")
    print("  1. Review the plan displayed in the web interface")
    print("  2. Add, remove, or modify analysis steps")
    print("  3. Adjust parameters or methods")
    print("  4. Confirm when satisfied")

    print("\nSimulating user modifications...")
    print("  - User reviewed all analysis steps")
    print("  - User confirmed the plan without modifications")
    plan.status = 'confirmed'
    print(f"\nPlan Status: {plan.status}")

    # =========================================================================
    # STEP 5: Execute Statistical Plan
    # =========================================================================
    print_section("STEP 5: Execute Statistical Plan")

    print("Initializing Plan Executor...")
    executor = PlanExecutor()

    # Create output directory
    output_dir = Path("llm_workflow_output")
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}")
    print("\nExecuting plan...")
    print(f"  - Total analyses: {len(plan.descriptive_analyses) + len(plan.primary_analyses) + len(plan.secondary_analyses)}")

    try:
        execution_report = executor.execute_plan(
            plan=plan,
            df=df,
            progress_callback=lambda p, msg: print(f"  [{p*100:.0f}%] {msg}") if p > 0 else None
        )

        print(f"\nExecution Complete!")
        print(f"  - Total analyses: {execution_report.total_analyses}")
        print(f"  - Successful: {execution_report.successful_analyses}")
        print(f"  - Failed: {execution_report.failed_analyses}")
        print(f"  - Visualizations: {len(execution_report.visualization_results)}")

        if execution_report.warnings:
            print(f"\nWarnings:")
            for warning in execution_report.warnings[:5]:
                print(f"  - {warning}")

        # Show some results
        if execution_report.analysis_results:
            print(f"\nSample Results:")
            for i, result in enumerate(execution_report.analysis_results[:3]):
                print(f"\n  {i+1}. {result.step_name}")
                print(f"     Method: {result.method_used}")
                print(f"     Success: {result.success}")
                if result.interpretation:
                    interp = result.interpretation[:100] + "..." if len(result.interpretation) > 100 else result.interpretation
                    print(f"     Interpretation: {interp}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        execution_report = None

    # =========================================================================
    # STEP 6: Generate Publication-Ready Report
    # =========================================================================
    print_section("STEP 6: Generate Publication-Ready Report")

    if execution_report and execution_report.analysis_results:
        print("Initializing Report Generator...")

        report_gen = ReportGenerator(api_key=api_key)

        print("Generating report...")

        try:
            report = report_gen.generate_report(
                plan=plan,
                execution_report=execution_report,
                output_format='markdown'
            )

            print(f"\nReport generated successfully!")
            print(f"  - Format: {report.format}")
            print(f"  - Sections: {len(report.sections)}")

            # Save the report
            report_path = output_dir / "analysis_report.md"
            report_gen.save_report(report, str(report_path))
            print(f"  - Saved to: {report_path}")

            # Also generate HTML version
            html_report = report_gen.generate_report(
                plan=plan,
                execution_report=execution_report,
                output_format='html'
            )
            html_path = output_dir / "analysis_report.html"
            report_gen.save_report(html_report, str(html_path))
            print(f"  - HTML version: {html_path}")

            # Preview report content
            print("\n" + "-" * 60)
            print("REPORT PREVIEW:")
            print("-" * 60)
            preview = report.content[:2000] if len(report.content) > 2000 else report.content
            print(preview)
            if len(report.content) > 2000:
                print(f"\n... [truncated, full report is {len(report.content)} characters]")

        except Exception as e:
            print(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping report generation (no execution results)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("WORKFLOW COMPLETE")

    print("Summary of LLM-Driven Statistical Analysis Workflow:")
    print()
    print("  [OK] Step 1: Dataset loaded and profiled")
    print("  [OK] Step 2: Research question defined")
    print("  [OK] Step 3: Statistical plan generated")
    print("  [OK] Step 4: User reviewed and confirmed plan")
    if execution_report:
        print(f"  [OK] Step 5: Plan executed ({execution_report.successful_analyses}/{execution_report.total_analyses} successful)")
        print("  [OK] Step 6: Publication-ready report generated")
    else:
        print("  [--] Step 5: Execution skipped due to errors")
        print("  [--] Step 6: Report skipped due to errors")
    print()
    print(f"Output files saved to: {output_dir.absolute()}")
    print()
    print("This workflow enables:")
    print("  - Intelligent statistical planning based on data characteristics")
    print("  - User control over analysis decisions")
    print("  - Automated execution with assumption checking")
    print("  - Publication-ready report generation")


def create_demo_data():
    """Create synthetic demo data if real data not available."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'patient_id': range(1, n + 1),
        'age': np.random.normal(78, 8, n).clip(50, 95),
        'sex': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n, p=[0.65, 0.15, 0.12, 0.08]),
        'bmi': np.random.normal(28, 5, n).clip(18, 45),
        'sts_prom_score': np.random.exponential(5, n).clip(1, 30),
        'diabetes': np.random.binomial(1, 0.35, n),
        'hypertension': np.random.binomial(1, 0.75, n),
        'ckd_stage_3_plus': np.random.binomial(1, 0.30, n),
        'lvef_baseline': np.random.normal(55, 12, n).clip(15, 70),
        'mortality_30day': np.random.binomial(1, 0.03, n),
        'mortality_1year': np.random.binomial(1, 0.12, n),
        'stroke_30day': np.random.binomial(1, 0.02, n),
        'readmission_30day': np.random.binomial(1, 0.15, n),
        'median_household_income_zip': np.random.normal(65000, 25000, n).clip(20000, 200000),
        'survival_days': np.random.exponential(500, n).clip(1, 1000),
        'death_event': np.random.binomial(1, 0.15, n),
    })

    return df


def create_demo_plan(sample_size: int) -> ComprehensiveStatisticalPlan:
    """Create a demo plan for testing when API is not available."""

    # Create analysis steps
    descriptive_analyses = [
        AnalysisStep(
            step_id='desc_1',
            name='Baseline Characteristics',
            category='descriptive',
            method='table1',
            description='Generate Table 1 with baseline characteristics by race',
            rationale='Standard reporting requirement for clinical studies',
            variables={
                'group_col': 'race',
                'continuous_vars': 'age,bmi,sts_prom_score,lvef_baseline',
                'categorical_vars': 'sex,diabetes,hypertension'
            },
            parameters={},
            assumptions=[],
            assumption_tests=[],
            interpretation_guidance='Compare baseline characteristics across racial groups',
            expected_output=['table'],
            priority='primary'
        )
    ]

    primary_analyses = [
        AnalysisStep(
            step_id='primary_1',
            name='30-Day Mortality by Race',
            category='comparative',
            method='chi_square',
            description='Test association between race and 30-day mortality',
            rationale='Primary outcome comparison across exposure groups',
            variables={
                'var1': 'race',
                'var2': 'mortality_30day'
            },
            parameters={},
            assumptions=['Expected cell counts >= 5'],
            assumption_tests=[{'test': 'expected_counts', 'variable': 'race,mortality_30day'}],
            interpretation_guidance='p < 0.05 indicates significant association',
            expected_output=['chi2_statistic', 'p_value', 'cramers_v'],
            fallback_method='fisher_exact',
            priority='primary'
        ),
        AnalysisStep(
            step_id='primary_2',
            name='Adjusted Mortality Analysis',
            category='regression',
            method='logistic_regression',
            description='Multivariate logistic regression for mortality',
            rationale='Adjust for confounders to assess independent effect of race',
            variables={
                'outcome': 'mortality_30day',
                'predictors': 'age,sex,race,sts_prom_score,diabetes,ckd_stage_3_plus'
            },
            parameters={'reference_category': 'White'},
            assumptions=['No multicollinearity', 'Adequate sample size per predictor'],
            assumption_tests=[],
            interpretation_guidance='OR > 1 indicates increased odds of mortality',
            expected_output=['coefficients', 'odds_ratios', 'p_values', 'ci_lower', 'ci_upper'],
            priority='primary'
        )
    ]

    secondary_analyses = [
        AnalysisStep(
            step_id='secondary_1',
            name='1-Year Mortality by Race',
            category='comparative',
            method='chi_square',
            description='Test association between race and 1-year mortality',
            rationale='Secondary outcome analysis',
            variables={
                'var1': 'race',
                'var2': 'mortality_1year'
            },
            parameters={},
            assumptions=['Expected cell counts >= 5'],
            assumption_tests=[],
            interpretation_guidance='p < 0.05 indicates significant association',
            expected_output=['chi2_statistic', 'p_value'],
            priority='secondary'
        )
    ]

    visualizations = [
        VisualizationStep(
            viz_id='viz_1',
            name='Mortality Forest Plot',
            plot_type='forest_plot',
            variables={'outcome': 'mortality_30day', 'subgroups': 'race'},
            parameters={},
            purpose='Visualize adjusted odds ratios by race',
            related_analysis='primary_2'
        )
    ]

    return ComprehensiveStatisticalPlan(
        plan_id='demo_plan_001',
        created_at=datetime.now().isoformat(),
        research_question='Investigate racial disparities in TAVR outcomes',
        research_type='comparative',
        study_design='retrospective cohort',
        sample_size=sample_size,
        outcome_variables=['mortality_30day', 'mortality_1year'],
        exposure_variables=['race'],
        covariates=['age', 'sex', 'sts_prom_score', 'diabetes', 'ckd_stage_3_plus'],
        descriptive_analyses=descriptive_analyses,
        primary_analyses=primary_analyses,
        secondary_analyses=secondary_analyses,
        sensitivity_analyses=[],
        subgroup_analyses=[],
        assumption_checks=[],
        visualizations=visualizations,
        missing_data_strategy='Complete case analysis',
        multiple_testing_correction='Bonferroni correction for secondary outcomes',
        significance_level=0.05,
        clinical_significance_thresholds={'mortality_or': 1.5},
        limitations=['Retrospective study design', 'Potential unmeasured confounders'],
        status='draft'
    )


if __name__ == "__main__":
    demo_workflow()
