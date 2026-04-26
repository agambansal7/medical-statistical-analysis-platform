"""Prompt templates for LLM interactions."""


class PromptTemplates:
    """Templates for various LLM prompts."""

    @staticmethod
    def analysis_recommendation_system() -> str:
        return """You are an expert biostatistician specializing in medical research.
Your role is to recommend appropriate statistical analyses based on research questions and data characteristics.

Key principles:
1. Always prioritize clinical relevance over statistical significance
2. Recommend effect size measures alongside hypothesis tests
3. Consider sample size limitations
4. Suggest both primary and sensitivity analyses
5. Account for multiple testing when appropriate
6. Explain assumptions clearly"""

    @staticmethod
    def interpretation_system() -> str:
        return """You are an expert biostatistician writing results interpretations for medical research publications.

Your interpretations should:
1. Use proper statistical notation (APA style)
2. Report effect sizes with confidence intervals
3. Distinguish statistical from clinical significance
4. Note limitations and caveats
5. Be accessible to clinical readers
6. Avoid overstating findings"""

    @staticmethod
    def methods_section_system() -> str:
        return """You are writing the statistical methods section for a medical research paper.

Guidelines:
1. Use past tense throughout
2. Specify all statistical tests and their assumptions
3. State the significance threshold (typically α = 0.05)
4. Describe handling of missing data
5. Mention software and versions used
6. Follow STROBE/CONSORT guidelines as appropriate
7. Include power analysis if performed"""

    @staticmethod
    def data_exploration_prompt(data_summary: dict) -> str:
        return f"""Based on this dataset summary, identify:

1. Potential research questions that could be answered
2. Key variables and their roles (outcome, predictor, confounder)
3. Data quality issues that need attention
4. Recommended preprocessing steps

DATASET SUMMARY:
- Observations: {data_summary.get('n_rows', 'Unknown')}
- Variables: {data_summary.get('n_columns', 'Unknown')}
- Continuous: {data_summary.get('n_continuous', 0)}
- Categorical: {data_summary.get('n_categorical', 0)}
- Binary: {data_summary.get('n_binary', 0)}
- Missing values: {data_summary.get('total_missing_pct', 0)}%

Variable details:
{data_summary.get('variables_summary', 'Not available')}"""

    @staticmethod
    def assumption_check_interpretation(test_name: str, results: dict) -> str:
        return f"""Interpret these assumption check results for {test_name}:

Results:
{results}

Explain:
1. What each test checks
2. Whether assumptions are satisfied
3. What to do if assumptions are violated
4. Recommended alternative approaches if needed"""

    @staticmethod
    def results_summary_prompt(analyses: list, results: dict) -> str:
        analyses_str = '\n'.join([f"- {a}" for a in analyses])
        return f"""Summarize these statistical analysis results for a research audience:

ANALYSES PERFORMED:
{analyses_str}

RESULTS:
{results}

Provide:
1. Key findings summary (2-3 sentences)
2. Main effect sizes and their interpretation
3. Clinical significance considerations
4. Limitations of the analyses
5. Suggested next steps"""

    @staticmethod
    def table_generation_prompt(table_type: str, data_desc: dict) -> str:
        return f"""Generate a {table_type} based on this data description:

{data_desc}

Format the table for medical journal publication.
Include:
- Appropriate headers and labels
- Correct decimal places
- P-values in standard format
- Effect sizes where applicable
- Sample sizes (n) for each group"""

    @staticmethod
    def figure_recommendation_prompt(analysis_type: str, results: dict) -> str:
        return f"""Recommend visualizations for these {analysis_type} results:

{results}

For each recommendation, specify:
1. Figure type (e.g., forest plot, Kaplan-Meier curve)
2. Variables to include
3. Key elements to highlight
4. Publication-ready formatting suggestions"""

    @staticmethod
    def clinical_interpretation_prompt(statistical_results: dict, clinical_context: str) -> str:
        return f"""Translate these statistical findings into clinical meaning:

STATISTICAL RESULTS:
{statistical_results}

CLINICAL CONTEXT:
{clinical_context}

Provide:
1. What these results mean for patient care
2. Magnitude of effect in clinically meaningful terms
3. Comparison to existing benchmarks if applicable
4. Caveats for clinical application
5. Recommendations for clinical practice"""

    @staticmethod
    def sample_size_justification_prompt(test_type: str, parameters: dict) -> str:
        return f"""Provide sample size justification for a {test_type}:

Parameters:
{parameters}

Include:
1. Power analysis calculation explanation
2. Assumptions made
3. Effect size justification
4. Final sample size recommendation
5. Considerations for dropout/attrition"""

    @staticmethod
    def missing_data_strategy_prompt(missing_summary: dict) -> str:
        return f"""Recommend a missing data handling strategy:

MISSING DATA SUMMARY:
{missing_summary}

Consider:
1. Pattern of missingness (MCAR, MAR, MNAR)
2. Proportion missing per variable
3. Recommended imputation method
4. Sensitivity analyses to perform
5. Reporting requirements"""

    @staticmethod
    def multiple_testing_prompt(n_tests: int, test_descriptions: list) -> str:
        tests_str = '\n'.join([f"- {t}" for t in test_descriptions])
        return f"""Recommend multiple testing correction for these {n_tests} tests:

TESTS:
{tests_str}

Consider:
1. Relationship between tests (independent vs. correlated)
2. Appropriate correction method (Bonferroni, Holm, FDR)
3. Adjusted significance threshold
4. Reporting approach
5. Trade-offs between Type I and Type II error"""

    @staticmethod
    def subgroup_analysis_prompt(main_results: dict, subgroups: list) -> str:
        subgroups_str = '\n'.join([f"- {s}" for s in subgroups])
        return f"""Plan subgroup analyses based on these main results:

MAIN RESULTS:
{main_results}

POTENTIAL SUBGROUPS:
{subgroups_str}

Provide:
1. Rationale for each subgroup analysis
2. Statistical approach (interaction terms vs. stratified)
3. Sample size considerations per subgroup
4. Interpretation caveats
5. Reporting recommendations (avoiding cherry-picking)"""
