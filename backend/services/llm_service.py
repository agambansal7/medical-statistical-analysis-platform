"""LLM Service for intelligent analysis recommendations."""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher
import anthropic

from core.config import settings


def find_closest_column(target: str, columns: List[str], threshold: float = 0.6) -> Optional[str]:
    """Find the closest matching column name using fuzzy matching.

    Args:
        target: Target column name to match
        columns: List of available column names
        threshold: Minimum similarity score (0-1) to consider a match

    Returns:
        Closest matching column name or None if no match above threshold
    """
    if not target or not columns:
        return None

    target_lower = target.lower().replace('_', ' ').replace('-', ' ')
    best_match = None
    best_score = 0

    for col in columns:
        col_lower = col.lower().replace('_', ' ').replace('-', ' ')

        # Exact match (case-insensitive)
        if target_lower == col_lower:
            return col

        # Check if target is contained in column or vice versa
        if target_lower in col_lower or col_lower in target_lower:
            score = 0.85
        else:
            # Fuzzy matching
            score = SequenceMatcher(None, target_lower, col_lower).ratio()

        if score > best_score:
            best_score = score
            best_match = col

    return best_match if best_score >= threshold else None


def _load_api_key():
    """Load API key from settings or .env file directly."""
    # First try settings
    if settings.ANTHROPIC_API_KEY:
        return settings.ANTHROPIC_API_KEY

    # Then try environment
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ.get("ANTHROPIC_API_KEY")

    # Finally try loading .env file directly
    env_paths = [
        Path(__file__).parent.parent.parent / ".env",
        Path(__file__).parent.parent / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        return line.split("=", 1)[1].strip()

    return ""


class LLMService:
    """Service for LLM-powered analysis recommendations."""

    def __init__(self):
        api_key = _load_api_key()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in settings or .env file")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = settings.LLM_MODEL or "claude-sonnet-4-20250514"

    def analyze_research_question(self, question: str,
                                  data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research question and recommend analyses.

        Args:
            question: Research question from user
            data_profile: Data profile dictionary

        Returns:
            Analysis plan dictionary with validated variables
        """
        system_prompt = self._get_analysis_system_prompt()
        user_prompt = self._build_analysis_prompt(question, data_profile)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        parsed_result = self._parse_analysis_response(response.content[0].text, question)

        # Validate and map variables if parsing was successful
        if parsed_result.get("success") and parsed_result.get("plan"):
            # Get column names from data profile
            df_columns = [v.get("name", "") for v in data_profile.get("variables", [])]
            validated_plan = self._validate_and_map_variables(
                parsed_result["plan"],
                data_profile,
                df_columns
            )
            parsed_result["plan"] = validated_plan
            parsed_result["require_confirmation"] = True

        return parsed_result

    def _validate_and_map_variables(self, plan: Dict, data_profile: Dict,
                                    df_columns: List[str]) -> Dict:
        """Validate all variables in the plan exist in the dataset.

        Returns plan with validated mappings and any warnings.
        """
        validated_plan = plan.copy()
        warnings = []
        variable_mappings = {}

        # Validate primary analyses
        for analysis in validated_plan.get("primary_analyses", []):
            self._validate_analysis_variables(
                analysis, df_columns, warnings, variable_mappings
            )

        # Validate secondary analyses
        for analysis in validated_plan.get("secondary_analyses", []):
            self._validate_analysis_variables(
                analysis, df_columns, warnings, variable_mappings
            )

        validated_plan["variable_warnings"] = warnings
        validated_plan["variable_mappings"] = variable_mappings
        validated_plan["validated"] = len([w for w in warnings if "not found" in w.lower()]) == 0
        validated_plan["available_columns"] = df_columns

        return validated_plan

    def _validate_analysis_variables(self, analysis: Dict, df_columns: List[str],
                                     warnings: List[str],
                                     variable_mappings: Dict) -> None:
        """Validate variables in a single analysis and update mappings."""
        api_call = analysis.get("api_call", {})
        params = api_call.get("parameters", {})

        # Parameters that should be column names
        column_params = [
            "outcome", "outcome_var", "group", "group_var", "group_col",
            "time", "time_var", "event", "event_var", "row_var", "col_var",
            "var1", "var2", "variable", "value_col"
        ]

        # Parameters that should be lists of column names
        list_params = [
            "predictors", "covariates", "continuous_vars", "categorical_vars",
            "variables", "features"
        ]

        # Validate single column parameters
        for param_name in column_params:
            if param_name in params:
                param_value = params[param_name]
                if isinstance(param_value, str):
                    validated = self._validate_single_column(
                        param_value, df_columns, warnings, variable_mappings
                    )
                    if validated:
                        params[param_name] = validated

        # Validate list parameters
        for param_name in list_params:
            if param_name in params:
                param_value = params[param_name]
                if isinstance(param_value, list):
                    validated_list = []
                    for col in param_value:
                        if isinstance(col, str):
                            validated = self._validate_single_column(
                                col, df_columns, warnings, variable_mappings
                            )
                            if validated:
                                validated_list.append(validated)
                    params[param_name] = validated_list
                elif isinstance(param_value, str):
                    # Handle comma-separated string
                    cols = [c.strip() for c in param_value.split(',')]
                    validated_list = []
                    for col in cols:
                        validated = self._validate_single_column(
                            col, df_columns, warnings, variable_mappings
                        )
                        if validated:
                            validated_list.append(validated)
                    params[param_name] = validated_list

    def _validate_single_column(self, col_name: str, df_columns: List[str],
                                warnings: List[str],
                                variable_mappings: Dict) -> Optional[str]:
        """Validate a single column name and return the validated name."""
        if col_name in df_columns:
            variable_mappings[col_name] = col_name
            return col_name

        # Try fuzzy matching
        closest = find_closest_column(col_name, df_columns)
        if closest:
            warnings.append(f"Variable '{col_name}' mapped to '{closest}'")
            variable_mappings[col_name] = closest
            return closest
        else:
            warnings.append(f"Column '{col_name}' not found in dataset")
            return None

    def interpret_results(self, analysis_name: str,
                         results: Dict[str, Any],
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Generate interpretation of analysis results.

        Args:
            analysis_name: Name of analysis
            results: Results dictionary
            context: Additional context

        Returns:
            Interpretation text
        """
        system_prompt = """You are an expert biostatistician writing results interpretations
for medical research papers. Use APA-style statistical reporting. Always discuss both
statistical and clinical significance. Be clear and accessible."""

        user_prompt = f"""Interpret these statistical results:

ANALYSIS: {analysis_name}

RESULTS:
{json.dumps(results, indent=2, default=str)}

CONTEXT:
{json.dumps(context or {}, indent=2, default=str)}

Provide a clear interpretation suitable for a research paper's results section."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def chat(self, message: str, context: Dict[str, Any],
             chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Handle conversational interaction.

        Args:
            message: User message
            context: Current analysis context
            chat_history: Previous chat messages

        Returns:
            Response dictionary with message and optional actions
        """
        system_prompt = """You are a helpful biostatistics assistant helping researchers
analyze medical data. You can:
1. Answer questions about statistical methods
2. Recommend appropriate analyses
3. Explain results
4. Suggest visualizations
5. Help interpret findings

Be conversational but precise. When recommending analyses, be specific about
which test to use and why. If the user wants to run an analysis, provide the
exact parameters needed.

When you recommend running a specific analysis, include a JSON block like:
```json
{"action": "run_analysis", "analysis_type": "independent_ttest", "parameters": {...}}
```"""

        # Build messages with history
        messages = []
        for msg in chat_history[-10:]:  # Last 10 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add context summary if data is loaded
        context_summary = ""
        if context.get("data_loaded"):
            context_summary = f"""
[CONTEXT: Data loaded - {context.get('filename', 'Unknown file')},
{context.get('n_rows', 0)} rows, {context.get('n_cols', 0)} columns.
Variables: {', '.join(context.get('variables', [])[:10])}...]
"""

        messages.append({
            "role": "user",
            "content": f"{context_summary}\n\nUser: {message}"
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages
        )

        response_text = response.content[0].text

        # Check for action blocks
        action = None
        if "```json" in response_text:
            try:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
                action = json.loads(json_str)
            except:
                pass

        return {
            "message": response_text,
            "action": action
        }

    def suggest_visualizations(self, analysis_type: str,
                               results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations for results.

        Args:
            analysis_type: Type of analysis performed
            results: Analysis results

        Returns:
            List of visualization suggestions
        """
        viz_map = {
            "independent_ttest": [
                {"type": "boxplot", "description": "Compare group distributions"},
                {"type": "violin", "description": "Show distribution shape by group"}
            ],
            "paired_ttest": [
                {"type": "paired", "description": "Show before-after changes"},
                {"type": "histogram", "description": "Distribution of differences"}
            ],
            "one_way_anova": [
                {"type": "boxplot", "description": "Compare all groups"},
                {"type": "bar", "description": "Mean comparison with error bars"}
            ],
            "correlation": [
                {"type": "scatter", "description": "Scatter plot with regression"},
                {"type": "correlation_heatmap", "description": "Correlation matrix"}
            ],
            "linear_regression": [
                {"type": "scatter", "description": "Observed vs predicted"},
                {"type": "residual", "description": "Residual diagnostics"}
            ],
            "logistic_regression": [
                {"type": "roc", "description": "ROC curve"},
                {"type": "forest", "description": "Odds ratios forest plot"}
            ],
            "kaplan_meier": [
                {"type": "survival", "description": "Kaplan-Meier curves"}
            ],
            "roc_analysis": [
                {"type": "roc", "description": "ROC curve with AUC"}
            ]
        }

        return viz_map.get(analysis_type, [
            {"type": "histogram", "description": "Distribution plot"}
        ])

    def generate_methods_section(self, analyses: List[Dict],
                                 data_description: Dict) -> str:
        """Generate methods section for publication.

        Args:
            analyses: List of analyses performed
            data_description: Description of the dataset

        Returns:
            Methods section text
        """
        system_prompt = """Write the statistical methods section for a medical research paper.
Use past tense. Be specific about tests and assumptions. Follow STROBE guidelines."""

        user_prompt = f"""Write the statistical methods section:

DATA:
- Sample size: {data_description.get('n_rows', 'Unknown')}
- Variables analyzed: {data_description.get('variables', [])}

ANALYSES:
{json.dumps(analyses, indent=2, default=str)}

Include significance threshold, software used (Python with scipy, statsmodels),
and how assumptions were checked."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def _summarize_analysis_for_report(self, analysis: Dict) -> Dict:
        """Summarize an analysis result for report generation (reduce token count)."""
        summary = {
            "test_name": analysis.get("test_name", ""),
            "rationale": analysis.get("rationale", ""),
            "analysis_type": analysis.get("analysis_type", "Unknown"),
        }

        results = analysis.get("results", {})

        # Extract key statistics based on analysis type
        analysis_type = analysis.get("analysis_type", "").lower()

        if "table1" in analysis_type or "table_1" in analysis_type:
            # For Table 1, just note it exists - don't include full table
            summary["description"] = "Baseline characteristics table comparing groups"
            summary["n_observations"] = results.get("n_observations", "")

        elif "chi_square" in analysis_type or "chi" in analysis_type:
            summary["statistic"] = results.get("statistic")
            summary["p_value"] = results.get("p_value")
            summary["df"] = results.get("df") or results.get("degrees_of_freedom")
            summary["cramers_v"] = results.get("cramers_v") or results.get("effect_size")

        elif "logistic" in analysis_type:
            summary["n_observations"] = results.get("n_observations")
            summary["pseudo_r2"] = results.get("pseudo_r2")
            summary["auc"] = results.get("auc")
            # Only include significant coefficients
            coefficients = results.get("coefficients", {})
            sig_coefs = {}
            for var, stats in coefficients.items():
                if var.lower() not in ["intercept", "const"]:
                    sig_coefs[var] = {
                        "odds_ratio": stats.get("odds_ratio"),
                        "or_ci_lower": stats.get("or_ci_lower"),
                        "or_ci_upper": stats.get("or_ci_upper"),
                        "p_value": stats.get("p_value"),
                    }
            summary["coefficients"] = sig_coefs

        elif "cox" in analysis_type:
            summary["n_observations"] = results.get("n_observations")
            summary["n_events"] = results.get("n_events")
            # Only include coefficients summary
            coefficients = results.get("coefficients", {})
            sig_coefs = {}
            for var, stats in coefficients.items():
                sig_coefs[var] = {
                    "hazard_ratio": stats.get("hazard_ratio"),
                    "hr_ci_lower": stats.get("hr_ci_lower"),
                    "hr_ci_upper": stats.get("hr_ci_upper"),
                    "p_value": stats.get("p_value"),
                }
            summary["coefficients"] = sig_coefs

        elif "kaplan" in analysis_type or "km" in analysis_type:
            summary["p_value"] = results.get("p_value")
            summary["log_rank_statistic"] = results.get("log_rank", {}).get("statistic") if isinstance(results.get("log_rank"), dict) else results.get("test_statistic")
            summary["has_figure"] = True

        elif "anova" in analysis_type or "kruskal" in analysis_type:
            summary["statistic"] = results.get("statistic")
            summary["p_value"] = results.get("p_value")
            summary["eta_squared"] = results.get("eta_squared")

        elif "ttest" in analysis_type or "t_test" in analysis_type:
            summary["statistic"] = results.get("statistic")
            summary["p_value"] = results.get("p_value")
            summary["cohens_d"] = results.get("cohens_d") or results.get("effect_size")

        elif "linear" in analysis_type and "regression" in analysis_type:
            summary["n_observations"] = results.get("n_observations")
            summary["r_squared"] = results.get("r_squared")
            summary["adj_r_squared"] = results.get("adj_r_squared")
            summary["f_statistic"] = results.get("f_statistic")
            summary["f_pvalue"] = results.get("f_pvalue")

        else:
            # Generic - include p_value and statistic if available
            summary["statistic"] = results.get("statistic")
            summary["p_value"] = results.get("p_value")

        # Include interpretation if available (already summarized)
        if analysis.get("interpretation"):
            # Truncate interpretation to first 500 chars
            interp = analysis.get("interpretation", "")
            summary["interpretation"] = interp[:500] + "..." if len(interp) > 500 else interp

        return summary

    def generate_results_section(self, research_question: str,
                                  analyses: List[Dict],
                                  data_description: Dict) -> str:
        """Generate publication-ready Results section for a research paper.

        Args:
            research_question: The research question being addressed
            analyses: List of analysis results with interpretations
            data_description: Description of the dataset

        Returns:
            Formatted Results section text in markdown
        """
        system_prompt = """You are an expert medical writer generating the Results section
for a peer-reviewed publication. Follow these guidelines:

1. Use past tense throughout
2. Report exact statistics (test statistic, degrees of freedom, p-values, confidence intervals)
3. Include effect sizes where appropriate (odds ratios, hazard ratios, Cohen's d, etc.)
4. Present results in logical order: baseline characteristics → primary outcomes → secondary outcomes
5. Be objective - report what was found without interpretation of clinical implications (save for Discussion)
6. Follow STROBE/CONSORT guidelines for reporting
7. Use proper statistical notation (e.g., χ² not chi-square, p < 0.001 not p = 0.000)
8. Include sample sizes for each analysis
9. Format tables references as "Table X" and figures as "Figure Y"
10. Report both significant and non-significant findings

Structure the Results section with clear headings:
- Study Population / Baseline Characteristics
- Primary Outcome(s)
- Secondary Outcome(s) (if applicable)
- Subgroup/Sensitivity Analyses (if applicable)"""

        # Format analyses for the prompt - SUMMARIZED to reduce token count
        analyses_summary = []
        for i, analysis in enumerate(analyses):
            if analysis.get("success"):
                # Use summarized version to reduce tokens
                summary = self._summarize_analysis_for_report(analysis)
                analyses_summary.append(summary)

        user_prompt = f"""Generate a publication-quality Results section for this study:

RESEARCH QUESTION:
{research_question}

DATASET:
- Total sample size: {data_description.get('n_rows', 'Unknown')} subjects
- Variables: {data_description.get('n_columns', 'Unknown')} variables analyzed

ANALYSES PERFORMED AND RESULTS:
{json.dumps(analyses_summary, indent=2, default=str)}

Write a comprehensive Results section that:
1. Starts with a brief description of the study population
2. Reports each analysis with proper statistical notation
3. Includes effect sizes and confidence intervals
4. Notes any missing data or exclusions
5. References tables and figures where analyses produced visualizations

Format the output in markdown with appropriate headings (##, ###).
Include specific numbers from the results - do not use placeholders."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def _get_analysis_system_prompt(self) -> str:
        return """You are an expert biostatistician helping researchers analyze medical data.
Generate a COMPREHENSIVE statistical analysis plan for publication-quality research.

## REQUIRED ANALYSES (always include these for comparative studies):

1. **TABLE 1: Baseline Characteristics**
   - Compare demographics by group
   - Include all relevant continuous and categorical variables
   - Show p-values for group differences
   - api_call: {"analysis_type": "table1", "parameters": {"group_col": "...", "continuous_vars": [...], "categorical_vars": [...]}}

2. **UNADJUSTED COMPARISONS** (primary analysis)
   - Chi-square/Fisher's for categorical outcomes
   - T-tests/Mann-Whitney for continuous outcomes (2 groups)
   - ANOVA/Kruskal-Wallis for continuous outcomes (3+ groups)
   - Log-rank test for survival outcomes

3. **ADJUSTED ANALYSES** (CRITICAL FOR PUBLICATION)
   - Logistic regression for binary outcomes WITH covariates (age, sex, comorbidities)
   - Linear regression for continuous outcomes WITH covariates
   - Cox proportional hazards for survival outcomes WITH covariates
   - api_call: {"analysis_type": "logistic_regression", "parameters": {"outcome": "...", "predictors": ["group_var", "age", "sex", ...]}}
   - api_call: {"analysis_type": "cox_regression", "parameters": {"time": "...", "event": "...", "covariates": ["group_var", "age", "sex", ...]}}

4. **SURVIVAL ANALYSIS** (if time-to-event data available)
   - Kaplan-Meier curves by group
   - Median survival with 95% CI
   - Log-rank test
   - api_call: {"analysis_type": "kaplan_meier", "parameters": {"time": "...", "event": "...", "group": "..."}}

5. **VISUALIZATIONS**
   - Kaplan-Meier curves with at-risk table
   - Forest plot for adjusted effects
   - Box plots for continuous outcomes by group

## VARIABLE MAPPING REQUIREMENTS (CRITICAL):
- Use EXACT column names from the dataset provided
- Map each analysis parameter to a specific column that EXISTS in the data
- Include all clinically relevant covariates for adjustment (age, sex, BMI, comorbidities, etc.)
- Do NOT guess column names - use only what is provided

## OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no explanation text):
{
    "research_type": "COMPARISON|ASSOCIATION|PREDICTION|DIAGNOSTIC|SURVIVAL",
    "primary_analyses": [
        {
            "test_name": "Descriptive Statistics (Table 1)",
            "category": "descriptive",
            "priority": 1,
            "rationale": "Characterize baseline differences between groups",
            "assumptions": [],
            "variables": {"group": "exact_column_name", "continuous": ["col1", "col2"], "categorical": ["col3"]},
            "api_call": {"analysis_type": "table1", "parameters": {"group_col": "exact_column_name", "continuous_vars": ["col1", "col2"], "categorical_vars": ["col3"]}}
        },
        {
            "test_name": "Chi-Square Test",
            "category": "comparative",
            "priority": 2,
            "rationale": "Test unadjusted association",
            "assumptions": ["expected_counts >= 5"],
            "variables": {"outcome": "exact_outcome_col", "group": "exact_group_col"},
            "api_call": {"analysis_type": "chi_square", "parameters": {"row_var": "exact_outcome_col", "col_var": "exact_group_col"}}
        },
        {
            "test_name": "Adjusted Logistic Regression",
            "category": "regression",
            "priority": 3,
            "rationale": "Test association adjusting for confounders",
            "assumptions": ["binary_outcome", "no_multicollinearity"],
            "variables": {"outcome": "exact_outcome_col", "predictors": ["group_col", "age_col", "sex_col"]},
            "api_call": {"analysis_type": "logistic_regression", "parameters": {"outcome": "exact_outcome_col", "predictors": ["group_col", "age_col", "sex_col"]}}
        }
    ],
    "secondary_analyses": [],
    "assumption_checks": ["normality", "homogeneity", "proportional_hazards"],
    "visualizations": ["boxplot", "forest_plot", "kaplan_meier"],
    "notes": ["Important considerations for interpretation"]
}"""

    def _build_analysis_prompt(self, question: str, profile: Dict) -> str:
        # Build variable summary - include ALL variables for complete picture
        variables = profile.get("variables", [])

        # Separate variables by type for better organization
        continuous_vars = []
        categorical_vars = []
        binary_vars = []
        other_vars = []

        for var in variables:
            var_name = var.get('name', '')
            stat_type = var.get('statistical_type', 'unknown')

            if stat_type == 'continuous':
                mean_str = f" (mean={var['mean']:.2f})" if var.get('mean') is not None else ""
                continuous_vars.append(f"{var_name}{mean_str}")
            elif stat_type == 'binary':
                binary_vars.append(var_name)
            elif stat_type == 'categorical':
                cats = var.get('categories', [])
                cat_str = f" ({len(cats)} categories)" if cats else ""
                categorical_vars.append(f"{var_name}{cat_str}")
            else:
                other_vars.append(var_name)

        # Get potential outcomes and groups
        potential_outcomes = profile.get('potential_outcome_columns', [])
        potential_groups = profile.get('potential_group_columns', [])

        return f"""RESEARCH QUESTION:
{question}

DATASET:
- Observations: {profile.get('n_rows', 0)}
- Total variables: {len(variables)}
- Continuous variables: {profile.get('n_continuous', 0)}
- Categorical variables: {profile.get('n_categorical', 0)}
- Binary variables: {profile.get('n_binary', 0)}

CONTINUOUS VARIABLES:
{', '.join(continuous_vars)}

BINARY VARIABLES:
{', '.join(binary_vars)}

CATEGORICAL VARIABLES:
{', '.join(categorical_vars)}

POTENTIAL OUTCOME VARIABLES: {', '.join(potential_outcomes)}
POTENTIAL GROUPING VARIABLES: {', '.join(potential_groups)}

Based on the research question and available variables, recommend appropriate statistical analyses.
Use the EXACT variable names from above in your api_call parameters."""

    def _parse_analysis_response(self, response: str, question: str) -> Dict[str, Any]:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                data["research_question"] = question
                return {"success": True, "plan": data}
        except:
            pass

        return {
            "success": False,
            "error": "Could not parse analysis plan",
            "raw_response": response
        }


# Global LLM service instance
llm_service = LLMService()
