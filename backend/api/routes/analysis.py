"""Statistical analysis endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import math

from services import session_manager, analysis_service
from models import AnalysisRequest, AnalysisResponse


def sanitize_for_json(obj):
    """Recursively sanitize an object for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, bool):
        return obj
    else:
        return obj

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/run")
async def run_analysis(request: AnalysisRequest):
    """Run a statistical analysis.

    Available analysis types:
    - descriptive_continuous: Descriptive stats for continuous variable
    - descriptive_categorical: Descriptive stats for categorical variable
    - table1: Generate Table 1 (baseline characteristics)
    - independent_ttest: Independent samples t-test
    - paired_ttest: Paired samples t-test
    - one_way_anova: One-way ANOVA
    - mann_whitney: Mann-Whitney U test
    - kruskal_wallis: Kruskal-Wallis test
    - chi_square: Chi-square test of independence
    - fisher_exact: Fisher's exact test
    - pearson_correlation: Pearson correlation
    - spearman_correlation: Spearman correlation
    - correlation_matrix: Correlation matrix
    - linear_regression: Linear regression
    - logistic_regression: Logistic regression
    - kaplan_meier: Kaplan-Meier survival analysis
    - log_rank: Log-rank test
    - cox_regression: Cox proportional hazards
    - roc_analysis: ROC curve analysis
    - normality_test: Test for normality
    - power_analysis: Power/sample size calculation
    """
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded in session")

    # Run analysis
    result = analysis_service.run_analysis(
        session.data,
        request.analysis_type,
        request.parameters
    )

    # Sanitize results for JSON
    result = sanitize_for_json(result)

    # Store in session
    if result.get("success"):
        session_manager.add_analysis(request.session_id, result)

    return JSONResponse(content=result)


@router.get("/types")
async def get_analysis_types():
    """Get available analysis types with descriptions."""
    return JSONResponse(content={
        "analysis_types": {
            "descriptive": {
                "descriptive_continuous": {
                    "name": "Descriptive Statistics (Continuous)",
                    "description": "Mean, SD, median, IQR, range for continuous variables",
                    "required_params": ["variable"],
                    "variable_types": ["continuous"]
                },
                "descriptive_categorical": {
                    "name": "Descriptive Statistics (Categorical)",
                    "description": "Frequencies and percentages for categorical variables",
                    "required_params": ["variable"],
                    "variable_types": ["categorical", "binary"]
                },
                "table1": {
                    "name": "Table 1 (Baseline Characteristics)",
                    "description": "Summary table by group",
                    "required_params": ["continuous_vars", "categorical_vars"],
                    "optional_params": ["group_col"]
                }
            },
            "comparative": {
                "independent_ttest": {
                    "name": "Independent Samples t-test",
                    "description": "Compare means between two independent groups",
                    "required_params": ["outcome", "group"],
                    "assumptions": ["normality", "equal_variance"]
                },
                "paired_ttest": {
                    "name": "Paired Samples t-test",
                    "description": "Compare means from paired/matched samples",
                    "required_params": ["before", "after"],
                    "assumptions": ["normality of differences"]
                },
                "one_way_anova": {
                    "name": "One-way ANOVA",
                    "description": "Compare means across 3+ groups",
                    "required_params": ["outcome", "group"],
                    "assumptions": ["normality", "homogeneity of variance"]
                },
                "mann_whitney": {
                    "name": "Mann-Whitney U Test",
                    "description": "Non-parametric comparison of two groups",
                    "required_params": ["outcome", "group"]
                },
                "kruskal_wallis": {
                    "name": "Kruskal-Wallis Test",
                    "description": "Non-parametric comparison of 3+ groups",
                    "required_params": ["outcome", "group"]
                },
                "chi_square": {
                    "name": "Chi-square Test",
                    "description": "Test association between categorical variables",
                    "required_params": ["row_var", "col_var"]
                },
                "fisher_exact": {
                    "name": "Fisher's Exact Test",
                    "description": "Exact test for 2x2 tables",
                    "required_params": ["row_var", "col_var"]
                }
            },
            "correlation": {
                "pearson_correlation": {
                    "name": "Pearson Correlation",
                    "description": "Linear correlation between two continuous variables",
                    "required_params": ["var1", "var2"]
                },
                "spearman_correlation": {
                    "name": "Spearman Correlation",
                    "description": "Rank correlation (non-parametric)",
                    "required_params": ["var1", "var2"]
                },
                "correlation_matrix": {
                    "name": "Correlation Matrix",
                    "description": "Correlations between multiple variables",
                    "optional_params": ["variables", "method"]
                }
            },
            "regression": {
                "linear_regression": {
                    "name": "Linear Regression",
                    "description": "Predict continuous outcome from predictors",
                    "required_params": ["outcome", "predictors"]
                },
                "logistic_regression": {
                    "name": "Logistic Regression",
                    "description": "Predict binary outcome from predictors",
                    "required_params": ["outcome", "predictors"]
                }
            },
            "survival": {
                "kaplan_meier": {
                    "name": "Kaplan-Meier Analysis",
                    "description": "Survival curve estimation",
                    "required_params": ["time", "event"],
                    "optional_params": ["group"]
                },
                "log_rank": {
                    "name": "Log-rank Test",
                    "description": "Compare survival curves between groups",
                    "required_params": ["time", "event", "group"]
                },
                "cox_regression": {
                    "name": "Cox Regression",
                    "description": "Survival analysis with covariates",
                    "required_params": ["time", "event", "covariates"]
                }
            },
            "diagnostic": {
                "roc_analysis": {
                    "name": "ROC Analysis",
                    "description": "Evaluate diagnostic test performance",
                    "required_params": ["true", "scores"]
                }
            },
            "assumptions": {
                "normality_test": {
                    "name": "Normality Test",
                    "description": "Test if variable follows normal distribution",
                    "required_params": ["variable"],
                    "optional_params": ["method"]
                }
            },
            "power": {
                "power_analysis": {
                    "name": "Power Analysis",
                    "description": "Calculate power or required sample size",
                    "required_params": ["test_type"],
                    "optional_params": ["effect_size", "n", "power"]
                }
            }
        }
    })


@router.get("/history/{session_id}")
async def get_analysis_history(session_id: str):
    """Get analysis history for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse(content=sanitize_for_json({
        "success": True,
        "analyses": session.analyses
    }))


@router.post("/interpret")
async def interpret_results(
    session_id: str,
    analysis_id: str
):
    """Get LLM interpretation of analysis results."""
    from services import llm_service

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find analysis
    analysis = None
    for a in session.analyses:
        if a.get("analysis_id") == analysis_id:
            analysis = a
            break

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    interpretation = llm_service.interpret_results(
        analysis.get("analysis_type", ""),
        analysis.get("results", {}),
        {"filename": session.filename, "n_rows": len(session.data)}
    )

    return JSONResponse(content={
        "success": True,
        "interpretation": interpretation
    })


@router.post("/survival/stratified-cox")
async def run_stratified_cox(request: Dict[str, Any]):
    """Run stratified Cox proportional hazards regression.

    Request body:
    {
        "session_id": "uuid",
        "time": "time_column",
        "event": "event_column",
        "covariates": ["var1", "var2"],
        "strata": "stratification_variable"
    }
    """
    session = session_manager.get_session(request.get("session_id"))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    result = analysis_service.run_analysis(
        session.data,
        "stratified_cox",
        request
    )

    result = sanitize_for_json(result)
    if result.get("success"):
        result["test_name"] = f"Stratified Cox Regression (by {request.get('strata')})"
        session_manager.add_analysis(request.get("session_id"), result)

    return JSONResponse(content=result)


@router.post("/survival/rmst-comparison")
async def run_rmst_comparison(request: Dict[str, Any]):
    """Compare restricted mean survival times between groups.

    Request body:
    {
        "session_id": "uuid",
        "time": "time_column",
        "event": "event_column",
        "group": "group_column",
        "tau": 365  # Optional restriction time
    }
    """
    session = session_manager.get_session(request.get("session_id"))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    result = analysis_service.run_analysis(
        session.data,
        "rmst_comparison",
        request
    )

    result = sanitize_for_json(result)
    if result.get("success"):
        result["test_name"] = "RMST Comparison"
        session_manager.add_analysis(request.get("session_id"), result)

    return JSONResponse(content=result)


@router.post("/survival/landmark")
async def run_landmark_analysis(request: Dict[str, Any]):
    """Perform landmark survival analysis.

    Request body:
    {
        "session_id": "uuid",
        "time": "time_column",
        "event": "event_column",
        "group": "group_column",
        "landmark_times": [30, 90, 180]
    }
    """
    session = session_manager.get_session(request.get("session_id"))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    result = analysis_service.run_analysis(
        session.data,
        "landmark_analysis",
        request
    )

    result = sanitize_for_json(result)
    if result.get("success"):
        result["test_name"] = "Landmark Analysis"
        session_manager.add_analysis(request.get("session_id"), result)

    return JSONResponse(content=result)


@router.post("/survival/competing-risks")
async def run_competing_risks(request: Dict[str, Any]):
    """Perform competing risks analysis.

    Request body:
    {
        "session_id": "uuid",
        "time": "time_column",
        "event": "event_column",  # Should have multiple event types
        "event_of_interest": 1  # The event type to analyze
    }
    """
    session = session_manager.get_session(request.get("session_id"))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    result = analysis_service.run_analysis(
        session.data,
        "competing_risks",
        request
    )

    result = sanitize_for_json(result)
    if result.get("success"):
        result["test_name"] = "Competing Risks Analysis"
        session_manager.add_analysis(request.get("session_id"), result)

    return JSONResponse(content=result)


@router.post("/survival/ph-test")
async def run_proportional_hazards_test(request: Dict[str, Any]):
    """Test proportional hazards assumption.

    Request body:
    {
        "session_id": "uuid",
        "time": "time_column",
        "event": "event_column",
        "covariates": ["var1", "var2"]
    }
    """
    session = session_manager.get_session(request.get("session_id"))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    result = analysis_service.run_analysis(
        session.data,
        "ph_test",
        request
    )

    result = sanitize_for_json(result)
    if result.get("success"):
        result["test_name"] = "Proportional Hazards Test"
        session_manager.add_analysis(request.get("session_id"), result)

    return JSONResponse(content=result)
