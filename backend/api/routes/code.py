"""Code Generation API Routes.

Provides endpoints for generating reproducible Python and R code.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Add src to path for codegen imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from services import session_manager

router = APIRouter(prefix="/code", tags=["code"])


class CodeGenerationRequest(BaseModel):
    session_id: str
    analysis_type: str
    parameters: Dict[str, Any] = {}
    languages: List[str] = ["python", "r"]


class CodeGenerationResponse(BaseModel):
    success: bool
    python_code: Optional[str] = None
    r_code: Optional[str] = None
    python_packages: Optional[List[str]] = None
    r_packages: Optional[List[str]] = None
    methodology: Optional[str] = None
    error: Optional[str] = None


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    """Generate reproducible code for a statistical analysis."""
    try:
        # Import the code generator
        from codegen import CodeGenerator

        generator = CodeGenerator(default_seed=42)

        # Get data info from session if available
        data_info = None
        session = session_manager.get_session(request.session_id)
        if session and session.data is not None:
            data_info = {
                'n_rows': len(session.data),
                'columns': list(session.data.columns),
            }

        # Generate code
        bundle = generator.generate(
            analysis_type=request.analysis_type,
            parameters=request.parameters,
            data_info=data_info,
            languages=request.languages,
            seed=42
        )

        return CodeGenerationResponse(
            success=True,
            python_code=bundle.python_code.code if bundle.python_code else None,
            r_code=bundle.r_code.code if bundle.r_code else None,
            python_packages=bundle.python_code.packages_required if bundle.python_code else None,
            r_packages=bundle.r_code.packages_required if bundle.r_code else None,
            methodology=bundle.methodology_notes
        )

    except ImportError:
        # Fallback if codegen module not available
        return CodeGenerationResponse(
            success=True,
            python_code=_generate_fallback_python(request.analysis_type, request.parameters),
            r_code=_generate_fallback_r(request.analysis_type, request.parameters),
            python_packages=["pandas", "numpy", "scipy", "statsmodels"],
            r_packages=["tidyverse", "broom"],
            methodology="Code generation module not available, using fallback templates."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{session_id}/{analysis_id}")
async def get_code_for_analysis(session_id: str, analysis_id: str):
    """Get generated code for a completed analysis."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the analysis in history
    analysis = None
    for a in session.analysis_history:
        if a.get('analysis_id') == analysis_id:
            analysis = a
            break

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        from codegen import CodeGenerator

        generator = CodeGenerator(default_seed=42)
        bundle = generator.generate(
            analysis_type=analysis.get('analysis_type', 'generic'),
            parameters=analysis.get('parameters', {}),
            languages=['python', 'r']
        )

        return {
            "success": True,
            "python_code": bundle.python_code.code if bundle.python_code else None,
            "r_code": bundle.r_code.code if bundle.r_code else None,
            "python_packages": bundle.python_code.packages_required if bundle.python_code else [],
            "r_packages": bundle.r_code.packages_required if bundle.r_code else [],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _generate_fallback_python(analysis_type: str, parameters: Dict) -> str:
    """Generate fallback Python code when codegen module is unavailable."""
    return f'''"""
Statistical Analysis: {analysis_type.replace('_', ' ').title()}
Generated code for reproducibility
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Set random seed
np.random.seed(42)

# Load your data
# df = pd.read_csv('your_data.csv')

# Analysis: {analysis_type}
# Parameters: {parameters}

# Add your analysis code here based on the type:
# - For t-test: stats.ttest_ind(group1, group2)
# - For ANOVA: stats.f_oneway(*groups)
# - For regression: sm.OLS(y, X).fit()
# - For chi-square: stats.chi2_contingency(table)

print("Analysis complete")
'''


def _generate_fallback_r(analysis_type: str, parameters: Dict) -> str:
    """Generate fallback R code when codegen module is unavailable."""
    return f'''# Statistical Analysis: {analysis_type.replace('_', ' ').title()}
# Generated code for reproducibility

library(tidyverse)
library(broom)

# Set random seed
set.seed(42)

# Load your data
# df <- read_csv("your_data.csv")

# Analysis: {analysis_type}
# Parameters: {parameters}

# Add your analysis code here based on the type:
# - For t-test: t.test(group1, group2)
# - For ANOVA: aov(outcome ~ group, data = df)
# - For regression: lm(y ~ x, data = df)
# - For chi-square: chisq.test(table)

print("Analysis complete")
'''
