"""Code Generation Module for Statistical Analysis.

Generates reproducible code in Python and R for all statistical analyses.
Supports standalone scripts, Jupyter notebooks, and R Markdown documents.

Features:
- 50+ statistical methods supported
- Both Python and R code generation
- Jupyter notebook and R Markdown output
- Complete documentation and comments
- Package installation instructions
- Reproducibility with seeds and hashes

Version: 1.0.0
"""

from .code_generator import (
    CodeGenerator,
    GeneratedCode,
    CodeFormat,
    AnalysisCodeBundle
)
from .python_generator import PythonCodeGenerator
from .r_generator import RCodeGenerator
from .notebook_generator import NotebookGenerator
from .code_validator import CodeValidator
from .integration import (
    CodeGenerationIntegrator,
    attach_code_to_results
)

__all__ = [
    'CodeGenerator',
    'GeneratedCode',
    'CodeFormat',
    'AnalysisCodeBundle',
    'PythonCodeGenerator',
    'RCodeGenerator',
    'NotebookGenerator',
    'CodeValidator',
    'CodeGenerationIntegrator',
    'attach_code_to_results'
]

__version__ = '1.0.0'
