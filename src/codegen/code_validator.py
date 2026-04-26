"""Code Validator for Generated Statistical Code."""

from typing import Dict, Any, List, Tuple, Optional
import ast
import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    language: str
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'language': self.language,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions
        }


class CodeValidator:
    """Validate generated statistical code."""

    def validate(self, code: str, language: str) -> ValidationResult:
        """Validate code syntax and common issues."""
        if language.lower() == 'python':
            return self._validate_python(code)
        elif language.lower() == 'r':
            return self._validate_r(code)
        else:
            return ValidationResult(
                is_valid=False,
                language=language,
                errors=[f"Unknown language: {language}"],
                warnings=[],
                suggestions=[]
            )

    def _validate_python(self, code: str) -> ValidationResult:
        """Validate Python code."""
        errors = []
        warnings = []
        suggestions = []

        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

        # Check for common issues
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for hardcoded paths
            if re.search(r'["\'][A-Za-z]:\\|/Users/|/home/', line):
                warnings.append(f"Line {i}: Hardcoded file path detected")

            # Check for missing imports
            if 'pd.' in line and 'import pandas' not in code:
                warnings.append(f"Line {i}: Using pandas but import not found")
            if 'np.' in line and 'import numpy' not in code:
                warnings.append(f"Line {i}: Using numpy but import not found")

        # Suggestions
        if 'random_state' not in code and 'seed' not in code.lower():
            suggestions.append("Consider setting random_state for reproducibility")

        if 'plt.savefig' not in code and 'plt.show' in code:
            suggestions.append("Consider saving figures with plt.savefig()")

        return ValidationResult(
            is_valid=len(errors) == 0,
            language='python',
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _validate_r(self, code: str) -> ValidationResult:
        """Validate R code (basic checks)."""
        errors = []
        warnings = []
        suggestions = []

        lines = code.split('\n')

        # Check for basic R syntax issues
        open_parens = 0
        open_braces = 0

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue

            open_parens += line.count('(') - line.count(')')
            open_braces += line.count('{') - line.count('}')

            # Check for common issues
            if re.search(r'["\'][A-Za-z]:\\|/Users/|/home/', line):
                warnings.append(f"Line {i}: Hardcoded file path detected")

        if open_parens != 0:
            errors.append("Unbalanced parentheses")
        if open_braces != 0:
            errors.append("Unbalanced braces")

        # Suggestions
        if 'set.seed' not in code:
            suggestions.append("Consider adding set.seed() for reproducibility")

        return ValidationResult(
            is_valid=len(errors) == 0,
            language='r',
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
