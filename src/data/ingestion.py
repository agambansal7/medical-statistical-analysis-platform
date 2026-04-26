"""Data ingestion module for loading various data formats."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import io


class DataIngestion:
    """Handle data loading from various sources and formats."""

    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
    MISSING_INDICATORS = ['', 'NA', 'N/A', 'null', 'NULL', '.', '-',
                          'NaN', 'nan', '#N/A', '#NA', 'None', 'none',
                          '-999', '999', '-99', '99']

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.load_warnings: List[str] = []

    def load_file(self, file_path: Union[str, Path],
                  sheet_name: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """Load data from file.

        Args:
            file_path: Path to data file
            sheet_name: Sheet name for Excel files
            **kwargs: Additional arguments for pandas read functions

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}. "
                           f"Supported: {self.SUPPORTED_FORMATS}")

        # Set common parameters
        read_kwargs = {
            'na_values': self.MISSING_INDICATORS,
            **kwargs
        }

        if suffix == '.csv':
            self.data = self._load_csv(file_path, **read_kwargs)
        elif suffix in ['.xlsx', '.xls']:
            self.data = self._load_excel(file_path, sheet_name, **read_kwargs)

        self._update_metadata(file_path)
        self._clean_data()

        return self.data

    def load_from_bytes(self, file_bytes: bytes, filename: str,
                        sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load data from bytes (for web upload).

        Args:
            file_bytes: File content as bytes
            filename: Original filename (for format detection)
            sheet_name: Sheet name for Excel files

        Returns:
            Loaded DataFrame
        """
        suffix = Path(filename).suffix.lower()

        read_kwargs = {'na_values': self.MISSING_INDICATORS}

        if suffix == '.csv':
            self.data = pd.read_csv(io.BytesIO(file_bytes), **read_kwargs)
        elif suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(io.BytesIO(file_bytes),
                                      sheet_name=sheet_name or 0,
                                      **read_kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        self._update_metadata(filename)
        self._clean_data()

        return self.data

    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding, **kwargs)
            except UnicodeDecodeError:
                continue

        # Fallback with error handling
        return pd.read_csv(file_path, encoding='utf-8',
                          errors='replace', **kwargs)

    def _load_excel(self, file_path: Path,
                    sheet_name: Optional[str] = None,
                    **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        # Remove na_values from kwargs for Excel (handled differently)
        kwargs.pop('na_values', None)

        if sheet_name is None:
            # Get first sheet
            xl = pd.ExcelFile(file_path)
            self.metadata['available_sheets'] = xl.sheet_names
            sheet_name = xl.sheet_names[0]
            self.load_warnings.append(
                f"No sheet specified. Using first sheet: '{sheet_name}'"
            )

        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    def _update_metadata(self, source: Union[Path, str]):
        """Update metadata about loaded data."""
        self.metadata.update({
            'source': str(source),
            'n_rows': len(self.data),
            'n_columns': len(self.data.columns),
            'columns': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in
                      self.data.dtypes.items()},
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        })

    def _clean_data(self):
        """Clean loaded data."""
        # Strip whitespace from string columns
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

        # Clean column names
        self.data.columns = [self._clean_column_name(col)
                            for col in self.data.columns]

        # Try to convert object columns to appropriate types
        for col in self.data.columns:
            self.data[col] = self._infer_and_convert(self.data[col])

    def _clean_column_name(self, name: str) -> str:
        """Clean column name."""
        if not isinstance(name, str):
            name = str(name)

        # Replace problematic characters
        name = name.strip()
        name = name.replace('\n', ' ')
        name = name.replace('\t', ' ')

        # Collapse multiple spaces
        while '  ' in name:
            name = name.replace('  ', ' ')

        return name

    def _infer_and_convert(self, series: pd.Series) -> pd.Series:
        """Attempt to convert series to appropriate type."""
        if series.dtype == 'object':
            # Try numeric conversion
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                if numeric.notna().sum() > series.notna().sum() * 0.5:
                    return numeric
            except:
                pass

            # Try datetime conversion
            try:
                datetime = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                if datetime.notna().sum() > series.notna().sum() * 0.5:
                    return datetime
            except:
                pass

        return series

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if self.data is None:
            return {'error': 'No data loaded'}

        return {
            'metadata': self.metadata,
            'warnings': self.load_warnings,
            'missing_summary': self._missing_summary(),
            'sample': self.data.head().to_dict()
        }

    def _missing_summary(self) -> Dict[str, Any]:
        """Get missing data summary."""
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100

        return {
            'total_missing': int(missing.sum()),
            'columns_with_missing': int((missing > 0).sum()),
            'by_column': {
                col: {
                    'count': int(missing[col]),
                    'percentage': round(missing_pct[col], 2)
                }
                for col in self.data.columns if missing[col] > 0
            }
        }

    def validate_required_columns(self, required: List[str]) -> List[str]:
        """Check if required columns exist.

        Args:
            required: List of required column names

        Returns:
            List of missing columns
        """
        if self.data is None:
            return required

        existing = set(self.data.columns)
        return [col for col in required if col not in existing]

    def get_column_values(self, column: str) -> List[Any]:
        """Get unique values for a column."""
        if self.data is None or column not in self.data.columns:
            return []
        return self.data[column].dropna().unique().tolist()
