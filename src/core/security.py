"""Security and Compliance Module.

Provides:
- HIPAA-ready deployment options
- Data encryption at rest
- PHI detection and masking
- Audit logging for compliance
"""

import pandas as pd
import numpy as np
import re
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


@dataclass
class PHIDetectionResult:
    """Result of PHI detection scan."""
    has_phi: bool
    phi_columns: List[str]
    phi_types: Dict[str, List[str]]  # column -> list of PHI types
    risk_level: str  # high, medium, low
    recommendations: List[str]
    scan_timestamp: str


@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""
    timestamp: str
    action: str
    user: Optional[str]
    resource: str
    result: str  # success, denied, warning
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None


class PHIDetector:
    """Detect Protected Health Information in datasets."""

    # Regex patterns for common PHI types
    PHI_PATTERNS = {
        'ssn': [
            r'\b\d{3}-\d{2}-\d{4}\b',
            r'\b\d{9}\b'
        ],
        'phone': [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'
        ],
        'email': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ],
        'mrn': [
            r'\bMRN[:\s]*\d+\b',
            r'\bMR[:\s]*\d{6,}\b'
        ],
        'date_of_birth': [
            r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b',
            r'\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'
        ],
        'address': [
            r'\b\d+\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b',
        ],
        'zip_code': [
            r'\b\d{5}(-\d{4})?\b'
        ],
        'ip_address': [
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        ],
        'credit_card': [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        ]
    }

    # Column name patterns suggesting PHI
    PHI_COLUMN_PATTERNS = {
        'name': ['name', 'first_name', 'last_name', 'patient_name', 'full_name'],
        'ssn': ['ssn', 'social_security', 'ss_number'],
        'phone': ['phone', 'telephone', 'mobile', 'cell'],
        'email': ['email', 'e_mail', 'email_address'],
        'address': ['address', 'street', 'city', 'state', 'zip', 'postal'],
        'dob': ['dob', 'date_of_birth', 'birth_date', 'birthdate'],
        'mrn': ['mrn', 'medical_record', 'patient_id', 'record_number'],
        'account': ['account', 'account_number', 'member_id']
    }

    def __init__(self):
        self.compiled_patterns = self._compile_patterns()

    def scan(self, data: pd.DataFrame) -> PHIDetectionResult:
        """Scan DataFrame for potential PHI.

        Args:
            data: DataFrame to scan

        Returns:
            PHIDetectionResult with findings
        """
        phi_columns = []
        phi_types: Dict[str, List[str]] = {}

        for col in data.columns:
            detected_types = []

            # Check column name
            name_matches = self._check_column_name(col)
            detected_types.extend(name_matches)

            # Check content for text columns
            if data[col].dtype == 'object':
                content_matches = self._check_column_content(data[col])
                detected_types.extend(content_matches)

            # Check for high-cardinality identifiers
            if self._looks_like_identifier(data[col]):
                detected_types.append('potential_identifier')

            if detected_types:
                phi_columns.append(col)
                phi_types[col] = list(set(detected_types))

        # Determine risk level
        risk_level = self._assess_risk(phi_types)

        # Generate recommendations
        recommendations = self._generate_recommendations(phi_columns, phi_types)

        return PHIDetectionResult(
            has_phi=len(phi_columns) > 0,
            phi_columns=phi_columns,
            phi_types=phi_types,
            risk_level=risk_level,
            recommendations=recommendations,
            scan_timestamp=datetime.now().isoformat()
        )

    def mask(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'hash'
    ) -> pd.DataFrame:
        """Mask PHI columns.

        Args:
            data: DataFrame with PHI
            columns: Specific columns to mask (None = auto-detect)
            method: 'hash', 'redact', 'pseudonymize'

        Returns:
            DataFrame with masked PHI
        """
        masked = data.copy()

        # Auto-detect if no columns specified
        if columns is None:
            scan_result = self.scan(data)
            columns = scan_result.phi_columns

        for col in columns:
            if col not in masked.columns:
                continue

            if method == 'hash':
                masked[col] = masked[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]
                    if pd.notna(x) else x
                )
            elif method == 'redact':
                masked[col] = '[REDACTED]'
            elif method == 'pseudonymize':
                unique_vals = masked[col].unique()
                mapping = {v: f'ID_{i:06d}' for i, v in enumerate(unique_vals) if pd.notna(v)}
                masked[col] = masked[col].map(mapping)

        return masked

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns."""
        compiled = {}
        for phi_type, patterns in self.PHI_PATTERNS.items():
            compiled[phi_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled

    def _check_column_name(self, col_name: str) -> List[str]:
        """Check column name for PHI indicators."""
        detected = []
        col_lower = col_name.lower().replace('_', '').replace('-', '').replace(' ', '')

        for phi_type, patterns in self.PHI_COLUMN_PATTERNS.items():
            for pattern in patterns:
                pattern_clean = pattern.replace('_', '')
                if pattern_clean in col_lower:
                    detected.append(phi_type)
                    break

        return detected

    def _check_column_content(self, series: pd.Series) -> List[str]:
        """Check column content for PHI patterns."""
        detected = []

        # Sample for performance
        sample = series.dropna().head(1000)

        for phi_type, patterns in self.compiled_patterns.items():
            for value in sample:
                if not isinstance(value, str):
                    continue
                for pattern in patterns:
                    if pattern.search(value):
                        detected.append(phi_type)
                        break
                if phi_type in detected:
                    break

        return detected

    def _looks_like_identifier(self, series: pd.Series) -> bool:
        """Check if column looks like a unique identifier."""
        n = len(series)
        n_unique = series.nunique()

        # High cardinality relative to size
        if n_unique / n > 0.9 and n > 50:
            return True

        return False

    def _assess_risk(self, phi_types: Dict[str, List[str]]) -> str:
        """Assess overall risk level."""
        high_risk = {'ssn', 'mrn', 'credit_card', 'name'}
        medium_risk = {'dob', 'phone', 'email', 'address'}

        all_types = set()
        for types in phi_types.values():
            all_types.update(types)

        if all_types & high_risk:
            return 'high'
        elif all_types & medium_risk:
            return 'medium'
        elif all_types:
            return 'low'
        return 'none'

    def _generate_recommendations(
        self,
        phi_columns: List[str],
        phi_types: Dict[str, List[str]]
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        if not phi_columns:
            recommendations.append("No PHI detected. Standard data handling procedures apply.")
            return recommendations

        recommendations.append(f"PHI detected in {len(phi_columns)} column(s). HIPAA compliance required.")

        # Specific recommendations
        all_types = set()
        for types in phi_types.values():
            all_types.update(types)

        if 'ssn' in all_types:
            recommendations.append("CRITICAL: SSN detected. Consider removing or using secure hashing.")

        if 'mrn' in all_types:
            recommendations.append("Medical Record Numbers detected. Apply de-identification before sharing.")

        if 'name' in all_types:
            recommendations.append("Names detected. Remove or pseudonymize for de-identified datasets.")

        recommendations.extend([
            "Ensure data access is logged and audited.",
            "Use encryption for data at rest and in transit.",
            "Implement minimum necessary access controls."
        ])

        return recommendations


class DataEncryption:
    """Encrypt and decrypt sensitive data."""

    def __init__(self, key: Optional[bytes] = None, password: Optional[str] = None):
        """Initialize encryption with key or password.

        Args:
            key: 32-byte encryption key
            password: Password to derive key from
        """
        if key:
            self.key = key
        elif password:
            self.key = self._derive_key(password)
        else:
            self.key = Fernet.generate_key()

        self.fernet = Fernet(self.key)

    def encrypt_value(self, value: Any) -> str:
        """Encrypt a single value."""
        if pd.isna(value):
            return value

        value_bytes = str(value).encode()
        encrypted = self.fernet.encrypt(value_bytes)
        return encrypted.decode()

    def decrypt_value(self, encrypted: str) -> str:
        """Decrypt a single value."""
        if pd.isna(encrypted):
            return encrypted

        encrypted_bytes = encrypted.encode()
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()

    def encrypt_dataframe(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Encrypt columns in DataFrame.

        Args:
            data: DataFrame to encrypt
            columns: Columns to encrypt (None = all object columns)

        Returns:
            DataFrame with encrypted columns
        """
        encrypted = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            if col in encrypted.columns:
                encrypted[col] = encrypted[col].apply(self.encrypt_value)

        return encrypted

    def decrypt_dataframe(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Decrypt columns in DataFrame."""
        decrypted = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            if col in decrypted.columns:
                decrypted[col] = decrypted[col].apply(self.decrypt_value)

        return decrypted

    def save_key(self, path: str):
        """Save encryption key to file."""
        with open(path, 'wb') as f:
            f.write(self.key)

    @classmethod
    def load_key(cls, path: str) -> 'DataEncryption':
        """Load encryption key from file."""
        with open(path, 'rb') as f:
            key = f.read()
        return cls(key=key)

    def _derive_key(self, password: str, salt: bytes = b'static_salt') -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key


class SecurityManager:
    """Unified security management."""

    def __init__(
        self,
        audit_path: Optional[str] = None,
        encryption_key: Optional[bytes] = None
    ):
        self.phi_detector = PHIDetector()
        self.encryption = DataEncryption(key=encryption_key) if encryption_key else None
        self.audit_path = Path(audit_path) if audit_path else Path("./security_audit")
        self.audit_path.mkdir(parents=True, exist_ok=True)
        self._audit_log: List[SecurityAuditEntry] = []

    def check_data(self, data: pd.DataFrame, user: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive security check on data.

        Args:
            data: DataFrame to check
            user: User performing the check

        Returns:
            Security assessment
        """
        # Scan for PHI
        phi_result = self.phi_detector.scan(data)

        # Log the check
        self._log_action(
            action='data_scan',
            user=user,
            resource='dataframe',
            result='completed',
            details={
                'rows': len(data),
                'columns': len(data.columns),
                'phi_detected': phi_result.has_phi,
                'risk_level': phi_result.risk_level
            }
        )

        return {
            'phi_scan': {
                'has_phi': phi_result.has_phi,
                'phi_columns': phi_result.phi_columns,
                'risk_level': phi_result.risk_level,
                'recommendations': phi_result.recommendations
            },
            'compliance_status': 'review_required' if phi_result.has_phi else 'compliant',
            'scan_timestamp': phi_result.scan_timestamp
        }

    def prepare_for_sharing(
        self,
        data: pd.DataFrame,
        mask_phi: bool = True,
        encrypt: bool = False,
        user: Optional[str] = None
    ) -> pd.DataFrame:
        """Prepare data for safe sharing.

        Args:
            data: Original data
            mask_phi: Whether to mask detected PHI
            encrypt: Whether to encrypt the data
            user: User preparing the data

        Returns:
            Safe-to-share DataFrame
        """
        result = data.copy()

        if mask_phi:
            result = self.phi_detector.mask(result, method='hash')
            self._log_action(
                action='phi_masking',
                user=user,
                resource='dataframe',
                result='success',
                details={'method': 'hash'}
            )

        if encrypt and self.encryption:
            result = self.encryption.encrypt_dataframe(result)
            self._log_action(
                action='encryption',
                user=user,
                resource='dataframe',
                result='success'
            )

        return result

    def create_deidentified_dataset(
        self,
        data: pd.DataFrame,
        remove_columns: Optional[List[str]] = None,
        user: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create a de-identified version of the dataset.

        Args:
            data: Original data with potential PHI
            remove_columns: Additional columns to remove
            user: User creating the dataset

        Returns:
            Tuple of (de-identified data, de-identification report)
        """
        # Scan for PHI
        scan = self.phi_detector.scan(data)

        # Columns to remove or mask
        columns_to_remove = set(scan.phi_columns)
        if remove_columns:
            columns_to_remove.update(remove_columns)

        # Create de-identified copy
        deidentified = data.drop(columns=list(columns_to_remove), errors='ignore')

        # Generate report
        report = {
            'original_columns': list(data.columns),
            'removed_columns': list(columns_to_remove),
            'retained_columns': list(deidentified.columns),
            'phi_types_found': scan.phi_types,
            'timestamp': datetime.now().isoformat()
        }

        self._log_action(
            action='deidentification',
            user=user,
            resource='dataframe',
            result='success',
            details=report
        )

        return deidentified, report

    def _log_action(
        self,
        action: str,
        user: Optional[str],
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security action."""
        entry = SecurityAuditEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            user=user,
            resource=resource,
            result=result,
            details=details or {}
        )

        self._audit_log.append(entry)
        self._persist_audit_entry(entry)

    def _persist_audit_entry(self, entry: SecurityAuditEntry):
        """Persist audit entry to file."""
        date = datetime.now().strftime('%Y-%m-%d')
        audit_file = self.audit_path / f"audit_{date}.jsonl"

        with open(audit_file, 'a') as f:
            f.write(json.dumps(entry.__dict__) + '\n')

    def get_audit_log(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        action: Optional[str] = None
    ) -> List[SecurityAuditEntry]:
        """Get filtered audit log."""
        filtered = self._audit_log.copy()

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]
        if action:
            filtered = [e for e in filtered if e.action == action]

        return filtered

    def export_audit_log(self, format: str = 'json') -> str:
        """Export audit log for compliance review."""
        if format == 'json':
            return json.dumps([e.__dict__ for e in self._audit_log], indent=2)
        elif format == 'csv':
            import csv
            import io

            output = io.StringIO()
            if self._audit_log:
                fieldnames = ['timestamp', 'action', 'user', 'resource', 'result']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for entry in self._audit_log:
                    writer.writerow({k: getattr(entry, k) for k in fieldnames})

            return output.getvalue()

        return ""
