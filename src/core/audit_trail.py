"""Audit Trail and Reproducibility Module.

Provides complete analysis tracking:
- Analysis log with timestamps
- Seed tracking for randomized procedures
- One-click reproduction of previous analyses
- Version control integration
"""

import pandas as pd
import numpy as np
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
import uuid
import copy


@dataclass
class AuditEntry:
    """Single audit log entry."""
    id: str
    timestamp: str
    action: str  # load_data, run_analysis, export_results, etc.
    user: Optional[str] = None
    session_id: Optional[str] = None

    # Analysis details
    analysis_type: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_hash: Optional[str] = None  # Hash of input data
    output_hash: Optional[str] = None  # Hash of output

    # Reproducibility
    random_seed: Optional[int] = None
    package_versions: Dict[str, str] = field(default_factory=dict)

    # Status
    status: str = "completed"  # completed, failed, in_progress
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None

    # Result reference
    result_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisSnapshot:
    """Complete snapshot for reproduction."""
    id: str
    timestamp: str
    description: str

    # Data snapshot
    data_hash: str
    data_shape: tuple
    column_names: List[str]

    # Analysis configuration
    analysis_type: str
    parameters: Dict[str, Any]
    random_seed: int

    # Environment
    package_versions: Dict[str, str]
    python_version: str

    # Results reference
    result_hash: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['data_shape'] = list(self.data_shape)
        return result


class AuditTrail:
    """Complete audit trail for all analyses."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        session_id: Optional[str] = None,
        user: Optional[str] = None
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./audit_logs")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.user = user
        self.entries: List[AuditEntry] = []
        self.snapshots: Dict[str, AnalysisSnapshot] = {}

        # Load existing entries for this session
        self._load_session()

    def log(
        self,
        action: str,
        analysis_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        input_data: Optional[pd.DataFrame] = None,
        output_data: Optional[Any] = None,
        random_seed: Optional[int] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> AuditEntry:
        """Log an action to the audit trail.

        Args:
            action: Type of action performed
            analysis_type: Specific analysis type
            parameters: Analysis parameters
            input_data: Input DataFrame (will be hashed)
            output_data: Output data (will be hashed)
            random_seed: Random seed used
            status: Completion status
            error_message: Error message if failed
            duration_ms: Duration in milliseconds

        Returns:
            Created AuditEntry
        """
        entry_id = str(uuid.uuid4())[:12]

        # Hash input data
        input_hash = None
        if input_data is not None:
            input_hash = self._hash_dataframe(input_data)

        # Hash output
        output_hash = None
        if output_data is not None:
            output_hash = self._hash_object(output_data)

        # Get package versions
        versions = self._get_package_versions()

        entry = AuditEntry(
            id=entry_id,
            timestamp=datetime.now().isoformat(),
            action=action,
            user=self.user,
            session_id=self.session_id,
            analysis_type=analysis_type,
            parameters=parameters or {},
            input_hash=input_hash,
            output_hash=output_hash,
            random_seed=random_seed,
            package_versions=versions,
            status=status,
            error_message=error_message,
            duration_ms=duration_ms
        )

        self.entries.append(entry)
        self._save_entry(entry)

        return entry

    def create_snapshot(
        self,
        description: str,
        data: pd.DataFrame,
        analysis_type: str,
        parameters: Dict[str, Any],
        random_seed: int,
        result: Any
    ) -> AnalysisSnapshot:
        """Create a reproducibility snapshot.

        Args:
            description: Human-readable description
            data: Input DataFrame
            analysis_type: Type of analysis
            parameters: Analysis parameters
            random_seed: Random seed used
            result: Analysis result

        Returns:
            AnalysisSnapshot for reproduction
        """
        snapshot_id = str(uuid.uuid4())[:12]

        snapshot = AnalysisSnapshot(
            id=snapshot_id,
            timestamp=datetime.now().isoformat(),
            description=description,
            data_hash=self._hash_dataframe(data),
            data_shape=data.shape,
            column_names=list(data.columns),
            analysis_type=analysis_type,
            parameters=copy.deepcopy(parameters),
            random_seed=random_seed,
            package_versions=self._get_package_versions(),
            python_version=self._get_python_version(),
            result_hash=self._hash_object(result)
        )

        self.snapshots[snapshot_id] = snapshot
        self._save_snapshot(snapshot)

        return snapshot

    def get_history(
        self,
        action: Optional[str] = None,
        analysis_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Get filtered audit history.

        Args:
            action: Filter by action type
            analysis_type: Filter by analysis type
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        filtered = self.entries.copy()

        if action:
            filtered = [e for e in filtered if e.action == action]

        if analysis_type:
            filtered = [e for e in filtered if e.analysis_type == analysis_type]

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered[:limit]

    def export_log(self, format: str = "json") -> str:
        """Export audit log.

        Args:
            format: Export format ("json", "csv", "markdown")

        Returns:
            Formatted audit log string
        """
        if format == "json":
            return json.dumps([e.to_dict() for e in self.entries], indent=2)

        elif format == "csv":
            if not self.entries:
                return ""

            # Create CSV
            import csv
            import io

            output = io.StringIO()
            fieldnames = ['timestamp', 'action', 'analysis_type', 'status', 'user', 'session_id']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for entry in self.entries:
                row = {k: getattr(entry, k, '') for k in fieldnames}
                writer.writerow(row)

            return output.getvalue()

        elif format == "markdown":
            lines = ["# Audit Trail", "", f"Session: {self.session_id}", ""]
            lines.append("| Timestamp | Action | Analysis | Status |")
            lines.append("|-----------|--------|----------|--------|")

            for entry in self.entries[-50:]:  # Last 50
                lines.append(
                    f"| {entry.timestamp[:19]} | {entry.action} | "
                    f"{entry.analysis_type or '-'} | {entry.status} |"
                )

            return '\n'.join(lines)

        return ""

    def verify_reproducibility(
        self,
        snapshot_id: str,
        new_result: Any
    ) -> Dict[str, Any]:
        """Verify that a new result matches a snapshot.

        Args:
            snapshot_id: ID of snapshot to verify against
            new_result: New result to compare

        Returns:
            Verification result with match status
        """
        snapshot = self.snapshots.get(snapshot_id)
        if not snapshot:
            return {'verified': False, 'error': 'Snapshot not found'}

        new_hash = self._hash_object(new_result)
        matches = new_hash == snapshot.result_hash

        return {
            'verified': matches,
            'snapshot_id': snapshot_id,
            'original_hash': snapshot.result_hash,
            'new_hash': new_hash,
            'timestamp': snapshot.timestamp
        }

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Create stable hash of DataFrame."""
        # Sort columns for stability
        df_sorted = df.reindex(sorted(df.columns), axis=1)

        # Create hash from values and column names
        hash_input = str(list(df_sorted.columns)) + df_sorted.to_csv(index=False)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _hash_object(self, obj: Any) -> str:
        """Create hash of arbitrary object."""
        try:
            # Try JSON serialization first
            if isinstance(obj, dict):
                hash_input = json.dumps(obj, sort_keys=True, default=str)
            else:
                hash_input = pickle.dumps(obj)
                return hashlib.sha256(hash_input).hexdigest()[:16]

            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        except:
            return "unhashable"

    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        versions = {}

        packages = ['pandas', 'numpy', 'scipy', 'statsmodels', 'sklearn']
        for pkg in packages:
            try:
                import importlib
                module = importlib.import_module(pkg)
                versions[pkg] = getattr(module, '__version__', 'unknown')
            except:
                versions[pkg] = 'not installed'

        return versions

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _load_session(self):
        """Load existing entries for this session."""
        session_file = self.storage_path / f"session_{self.session_id}.json"
        if session_file.exists():
            try:
                with open(session_file) as f:
                    data = json.load(f)
                    self.entries = [AuditEntry(**e) for e in data.get('entries', [])]
            except:
                pass

    def _save_entry(self, entry: AuditEntry):
        """Save entry to storage."""
        session_file = self.storage_path / f"session_{self.session_id}.json"

        data = {
            'session_id': self.session_id,
            'entries': [e.to_dict() for e in self.entries]
        }

        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_snapshot(self, snapshot: AnalysisSnapshot):
        """Save snapshot to storage."""
        snapshot_file = self.storage_path / f"snapshot_{snapshot.id}.json"

        with open(snapshot_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)


class ReproducibilityManager:
    """Manage analysis reproducibility."""

    def __init__(self, audit_trail: AuditTrail):
        self.audit = audit_trail
        self.current_seed: Optional[int] = None

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.current_seed = seed
        np.random.seed(seed)

        # Set pandas random state
        try:
            import random
            random.seed(seed)
        except:
            pass

    def get_seed(self) -> int:
        """Get current seed or generate new one."""
        if self.current_seed is None:
            self.current_seed = np.random.randint(0, 2**31)
            self.set_seed(self.current_seed)
        return self.current_seed

    def reproducible_run(
        self,
        func: Callable,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        description: str = "Analysis"
    ) -> tuple:
        """Run analysis with full reproducibility tracking.

        Args:
            func: Analysis function to run
            data: Input data
            parameters: Analysis parameters
            description: Description for logging

        Returns:
            Tuple of (result, snapshot_id)
        """
        seed = self.get_seed()
        self.set_seed(seed)

        # Log start
        start_time = datetime.now()

        try:
            # Run analysis
            result = func(data, **parameters)

            # Calculate duration
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Create snapshot
            snapshot = self.audit.create_snapshot(
                description=description,
                data=data,
                analysis_type=func.__name__ if hasattr(func, '__name__') else 'unknown',
                parameters=parameters,
                random_seed=seed,
                result=result
            )

            # Log completion
            self.audit.log(
                action='analysis',
                analysis_type=func.__name__ if hasattr(func, '__name__') else 'unknown',
                parameters=parameters,
                input_data=data,
                output_data=result,
                random_seed=seed,
                status='completed',
                duration_ms=duration_ms
            )

            return result, snapshot.id

        except Exception as e:
            # Log failure
            self.audit.log(
                action='analysis',
                analysis_type=func.__name__ if hasattr(func, '__name__') else 'unknown',
                parameters=parameters,
                input_data=data,
                random_seed=seed,
                status='failed',
                error_message=str(e)
            )
            raise

    def reproduce(
        self,
        snapshot_id: str,
        func: Callable,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Reproduce an analysis from snapshot.

        Args:
            snapshot_id: Snapshot to reproduce
            func: Analysis function
            data: Input data

        Returns:
            Dict with result and verification status
        """
        snapshot = self.audit.snapshots.get(snapshot_id)
        if not snapshot:
            return {'error': 'Snapshot not found'}

        # Verify data matches
        data_hash = self.audit._hash_dataframe(data)
        if data_hash != snapshot.data_hash:
            return {
                'warning': 'Data hash mismatch - may not reproduce exactly',
                'original_hash': snapshot.data_hash,
                'current_hash': data_hash
            }

        # Set seed and run
        self.set_seed(snapshot.random_seed)
        result = func(data, **snapshot.parameters)

        # Verify result
        verification = self.audit.verify_reproducibility(snapshot_id, result)

        return {
            'result': result,
            'verification': verification,
            'seed_used': snapshot.random_seed
        }

    def generate_dockerfile(self) -> str:
        """Generate Dockerfile for exact reproduction."""
        versions = self.audit._get_package_versions()
        python_version = self.audit._get_python_version()

        lines = [
            f"FROM python:{python_version}-slim",
            "",
            "WORKDIR /app",
            "",
            "# Install dependencies",
            "RUN pip install --no-cache-dir \\",
        ]

        for pkg, ver in versions.items():
            if ver != 'not installed' and ver != 'unknown':
                lines.append(f"    {pkg}=={ver} \\")

        lines[-1] = lines[-1].rstrip(" \\")  # Remove trailing backslash
        lines.extend([
            "",
            "COPY . .",
            "",
            'CMD ["python", "reproduce.py"]'
        ])

        return '\n'.join(lines)
