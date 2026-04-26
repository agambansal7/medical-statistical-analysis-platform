"""Core Infrastructure Package.

Provides core system capabilities:
- audit_trail: Complete analysis logging and reproducibility
- educational: Educational content and explanations
- performance: Async execution, caching, and scalability
- security: HIPAA compliance, encryption, and PHI detection
"""

from .audit_trail import AuditTrail, AuditEntry, ReproducibilityManager
from .educational import EducationalContent, StatisticalExplainer
from .performance import PerformanceManager, AsyncExecutor, CacheManager
from .security import SecurityManager, PHIDetector, DataEncryption

__all__ = [
    "AuditTrail",
    "AuditEntry",
    "ReproducibilityManager",
    "EducationalContent",
    "StatisticalExplainer",
    "PerformanceManager",
    "AsyncExecutor",
    "CacheManager",
    "SecurityManager",
    "PHIDetector",
    "DataEncryption",
]
