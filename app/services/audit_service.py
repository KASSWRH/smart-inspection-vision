"""
Audit Service
=============
Immutable audit trail for every AI decision made by the system.
Logs are stored as JSONL (append-only) and queryable via API.
No PII is stored — only image hashes and decision metadata.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import AuditLogEntry

logger = get_logger(__name__)
settings = get_settings()


class AuditService:
    """
    Append-only audit logger for AI decisions.

    Each log entry records:
    - Which model made the decision (version)
    - Input image hash (SHA-256, no PII)
    - Decision output summary
    - Confidence distribution
    - Endpoint and user context
    """

    def __init__(self) -> None:
        self._log_path = settings.AUDIT_LOG_PATH
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        request_id: UUID,
        endpoint: str,
        input_hash: str,
        model_version: str,
        decision_summary: Dict[str, Any],
        confidence_scores: List[float],
        processing_time_ms: float,
        user_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Write a single audit entry to the JSONL log file."""
        if not settings.AUDIT_LOG_ENABLED:
            logger.debug("Audit logging disabled — skipping.")
            return AuditLogEntry(
                request_id=request_id,
                endpoint=endpoint,
                input_hash=input_hash,
                model_version=model_version,
                decision_summary=decision_summary,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time_ms,
                environment=settings.ENVIRONMENT,
            )

        entry = AuditLogEntry(
            request_id=request_id,
            endpoint=endpoint,
            user_id=user_id,
            input_hash=input_hash,
            model_version=model_version,
            decision_summary=decision_summary,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time_ms,
            environment=settings.ENVIRONMENT,
        )

        self._append(entry)
        logger.debug("Audit entry written", log_id=str(entry.log_id), endpoint=endpoint)
        return entry

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent N audit entries."""
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-limit:]
        return [json.loads(line) for line in reversed(recent)]

    # ── Private ───────────────────────────────────────────────────────────────

    def _append(self, entry: AuditLogEntry) -> None:
        """Append serialized entry to JSONL file (thread-safe open/close)."""
        record = entry.model_dump(mode="json")
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ── Singleton ─────────────────────────────────────────────────────────────────

_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
