"""
Compliance Service
==================
Evaluates detected violations against a configurable rule set
and produces a weighted compliance score with full explainability.
"""

import time
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    ComplianceResponse,
    ComplianceRule,
    Detection,
    InspectionStatus,
    SeverityLevel,
    STATUS_AR,
)

logger = get_logger(__name__)
settings = get_settings()

# ── Rule Definitions ──────────────────────────────────────────────────────────
# Each rule references violation class_ids that constitute a failure.

COMPLIANCE_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "SAFETY-001",
        "rule_name": "No Blocked Emergency Exits",
        "rule_name_ar": "لا يوجد مخارج طوارئ مسدودة",
        "violation_class_ids": [1],
        "weight": 0.35,
        "pass_description": "All emergency exits are clear.",
        "pass_description_ar": "جميع مخارج الطوارئ خالية.",
        "fail_description": "Blocked emergency exit detected — critical safety violation.",
        "fail_description_ar": "تم اكتشاف مخرج طوارئ مسدود — انتهاك أمني حرج.",
    },
    {
        "rule_id": "ENV-001",
        "rule_name": "No Illegal Dumping",
        "rule_name_ar": "لا يوجد إلقاء غير قانوني للنفايات",
        "violation_class_ids": [0],
        "weight": 0.25,
        "pass_description": "No illegal waste disposal detected.",
        "pass_description_ar": "لم يتم اكتشاف أي إلقاء غير قانوني للنفايات.",
        "fail_description": "Illegal dumping detected — environmental violation.",
        "fail_description_ar": "تم اكتشاف إلقاء غير قانوني — انتهاك بيئي.",
    },
    {
        "rule_id": "SAFETY-002",
        "rule_name": "Safety Signage Present",
        "rule_name_ar": "وجود لافتات السلامة",
        "violation_class_ids": [2],
        "weight": 0.20,
        "pass_description": "Required safety signs are present.",
        "pass_description_ar": "لافتات السلامة المطلوبة موجودة.",
        "fail_description": "Missing safety signage detected.",
        "fail_description_ar": "تم اكتشاف لافتات سلامة مفقودة.",
    },
    {
        "rule_id": "STRUCT-001",
        "rule_name": "No Structural Damage",
        "rule_name_ar": "لا يوجد تلف هيكلي",
        "violation_class_ids": [3],
        "weight": 0.20,
        "pass_description": "No structural damage detected.",
        "pass_description_ar": "لم يتم اكتشاف أي تلف هيكلي.",
        "fail_description": "Structural damage found — requires immediate inspection.",
        "fail_description_ar": "تم اكتشاف تلف هيكلي — يتطلب تفتيشاً فورياً.",
    },
]

SEVERITY_PENALTY: Dict[SeverityLevel, float] = {
    SeverityLevel.LOW: 0.0,
    SeverityLevel.MEDIUM: 5.0,
    SeverityLevel.HIGH: 10.0,
    SeverityLevel.CRITICAL: 20.0,
}


class ComplianceService:
    """
    Converts detections into a weighted compliance score and structured report.
    Supports explainability metadata for audit and non-technical stakeholders.
    """

    def evaluate(
        self,
        detections: List[Detection],
        model_version: str,
        location_id: str | None = None,
        inspector_id: str | None = None,
        processing_time_ms: float = 0.0,
    ) -> ComplianceResponse:
        start = time.perf_counter()

        violated_class_ids = {d.class_id for d in detections if d.is_violation}
        rules_evaluated = self._evaluate_rules(violated_class_ids)

        base_score = self._compute_base_score(rules_evaluated)
        penalty = self._compute_severity_penalty(detections)
        final_score = max(0.0, min(100.0, base_score - penalty))

        status = self._score_to_status(final_score)
        violation_summary = self._build_violation_summary(detections)
        recommendations, recommendations_ar = self._build_recommendations(rules_evaluated)
        explainability = self._build_explainability(
            rules_evaluated, base_score, penalty, final_score, detections
        )

        elapsed_ms = processing_time_ms + (time.perf_counter() - start) * 1000

        logger.info(
            "Compliance evaluated",
            score=round(final_score, 2),
            status=status,
            violations=len(violated_class_ids),
        )

        return ComplianceResponse(
            request_id=uuid4(),
            location_id=location_id,
            inspector_id=inspector_id,
            overall_score=round(final_score, 2),
            status=status,
            status_ar=STATUS_AR[status],
            rules_evaluated=rules_evaluated,
            detections=detections,
            violation_summary=violation_summary,
            recommendations=recommendations,
            recommendations_ar=recommendations_ar,
            processing_time_ms=round(elapsed_ms, 2),
            model_version=model_version,
            explainability=explainability,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _evaluate_rules(self, violated_ids: set) -> List[ComplianceRule]:
        results = []
        for rule in COMPLIANCE_RULES:
            failed = any(cid in violated_ids for cid in rule["violation_class_ids"])
            results.append(
                ComplianceRule(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    rule_name_ar=rule["rule_name_ar"],
                    passed=not failed,
                    weight=rule["weight"],
                    details=rule["pass_description"] if not failed else rule["fail_description"],
                    details_ar=rule["pass_description_ar"] if not failed else rule["fail_description_ar"],
                )
            )
        return results

    def _compute_base_score(self, rules: List[ComplianceRule]) -> float:
        """Weighted score: sum(weight * pass) / sum(weight) * 100."""
        total_weight = sum(r.weight for r in rules)
        if total_weight == 0:
            return 100.0
        passed_weight = sum(r.weight for r in rules if r.passed)
        return (passed_weight / total_weight) * 100.0

    def _compute_severity_penalty(self, detections: List[Detection]) -> float:
        """Additional penalty for high/critical severity detections."""
        return sum(SEVERITY_PENALTY.get(d.severity, 0.0) for d in detections if d.is_violation)

    def _score_to_status(self, score: float) -> InspectionStatus:
        if score >= 85.0:
            return InspectionStatus.COMPLIANT
        if score >= 60.0:
            return InspectionStatus.NEEDS_REVIEW
        return InspectionStatus.NON_COMPLIANT

    def _build_violation_summary(self, detections: List[Detection]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for d in detections:
            if d.is_violation:
                summary[d.class_name] = summary.get(d.class_name, 0) + 1
        return summary

    def _build_recommendations(
        self, rules: List[ComplianceRule]
    ) -> Tuple[List[str], List[str]]:
        rec_en, rec_ar = [], []
        failed = [r for r in rules if not r.passed]
        for r in failed:
            rec_en.append(f"[{r.rule_id}] {r.details}")
            rec_ar.append(f"[{r.rule_id}] {r.details_ar}")
        if not failed:
            rec_en.append("All compliance rules passed. Continue regular monitoring.")
            rec_ar.append("اجتازت جميع قواعد الامتثال. تابع المراقبة المنتظمة.")
        return rec_en, rec_ar

    def _build_explainability(
        self,
        rules: List[ComplianceRule],
        base_score: float,
        penalty: float,
        final_score: float,
        detections: List[Detection],
    ) -> Dict[str, Any]:
        return {
            "scoring_method": "weighted_rules_with_severity_penalty",
            "base_score": round(base_score, 2),
            "severity_penalty": round(penalty, 2),
            "final_score": round(final_score, 2),
            "rule_contributions": [
                {
                    "rule_id": r.rule_id,
                    "weight": r.weight,
                    "passed": r.passed,
                    "contribution": round(r.weight * 100 if r.passed else 0, 2),
                }
                for r in rules
            ],
            "violation_severities": {
                d.class_name: d.severity.value
                for d in detections
                if d.is_violation
            },
        }
