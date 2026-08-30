"""
CouncilMed — Council Fusion Engine
====================================
Evidence-aware multi-model fusion that evaluates agreement, disagreement,
uncertainty, and localization consensus to produce the most evidence-supported result.

This replaces the previous hardcoded 40/30/30 weighted average with a scientifically
defensible fusion strategy: Calibrated Ensemble with Uncertainty Weighting.

Fusion Method Rationale
-----------------------
We use a weighted evidence fusion approach where each model's contribution is
weighted by its:
  1. Validated calibration quality (from model registry)
  2. Prediction confidence (entropy-based)
  3. Uncertainty estimate (lower uncertainty → higher weight)
  4. Historical validation performance (AUROC from registry)

This is preferred over simple averaging because:
  - Models with high uncertainty on a specific case are automatically down-weighted
  - Well-calibrated models naturally receive more influence
  - The system adapts per-case rather than using static weights
  - It surfaces genuine disagreement rather than hiding it

Reference: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles" (NeurIPS 2017)
"""

from __future__ import annotations

import math
import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .evidence import (
    CouncilResult,
    ConsensusLevel,
    EvidenceStrength,
    ModelEvidence,
    ModelStatus,
    SeverityLevel,
    LocalizationData,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelCalibration:
    """Calibration data for a model from the registry."""
    model_id: str
    auroc: float = 0.5       # Area Under ROC Curve
    calibration_ece: float = 1.0  # Expected Calibration Error (lower is better)
    reliability_weight: float = 1.0  # Overall reliability weight


# Default calibration — used when registry data is unavailable
DEFAULT_CALIBRATION = ModelCalibration(
    model_id="unknown",
    auroc=0.5,
    calibration_ece=0.5,
    reliability_weight=1.0,
)


def compute_entropy(probabilities: dict[str, float]) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    Returns 0 for a perfectly confident prediction, higher for uncertain ones.
    Normalized to [0, 1] range.
    """
    probs = np.array(list(probabilities.values()), dtype=np.float64)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    
    if len(probs) == 0 or len(probs) == 1:
        return 0.0
    
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = math.log2(len(probs))  # Maximum possible entropy
    
    if max_entropy == 0:
        return 0.0
    
    return float(entropy / max_entropy)  # Normalize to [0, 1]


def compute_model_weight(
    evidence: ModelEvidence,
    calibration: ModelCalibration | None = None,
) -> float:
    """
    Compute the fusion weight for a single model's evidence.
    
    Weight is computed from:
      - confidence (higher → more weight)
      - uncertainty (lower → more weight)
      - calibration quality (lower ECE → more weight)
      - validation performance (higher AUROC → more weight)
    """
    cal = calibration or DEFAULT_CALIBRATION
    
    # Component 1: Prediction confidence (use 1 - entropy for sharpness)
    entropy = compute_entropy(evidence.probabilities)
    sharpness = 1.0 - entropy  # 1.0 = perfectly sharp, 0.0 = maximum entropy
    
    # Component 2: Uncertainty (model-reported, lower is better)
    uncertainty_factor = 1.0 - min(evidence.uncertainty, 1.0)
    
    # Component 3: Calibration quality (lower ECE → better calibrated → more trust)
    calibration_factor = 1.0 - min(cal.calibration_ece, 1.0)
    
    # Component 4: Historical validation (AUROC, centered so 0.5 = no info)
    performance_factor = max(0.0, (cal.auroc - 0.5) * 2)  # Scale 0.5-1.0 → 0.0-1.0
    
    # Combine: geometric-like weighting to prevent any single factor from dominating
    weight = (
        sharpness * 0.30 +
        uncertainty_factor * 0.30 +
        calibration_factor * 0.20 +
        performance_factor * 0.20
    ) * cal.reliability_weight
    
    return max(weight, 0.01)  # Floor to prevent zero weight


def fuse_predictions(
    evidence_list: list[ModelEvidence],
    calibrations: dict[str, ModelCalibration] | None = None,
) -> tuple[str, dict[str, float], float]:
    """
    Fuse predictions from multiple models using uncertainty-weighted averaging.
    
    Returns:
        (primary_finding, fused_probabilities, fused_confidence)
    """
    calibrations = calibrations or {}
    
    # Compute weights
    weights: list[float] = []
    for ev in evidence_list:
        cal = calibrations.get(ev.model_id)
        w = compute_model_weight(ev, cal)
        weights.append(w)
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        norm_weights = [1.0 / len(weights)] * len(weights)
    else:
        norm_weights = [w / total_weight for w in weights]
    
    logger.info(
        "Model weights: %s",
        {ev.model_id: f"{w:.3f}" for ev, w in zip(evidence_list, norm_weights)}
    )
    
    # Collect all class labels across models
    all_classes: set[str] = set()
    for ev in evidence_list:
        all_classes.update(ev.probabilities.keys())
    
    # Weighted average of probabilities
    fused_probs: dict[str, float] = {}
    for cls in all_classes:
        weighted_sum = 0.0
        for ev, w in zip(evidence_list, norm_weights):
            weighted_sum += ev.probabilities.get(cls, 0.0) * w
        fused_probs[cls] = weighted_sum
    
    # Normalize fused probabilities to sum to 1
    prob_sum = sum(fused_probs.values())
    if prob_sum > 0:
        fused_probs = {k: v / prob_sum for k, v in fused_probs.items()}
    
    # Primary finding = class with highest fused probability
    primary_finding = max(fused_probs, key=lambda k: fused_probs[k])
    fused_confidence = fused_probs[primary_finding]
    
    return primary_finding, fused_probs, fused_confidence


def compute_agreement(
    evidence_list: list[ModelEvidence],
) -> tuple[float, int, list[dict]]:
    """
    Compute inter-model agreement.
    
    Returns:
        (agreement_score, models_agreeing_count, disagreement_details)
    """
    if len(evidence_list) < 2:
        return 1.0, len(evidence_list), []
    
    # Get each model's top prediction
    predictions = [ev.prediction for ev in evidence_list]
    prediction_counts = Counter(predictions)
    most_common_prediction, most_common_count = prediction_counts.most_common(1)[0]
    
    # Agreement score: fraction of models that agree with the majority
    agreement_score = most_common_count / len(evidence_list)
    
    # Identify disagreements
    disagreements = []
    for ev in evidence_list:
        if ev.prediction != most_common_prediction:
            disagreements.append({
                "model_id": ev.model_id,
                "model_prediction": ev.prediction,
                "model_confidence": ev.confidence,
                "majority_prediction": most_common_prediction,
                "confidence_gap": abs(
                    ev.probabilities.get(ev.prediction, 0) -
                    ev.probabilities.get(most_common_prediction, 0)
                ),
            })
    
    return agreement_score, most_common_count, disagreements


def compute_ensemble_uncertainty(
    evidence_list: list[ModelEvidence],
) -> float:
    """
    Compute ensemble-level uncertainty using prediction disagreement.
    
    Uses the average pairwise KL divergence between model probability distributions
    as a measure of ensemble uncertainty. Higher divergence = higher uncertainty.
    Also incorporates individual model uncertainties.
    """
    if len(evidence_list) < 2:
        return evidence_list[0].uncertainty if evidence_list else 1.0
    
    # Collect all class labels
    all_classes = sorted(set(
        cls for ev in evidence_list for cls in ev.probabilities
    ))
    
    # Build probability matrix (models × classes)
    prob_matrix = np.zeros((len(evidence_list), len(all_classes)))
    for i, ev in enumerate(evidence_list):
        for j, cls in enumerate(all_classes):
            prob_matrix[i, j] = ev.probabilities.get(cls, 1e-8)
    
    # Normalize rows
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = prob_matrix / np.maximum(row_sums, 1e-8)
    
    # Mean prediction
    mean_pred = prob_matrix.mean(axis=0)
    mean_pred = mean_pred / mean_pred.sum()
    
    # Compute average KL divergence from each model to the mean
    kl_divs = []
    for i in range(len(evidence_list)):
        p = prob_matrix[i]
        q = mean_pred
        # KL(p || q) with epsilon for numerical stability
        kl = np.sum(p * np.log(np.maximum(p, 1e-10) / np.maximum(q, 1e-10)))
        kl_divs.append(max(0.0, kl))
    
    # Mean KL divergence, normalized to [0, 1] with a sigmoid-like scaling
    mean_kl = np.mean(kl_divs)
    disagreement_uncertainty = 1.0 - math.exp(-mean_kl)
    
    # Average individual uncertainties
    individual_uncertainty = np.mean([ev.uncertainty for ev in evidence_list])
    
    # Combine: disagreement-based and individual uncertainty
    ensemble_uncertainty = 0.6 * disagreement_uncertainty + 0.4 * individual_uncertainty
    
    return float(min(ensemble_uncertainty, 1.0))


def compute_spatial_agreement(
    evidence_list: list[ModelEvidence],
) -> tuple[float, LocalizationData | None]:
    """
    Compute spatial agreement between models with localization data.
    
    Returns:
        (spatial_agreement_score, consensus_localization)
    """
    localizations = [
        ev.localization for ev in evidence_list
        if ev.localization is not None and ev.localization.center_world_mm is not None
    ]
    
    if len(localizations) < 2:
        # Return the single localization if available
        loc = localizations[0] if localizations else None
        return 1.0 if localizations else 0.0, loc
    
    # Compute pairwise distances between localization centers
    centers = np.array([loc.center_world_mm for loc in localizations])
    
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)
    
    mean_distance = np.mean(distances)
    
    # Normalize: 0mm distance = 1.0 agreement, >50mm = 0.0
    max_acceptable_distance = 50.0  # mm
    spatial_agreement = max(0.0, 1.0 - mean_distance / max_acceptable_distance)
    
    # Consensus localization: confidence-weighted centroid
    total_conf = sum(loc.confidence for loc in localizations)
    if total_conf > 0:
        consensus_center = tuple(
            sum(loc.center_world_mm[i] * loc.confidence for loc in localizations) / total_conf
            for i in range(3)
        )
    else:
        consensus_center = tuple(np.mean(centers, axis=0))
    
    consensus_loc = LocalizationData(
        center_world_mm=consensus_center,
        confidence=spatial_agreement,
    )
    
    return float(spatial_agreement), consensus_loc


def classify_severity(
    finding: str,
    confidence: float,
    consensus_level: ConsensusLevel,
    severity_map: dict[str, SeverityLevel] | None = None,
) -> SeverityLevel:
    """
    Determine clinical severity based on finding, confidence, and consensus.
    
    Conservative approach: high severity only with high confidence and strong consensus.
    """
    # Default severity mapping
    default_map: dict[str, SeverityLevel] = {
        "normal": SeverityLevel.CLEAR,
        "no finding": SeverityLevel.CLEAR,
        "clear": SeverityLevel.CLEAR,
    }
    
    sev_map = {**default_map, **(severity_map or {})}
    
    # Check explicit mapping (case-insensitive)
    finding_lower = finding.lower()
    for key, severity in sev_map.items():
        if key in finding_lower:
            return severity
    
    # If confidence is too low, don't escalate severity
    if confidence < 0.5 or consensus_level in (ConsensusLevel.WEAK, ConsensusLevel.NO_CONSENSUS):
        return SeverityLevel.INFO
    
    if confidence < 0.7:
        return SeverityLevel.WARNING
    
    return SeverityLevel.CRITICAL


def determine_human_review(
    consensus_level: ConsensusLevel,
    ensemble_uncertainty: float,
    fused_confidence: float,
    disagreements: list[dict],
) -> tuple[bool, list[str]]:
    """
    Determine whether human review is required and why.
    
    CouncilMed NEVER automatically finalizes a diagnosis.
    This function determines the URGENCY of review.
    """
    reasons: list[str] = []
    requires_review = True  # Default: always require review
    
    if consensus_level in (ConsensusLevel.WEAK, ConsensusLevel.NO_CONSENSUS):
        reasons.append(f"Model consensus is {consensus_level.value} — models disagree significantly")
    
    if ensemble_uncertainty > 0.5:
        reasons.append(f"Ensemble uncertainty is high ({ensemble_uncertainty:.2f})")
    
    if fused_confidence < 0.6:
        reasons.append(f"Overall confidence is low ({fused_confidence:.1%})")
    
    if disagreements:
        for d in disagreements:
            reasons.append(
                f"Model {d['model_id']} predicts '{d['model_prediction']}' "
                f"(confidence: {d['model_confidence']:.1%}) — "
                f"disagrees with majority '{d['majority_prediction']}'"
            )
    
    if not reasons:
        reasons.append("Routine review — all AI findings require clinician verification")
    
    return requires_review, reasons


class CouncilEngine:
    """
    The Council Engine — core of CouncilMed.
    
    Takes evidence from multiple independent AI models and produces
    the most evidence-supported consensus result.
    
    Architecture:
        EVIDENCE₁ ─┐
        EVIDENCE₂ ──┼─→ FUSION → AGREEMENT → UNCERTAINTY → CONSENSUS
        EVIDENCE₃ ─┘
    """
    
    def __init__(
        self,
        calibrations: dict[str, ModelCalibration] | None = None,
        severity_map: dict[str, SeverityLevel] | None = None,
    ):
        self.calibrations = calibrations or {}
        self.severity_map = severity_map
    
    def run(
        self,
        evidence_list: list[ModelEvidence],
        study_id: str | None = None,
        series_id: str | None = None,
        modality: str | None = None,
        anatomy: str | None = None,
    ) -> CouncilResult:
        """
        Run the Council fusion on collected evidence.
        
        Args:
            evidence_list: Evidence from each council member
            study_id: Optional study identifier
            series_id: Optional series identifier
            modality: Imaging modality (e.g., "MRI", "CT")
            anatomy: Body region (e.g., "brain", "chest")
        
        Returns:
            CouncilResult with consensus finding, agreement, and uncertainty
        """
        # Filter to only successful evidence
        valid_evidence = [
            ev for ev in evidence_list
            if ev.status == ModelStatus.SUCCESS
        ]
        
        if not valid_evidence:
            logger.warning("No valid evidence to fuse — all models failed or unavailable")
            return CouncilResult(
                consensus_level=ConsensusLevel.INSUFFICIENT,
                evidence_strength=EvidenceStrength.INSUFFICIENT,
                primary_finding="No analysis available",
                requires_human_review=True,
                human_review_reasons=["No models produced valid evidence"],
                member_evidence=evidence_list,
                study_id=study_id,
                series_id=series_id,
                modality=modality,
                anatomy=anatomy,
            )
        
        logger.info(
            "Running Council fusion on %d valid evidence items (from %d total)",
            len(valid_evidence), len(evidence_list)
        )
        
        # Step 1: Fuse predictions
        primary_finding, fused_probs, fused_confidence = fuse_predictions(
            valid_evidence, self.calibrations
        )
        
        # Step 2: Compute agreement
        agreement_score, models_agreeing, disagreements = compute_agreement(valid_evidence)
        
        # Step 3: Compute ensemble uncertainty
        ensemble_uncertainty = compute_ensemble_uncertainty(valid_evidence)
        
        # Step 4: Compute spatial agreement
        spatial_agreement, consensus_loc = compute_spatial_agreement(valid_evidence)
        
        # Step 5: Determine consensus level
        if len(valid_evidence) >= 3:
            if models_agreeing == len(valid_evidence):
                consensus_level = ConsensusLevel.STRONG
            elif models_agreeing >= len(valid_evidence) * 0.67:
                consensus_level = ConsensusLevel.MODERATE
            elif models_agreeing >= 2:
                consensus_level = ConsensusLevel.WEAK
            else:
                consensus_level = ConsensusLevel.NO_CONSENSUS
        elif len(valid_evidence) == 2:
            consensus_level = (
                ConsensusLevel.MODERATE if models_agreeing == 2
                else ConsensusLevel.WEAK
            )
        else:
            consensus_level = ConsensusLevel.INSUFFICIENT
        
        # Step 6: Determine evidence strength
        if fused_confidence > 0.8 and agreement_score > 0.8 and ensemble_uncertainty < 0.3:
            evidence_strength = EvidenceStrength.HIGH
        elif fused_confidence > 0.6 and agreement_score > 0.5:
            evidence_strength = EvidenceStrength.MODERATE
        elif fused_confidence > 0.4:
            evidence_strength = EvidenceStrength.LOW
        else:
            evidence_strength = EvidenceStrength.INSUFFICIENT
        
        # Step 7: Classify severity
        severity = classify_severity(
            primary_finding, fused_confidence, consensus_level, self.severity_map
        )
        
        # Step 8: Determine human review requirements
        requires_review, review_reasons = determine_human_review(
            consensus_level, ensemble_uncertainty, fused_confidence, disagreements
        )
        
        # Step 9: Compute total inference time
        total_time = sum(ev.processing_time_ms for ev in evidence_list)
        
        result = CouncilResult(
            consensus_level=consensus_level,
            evidence_strength=evidence_strength,
            primary_finding=primary_finding,
            primary_finding_confidence=fused_confidence,
            severity=severity,
            models_agreeing=models_agreeing,
            models_total=len(evidence_list),
            agreement_score=agreement_score,
            ensemble_uncertainty=ensemble_uncertainty,
            requires_human_review=requires_review,
            human_review_reasons=review_reasons,
            member_evidence=evidence_list,
            disagreements=disagreements,
            spatial_agreement=spatial_agreement,
            consensus_localization=consensus_loc,
            total_inference_time_ms=total_time,
            study_id=study_id,
            series_id=series_id,
            modality=modality,
            anatomy=anatomy,
        )
        
        logger.info(
            "Council result: finding=%s confidence=%.1f%% consensus=%s uncertainty=%.2f",
            primary_finding,
            fused_confidence * 100,
            consensus_level.value,
            ensemble_uncertainty,
        )
        
        return result
