# Algorithm: Entropy-Weighted Evidence Fusion

Location: `backend/council/engine.py`

## Purpose
In a multi-model ensemble, simple averaging is dangerous. If two highly accurate models agree, but a third poorly trained model wildly disagrees with 100% confidence, a simple average will skew the result. This algorithm weights each model's vote based on its statistical certainty and historical reliability.

## Mathematical Foundation

### 1. Shannon Entropy (Sharpness)
The algorithm calculates the Shannon entropy of a model's output probability distribution:
$$ H(P) = - \sum_{i} P_i \log_2(P_i) $$

- A model that is "guessing" will output similar probabilities for all classes (e.g., 33%, 33%, 33%). This results in **high entropy**.
- A model that is confident will output a sharp distribution (e.g., 99%, 0.5%, 0.5%). This results in **low entropy**.

The algorithm calculates "Sharpness" as `1.0 - normalized_entropy`.

### 2. Model Weight Calculation
Each model receives a weight computed from four factors:
1. **Sharpness**: (30% impact) derived from the entropy calculation above.
2. **Uncertainty**: (30% impact) A model-reported metric (if supported by the architecture).
3. **Calibration ECE**: (20% impact) Expected Calibration Error from the model's registry manifest. Well-calibrated models are trusted more.
4. **Historical AUROC**: (20% impact) Validation performance from the registry manifest.

$$ Weight_m = (S_m(0.3) + U_m(0.3) + C_m(0.2) + A_m(0.2)) \times Reliability_m $$

### 3. Weighted Fusion
The final probabilities are calculated by taking the weighted sum of each model's output distribution, and then normalizing the sums to equal 1.0.

## Why this algorithm?
This is a computationally efficient adaptation of "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (Lakshminarayanan et al.). It provides robust defense against out-of-distribution (OOD) inputs. If a chest X-ray is accidentally fed to the Brain MRI ensemble, all models will likely output high entropy (guessing). The fusion engine will detect this widespread uncertainty and flag the consensus as "Weak," triggering a mandatory human review.

## Alternatives Considered
- **Majority Voting**: Rejected because it discards the nuanced probability gradients (e.g., a 51% vote is treated the same as a 99% vote).
- **Simple Averaging**: Rejected because it gives equal weight to poorly calibrated or highly uncertain models.
