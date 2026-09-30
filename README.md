# Instance-Wise Concept Selection for Label-Free Concept Bottleneck Models
*(Authors Anonymous)*

Official implementation of **ACS-CBM**, a reinforcement learning-based framework for adaptive concept subset selection in Concept Bottleneck Models (CBMs).

## 🧩 Abstract
Concept Bottleneck Models (CBMs) provide interpretable predictions by
routing decisions through semantic concepts. Recent label-free CBMs use
vision--language models to construct concept banks automatically, but
these banks are typically noisy, redundant, and only weakly aligned with
the target task. We propose ACS-CBM, which formulates instance-wise
concept subset selection as a structured policy-learning problem and
optimizes it with Group Relative Policy Optimization (GRPO). The reward
jointly considers downstream task utility and concept quality while
penalizing redundancy. Since no pretrained reference policy exists in
this setting, we replace the KL term of the original GRPO formulation
with entropy regularization, which stabilizes exploration in large and
noisy concept spaces. Experiments on CUB, AwA2, and CheXpert show that
ACS-CBM outperforms full-concept label-free CBMs and representative
subset-selection baselines while activating at most 3.2\% of the concept
bank per input, yielding compact and less redundant instance-specific
concept subsets. Further analyses show improved robustness under concept noise and
substantially more compact instance-level explanations at comparable
computational cost.

---

## 📌 Overview

Concept Bottleneck Models (CBMs) aim to provide interpretable predictions via human-understandable concepts.  
However, existing CBMs suffer from:

- Redundant concept representations  
- Poor subset selection (independent ranking)  
- Limited controllability under intervention  

👉 We propose **ACS-CBM**, which formulates concept selection as a **subset-level optimization problem** and solves it using reinforcement learning.
---

## 🔥 Key Contributions

- 🎯 **RL-based subset optimization (GRPO)**  
  → selects *complementary* concept subsets instead of independent ranking  

- 🧠 **Disentangled concept representations**  
  → reduces redundancy and improves interpretability  

- ⚡ **Improved intervention controllability**  
  → stable and predictable behavior under concept intervention  

- 📈 **Robust performance across datasets**  
  → CUB, AwA2, CheXpert  
---

## 📄 Notes
- This repository is part of a **AAAI 2027 anonymized submission**.
- Author identities are removed for double-blind review.
