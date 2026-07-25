# Adaptive Concept Selection for Explainable Label-Free Concept Bottleneck Models via Group Relative Policy Optimization
*(AAAI 2027 Submission — Authors Anonymous)*

Official implementation of **ACS-CBM**, a reinforcement learning-based framework for adaptive concept subset selection in Concept Bottleneck Models (CBMs).

## 🧩 Abstract
Concept Bottleneck Models (CBMs) provide interpretable predictions by explicitly reasoning through semantic concepts. Recent label-free CBMs leverage vision-language models to automatically construct concept banks, eliminating the need for expensive concept annotations. However, these concept banks often contain noisy, redundant, and task-irrelevant concepts, limiting both predictive performance and interpretability. We propose Adaptive Concept Selection (ACS-CBM), a reinforcement learning framework that automatically discovers compact and informative concept subsets for label-free CBMs. Specifically, we formulate concept selection as a structured combinatorial optimization problem and employ Group Relative Policy Optimization to efficiently learn task-aware concept selection policies. Our reward function jointly considers downstream task performance, concept quality, and redundancy reduction, while entropy regularization encourages effective exploration in large concept spaces. Experiments on CUB, AwA2, and CheXpert demonstrate that ACS-CBM consistently achieves higher predictive accuracy using significantly fewer concepts than existing label-free CBMs and representative concept selection methods. Furthermore, the proposed approach improves robustness under noisy concept settings and reduces inference complexity, resulting in compact, efficient, and highly interpretable concept bottleneck models. These results highlight adaptive concept selection as a practical direction for scalable explainable learning without concept annotations.

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
