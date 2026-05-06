# GRaCE-CBM: RL-based Adaptive Concept Selection for CBMs
*(NeurIPS 2026 Submission — Authors Anonymous)*

Official implementation of **GRaCE-CBM**, a reinforcement learning-based framework for adaptive concept subset selection in Concept Bottleneck Models (CBMs).

## 🧩 Abstract
Concept Bottleneck Models (CBMs) improve interpretability by constraining predictions to pass through human-understandable concepts. However, conventional CBMs require costly concept annotations and assume that predefined concept sets are well-aligned with downstream tasks. Recent label-free CBMs alleviate the annotation bottleneck by leveraging vision-language models to construct large-scale concept banks, but these automatically generated concepts are often noisy, redundant, and weakly task-aligned. In this work, we address the problem of selecting a compact and informative subset of concepts from large label-free concept banks. We formulate concept subset selection as a structured and combinatorial decision-making problem and propose GRaCE-CBM, a reinforcement learning-based framework that learns to select task-relevant concepts via Group Relative Policy Optimization (GRPO). Our method introduces a reward function that jointly considers task utility, concept quality, and redundancy reduction, along with entropy regularization for stable exploration in noisy concept spaces. Extensive experiments on CUB, AwA2, and CheXpert demonstrate that GRaCE-CBM consistently outperforms both full-concept baselines and heuristic selection methods while using significantly fewer concepts. Furthermore, our approach improves robustness under noisy concept conditions and reduces inference cost, resulting in more compact, interpretable, and efficient models.

---

## 📌 Overview

Concept Bottleneck Models (CBMs) aim to provide interpretable predictions via human-understandable concepts.  
However, existing CBMs suffer from:

- Redundant concept representations  
- Poor subset selection (independent ranking)  
- Limited controllability under intervention  

👉 We propose **GRaCE-CBM**, which formulates concept selection as a **subset-level optimization problem** and solves it using reinforcement learning.
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
- This repository is part of a **NeurIPS 2026 anonymized submission**.
- Author identities are removed for double-blind review.
