# An Attention-Based Deep Learning Recommender Model for Sustainable Product Selection






> **Application to Concrete in Early-Stage Construction**

[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()




## Overview

The construction industry contributes significantly to global environmental degradation, accounting for over 31% of annual CO₂ emissions. Despite the availability of sustainable concrete alternatives (e.g., recycled aggregates, low-carbon cements), product selection in early-stage construction is predominantly driven by **cost and availability**, often ignoring environmental and social performance.

This repository hosts the implementation of a **Deep Learning-based Recommender Model** designed to automate the multi-criteria selection of concrete products. This model ranks concrete alternatives using **self-attention mechanisms** and **Transformer-style blocks** based on a holistic "Triple Bottom Line" perspective (Environmental, Social, Economic), tailored to specific stakeholder priorities and project constraints.

### Key Objectives
- **Automate Decision-Making:** Move beyond manual, cost-driven selection to automated, multi-criteria evaluation.
- **Integrate Heterogeneous Data:** Handle fragmented data sources (EPDs, technical sheets) with variable quality.
- **Context-Awareness:** Adapt recommendations based on physical constraints (e.g., acoustic requirements) and decision-maker profiles (e.g., sustainability maximalist vs. cost-conscious).

---

## Features

*   **Transformer-Based Architecture:** Utilizes stacked self-attention blocks to capture contextual relationships between competing product alternatives.
*   **Multi-Dimensional Inputs:** Processes 106+ input features per alternative, covering sustainability metrics, regulatory compliance (EN 206), and costs.
*   **8 Stakeholder Archetypes:** Pre-defined profiles to simulate diverse decision-making strategies (e.g., *Circular Economy Advocate*, *Risk-Averse Builder*).
*   **Scenario-Specific Logic:** "Concrete Situation" vectors adjust feature weights based on application needs (e.g., *Acoustic Separation*, *Thermal Insulation*, *Architectural Finish*).
*   **Hybrid Learning Strategy:** Trained on a composite dataset of:
    *   **Control Cases:** Deterministic scenarios for ground-truth anchoring.
    *   **LLM-Generated Labels:** Scalable synthetic data with confidence scoring.
    *   **Expert-Annotated Cases:** High-value data for complex trade-off resolution.

---


## Model Architecture
The system implements a ranking-based neural network:
1.  **Input Encoder:** MLP (128 $\to$ 64 $\to$ 32) with LayerNorm and Dropout to process individual alternative features.
2.  **Self-Attention Blocks:** Two Transformer-inspired residual blocks (Multi-Head Attention) to compare alternatives against each other within a single query.
3.  **Context Integration:** Concatenates a "Scenario Context Vector" (mean embedding of valid alternatives) to capture global batch information.
4.  **Scoring Head:** Final MLP with Sigmoid activation to output a preference score ($0-1$) for each alternative.
---
## Training Strategy
*   **Loss Function:** Custom Group-Weighted Smooth L1 Loss.
*   **Weighting:** Prioritizes reliable signals (Control/Expert) over synthetic noise (LLM) using confidence-based sample weighting.
    *   *Default Weights:* Control (2/5), LLM (1/5), Expert (2/5).
*   **Optimization:** Adam optimizer with ReduceLROnPlateau scheduler.

---

## Repository Structure

*(This section is currently under development)*


---

## Getting Started

*(This section is currently under development)*

### Prerequisites
- Python 3.12+
- PyTorch 2.6.0 (CUDA 11.8 support recommended)

### Installation

xyzy steps



---

## Use Cases

*(This section is currently under development)*

-   **Scenario 1: xxxx**

    An engineer has a ...

-   **Scenario 2: xxxx**

    A structural engineer is designing a ....

-   **Scenario 3: xxxx**

    A sustainability consultant wants ...


---
## Contact 

Roger Vergés - Corresponding author and lead developer - [roger.verges.eiras@upc.edu](mailto:roger.verges.eiras@upc.edu)


---
## Additional information 

Related publication: The associated academic paper is currently under review. The DOI will be added here upon acceptance.

---
### Paper contributors:
- Roger Vergés <sup>1, 2</sup> (<a href="mailto:roger.verges.eiras@upc.edu">roger.verges.eiras@upc.edu</a>) <a href="https://orcid.org/0009-0001-5887-4785" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
- Kàtia Gaspar <sup>1</sup> (<a href="mailto:katia.gaspar@upc.edu">katia.gaspar@upc.edu</a>) <a href="https://orcid.org/0000-0003-3842-1401" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
- Núria Forcada <sup>1</sup> (<a href="mailto:nuria.forcada@upc.edu">nuria.forcada@upc.edu</a>) <a href="https://orcid.org/0000-0003-2109-4205" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
- M. Reza Hosseini <sup>2</sup> (<a href="mailto:mreza.hosseini@unimelb.edu.au">mreza.hosseini@unimelb.edu.au</a>) <a href="https://orcid.org/0000-0001-8675-736X" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>



<sup>1</sup> Group of Construction Research and Innovation (GRIC), <a href="https://www.upc.edu/ca">Universitat Politècnica de Catalunya — BarcelonaTech (UPC)</a>, Terrassa, Catalonia

<sup>2</sup> Faculty of Architecture, Building and Planning, <a href="https://www.unimelb.edu.au/">The University of Melbourne</a>, Parkville, Australia