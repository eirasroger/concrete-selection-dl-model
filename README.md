# An Attention-Based Deep Learning Recommender Model for Sustainable Product Selection






> **Application to Concrete in Early-Stage Construction**

[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)]()




## Overview

The construction industry contributes significantly to global environmental degradation, accounting for over 31% of annual CO₂ emissions. Despite the availability of sustainable concrete alternatives (e.g., recycled aggregates, low-carbon cements), product selection in early-stage construction is predominantly driven by **cost and availability**, often ignoring environmental and social performance.

This repository hosts the implementation of a **Deep Learning-based Recommender Model** designed to automate the multi-criteria selection of concrete products. This model ranks concrete alternatives using **self-attention mechanisms** and **Transformer-style blocks** based on a holistic "Triple Bottom Line" perspective (Environmental, Social, Economic), tailored to specific stakeholder priorities and project constraints.

### Key Objectives
- **Automate decision-making:** Move beyond manual, cost-driven selection to automated, multi-criteria evaluation.
- **Integrate heterogeneous data:** Handle fragmented data sources (EPDs, technical sheets) with variable quality.
- **Context-awareness:** Adapt recommendations based on physical constraints (e.g., acoustic requirements) and decision-maker profiles (e.g., sustainability maximalist vs. cost-conscious).

---

## Features

*   **Transformer-based architecture:** Utilizes stacked self-attention blocks to capture contextual relationships between competing product alternatives.
*   **Multi-dimensional inputs:** Processes 66+ input features per alternative, covering sustainability metrics, performance (EN 206), and costs. The system explicitly encodes data availability through dedicated presence and relevance flags, allowing robust handling of sparse or incomplete datasets.
*   **8 Stakeholder archetypes:** Pre-defined profiles to simulate diverse decision-making strategies (e.g., *Circular Economy Advocate*, *Risk-Averse Builder*).
*   **Scenario-specific logic:** "Concrete Situation" vectors adjust feature weights based on application needs (e.g., *Acoustic Separation*, *Thermal Insulation*, *Architectural Finish*).
*   **Hybrid learning Strategy:** Trained on a composite dataset of:
    *   **Control cases:** Deterministic scenarios for ground-truth anchoring.
    *   **LLM-generated labels:** Scalable synthetic data with confidence scoring for complex cases.
    *   **Expert-annotated cases:** High-value data for complex trade-off resolution. These cases serve as gold-standard reference points to fine-tune the model and mitigate LLM hallucinations.

---


## Model Architecture
The system implements a ranking-based neural network:
1.  **Input encoder:** MLP (128 $\to$ 64 $\to$ 32) with LayerNorm and Dropout to process individual alternative features.
2.  **Self-attention blocks:** Two Transformer-inspired residual blocks (Multi-Head Attention) to compare alternatives against each other within a single query.
3.  **Context integration:** Concatenates a "Scenario Context Vector" (mean embedding of valid alternatives) to capture global batch information.
4.  **Scoring head:** Final MLP with Sigmoid activation to output a preference score ($0-1$) for each alternative.



![Model Architecture](figures/architecture.png)







---
## Training Strategy
*   **Loss function:** Custom Group-Weighted Smooth L1 Loss.
*   **Weighting:** Leverages control and LLM scenarios as a broad foundational corpus for representation learning, while utilising expert cases for fine-tuning to refine decision boundaries and enhance performance relative to the baseline data.
    *   *Default weights:* Control (2/5), LLM (2/5), Expert (1/5).
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



- **Scenario 1: The “we need the greenest option” meeting**

  The client opens with a simple ask: “pick the lowest-impact concrete we can justify”. The catch is that the shortlist on the table is messy: some products look great on GWP, others look better on circularity, and one has a questionable health profile.

  This is where the model earns its keep: instead of arguing over whichever single metric someone has latched onto, it produces a consistent ranking that reflects the chosen stakeholder priorities and makes the trade-offs comparable across the whole shortlist.

  In practice, it turns a vague sustainability brief into a decision-ready shortlist, plus a clear front-runner.

- **Scenario 2: The cost-driven choice that still has to look responsible**

  A developer is under pressure to keep costs down, but cannot afford to pick something that will be criticised later for poor environmental performance or thin documentation.

  The model helps by ranking options in a way that still rewards credible sustainability performance, but naturally favours value for money when the stakeholder preference is cost-conscious.

  The practical outcome is not “the cheapest at any price”, but a defensible option that is cost-led and still credible.

- **Scenario 3: Circular economy, without accidental side-effects**

  The team wants to push circularity (high secondary material content and strong end-of-life pathways) and they are keen to avoid greenwashing.

  The model supports this by lifting genuinely circular options to the top, whilst still accounting for the uncomfortable bits that often get ignored (water indicators, biodiversity proxy, life-cycle cost, and product health).

  Practically, it stops the team from optimising one circularity headline whilst quietly making something else worse.

- **Scenario 4: Same shortlist, different application context**

  The shortlist might be the same, but the application is not. A partition or slab with an acoustic requirement is a different decision to a lightweight solution where thermal intent (or reduced dead load) matters.

  By switching the “Concrete Situation” context, the ranking shifts in a predictable, transparent way: the model reflects what matters more in that situation, rather than pretending one universal ranking fits every job.

  In practice, it reduces late-stage rework by making context-driven preferences explicit early on.


---
## Contact 

Roger Vergés - Corresponding author and lead developer - [roger.verges.eiras@upc.edu](mailto:roger.verges.eiras@upc.edu)


---
## Additional information 

Related publication: The associated academic paper is currently under review. The DOI will be added here upon acceptance.


### Paper contributors:
- Roger Vergés <sup>1, 2</sup> (<a href="mailto:roger.verges.eiras@upc.edu">roger.verges.eiras@upc.edu</a>) <a href="https://orcid.org/0009-0001-5887-4785" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
- Kàtia Gaspar <sup>1</sup> (<a href="mailto:katia.gaspar@upc.edu">katia.gaspar@upc.edu</a>) <a href="https://orcid.org/0000-0003-3842-1401" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
- Núria Forcada <sup>1</sup> (<a href="mailto:nuria.forcada@upc.edu">nuria.forcada@upc.edu</a>) <a href="https://orcid.org/0000-0003-2109-4205" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
- M. Reza Hosseini <sup>2</sup> (<a href="mailto:mreza.hosseini@unimelb.edu.au">mreza.hosseini@unimelb.edu.au</a>) <a href="https://orcid.org/0000-0001-8675-736X" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>



<sup>1</sup> Group of Construction Research and Innovation (GRIC), <a href="https://www.upc.edu/ca">Universitat Politècnica de Catalunya — BarcelonaTech (UPC)</a>, Terrassa, Catalonia

<sup>2</sup> Faculty of Architecture, Building and Planning, <a href="https://www.unimelb.edu.au/">The University of Melbourne</a>, Parkville, Australia