import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
FROZEN_PATH = os.path.join(DIR_PATH, '../frozen_dataset.json')
LABELED_PATH = os.path.join(DIR_PATH, '../labelled_dataset.json')

BATCH_SIZE = 1024
EPOCHS = 100
LR = 5e-3   # Initial learning rate
HIDDEN_DIM = [128,64,32] 
TEST_SIZE = 0.25
RANDOM_STATE = 2
DROPOUT = 0.1

# Stakeholder preference descriptions
STAKEHOLDER_PREFS = [
    "Sustainability maximalist: tends to prioritise low environmental impact and high circularity above other considerations.",
    "Cost-conscious developer: tends to favour lower lifecycle costs while maintaining acceptable performance and compliance.",
    "Occupant comfort focused: tends to value thermal, acoustic, and safety performance for indoor comfort.",
    "Health & safety focused: tends to avoid hazardous content and prioritise health-related aspects or products easy-to-install (enhanced safety during installation).",
    "Circular economy advocate: tends to prefer materials with high circular origin, reusability, recyclability, and minimal waste.",
    "Regulatory-aligned builder: tends to prefer products that align with current or expected regulatory requirements (including costs) and show data completeness.",
    "Balanced optimizer: tends to seek well-rounded performance across all indicators without extreme trade-offs.",
    "Pragmatic contractor: tends to prefer practical, easy-to-install, design-for-disassembly-based, and low-risk materials that reduce on-site complexity."
]

# Scenario situations
SCENARIO_PREFS = [
    "Standard structural application",
    "Acoustic insulation",
    "Thermal insulation",
    "Architectural finish",
]

# Mapping of performance indicators relevant to each scenario situation
# Since scenario situations are exploratory, we consider all performance indicators relevant for all scenarios
PERFORMANCE_INDICATOR_RELEVANCE_MAPPING = {
    "Standard structural application":["compressive_strength","slump","water_to_cement_ratio","cement_content","SCM_content","density","d_max"],
    "Acoustic insulation": ["density"],
    "Thermal insulation": ["compressive_strength","density"],
    "Architectural finish": ["compressive_strength","slump","water_to_cement_ratio","SCM_content","density","d_max"],
}