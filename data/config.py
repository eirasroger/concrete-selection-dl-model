import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
FROZEN_PATH = os.path.join(DIR_PATH, '../frozen_dataset.json')
LABELED_PATH = os.path.join(DIR_PATH, '../labelled_dataset.json')

BATCH_SIZE = 5096
EPOCHS = 100
LR = 5e-3   # Initial learning rate
HIDDEN_DIM = [128,64,32] 
TEST_SIZE = 0.25
RANDOM_STATE = 2
DROPOUT = 0.1


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

SCENARIO_PREFS = [
    "Structural - load-bearing",
    "Non-structural - interior",
    "Exterior - general (outdoor, non-coastal)",
    "Coastal - splash or spray zone",
    "Coastal - permanently submerged",
    "Roadside or de-icing salts exposure",
    "Cold-climate - freeze thaw cycles",
    "Industrial - chemical exposure or spills",
    "High accoustic requirement",
    "High thermal insulation requirement",
    "Lightweight requirement",
    "Ready-mix specific workability requirement",
    "Contains steel reinforcement or other embedded metal",
    "Steel reinforcement in direct contact with concrete",
]


PERFORMANCE_INDICATOR_RELEVANCE_MAPPING = {
    "Structural - load-bearing":["compressive_strength", "d_min", "d_max", "density","chloride_content"],
    "Non-structural - interior": [ "d_min", "d_max", "density","chloride_content"],
    "Exterior - general (outdoor, non-coastal)": ["compressive_strength", "d_min", "d_max", "density","chloride_content","exposure_XC"],
    "Coastal - splash or spray zone": ["compressive_strength", "d_min", "d_max", "density","chloride_content","exposure_XD"],
    "Coastal - permanently submerged": ["compressive_strength", "d_min", "d_max", "density","chloride_content", "exposure_XS"],
    "Roadside or de-icing salts exposure": ["compressive_strength", "d_min", "d_max", "density","chloride_content","exposure_XF"],
    "Cold-climate - freeze thaw cycles": ["compressive_strength", "d_min", "d_max", "density","chloride_content","exposure_XF"],
    "Industrial - chemical exposure or spills": ["compressive_strength", "d_min", "d_max", "density","chloride_content","exposure_XA"],
    "High accoustic requirement": ["compressive_strength", "d_min", "d_max", "density","chloride_content"],
    "High thermal insulation requirement": ["compressive_strength", "d_min", "d_max", "density","chloride_content"],
    "Lightweight requirement": ["compressive_strength", "d_min", "d_max", "density","chloride_content"],
    "Ready-mix specific workability requirement": ["compressive_strength", "d_min", "d_max", "density","chloride_content","slump", "flow_diameter", "compactability_index", "slump_flow"],
    "Contains steel reinforcement or other embedded metal": ["compressive_strength", "d_min", "d_max", "density","chloride_content"],
    "Steel reinforcement in direct contact with concrete": ["compressive_strength", "d_min", "d_max", "density","chloride_content"],
}