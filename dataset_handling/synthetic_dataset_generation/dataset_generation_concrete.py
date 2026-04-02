
## this code wil be used for synthetic dataset generation (concrete products):

## each priotisation scenario will be made of:
## 2 to 5 alternatives 
## random values for each alternative but consistent with typical values 
## each scenario will be based on the same category (lets say scenario 1, 3 different concrete alternatives). 
            #   this is important to be able to define performance-related values (whether numerical or just text or both)

# every single scenario will be subjected to several stakeholder archetypes, scenarios, as well as combination of stakeholder archetypes & scenarios.
# to stress the model it will be beneficial to include combination of potentially conflicting archetypes (e.g. sustainability + cost-conscious)

import json
import random
import os


# ─────────────── CONFIGURATION ───────────────

current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "dataset.json")

# Total number of scenarios to generate:
N_SCENARIOS = 20000

# Minimum and maximum number of alternatives per scenario:
MIN_ALTS = 2    # At least 2 alternatives must be available for comparison
MAX_ALTS = 5    # No more than 5 alternatives should be presented in any scenario

# Probability that any given numerical or categorical indicator is missing (None):
P_MISSING = 0.01  # 1%

# Stakeholder preference archetypes:
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

# Scenarios (exploratory):
CONCRETE_SCENARIOS = [
    "Standard structural application",
    "Acoustic insulation",
    "Thermal insulation",
    "Architectural finish",
]

# ─────────────── DATA GENERATION LOGIC ───────────────
performance_indicator_values = {
    "compressive_strength":[8,60],            # N/mm2 or MPa
    # consistency / workability indicator 
    "slump":                [10, 300],        # mm
    # durability indicators
    "water_to_cement_ratio": [0.3, 0.7],     # ratio
    "cement_content":   [250, 450],          # kg/m3
    "SCM_content":     [0, 200],             # kg/m3
    "density":[1600,2800],                   #kg/m3, i should enforce most cases fall within 2300-2600 range
    "d_max": [4, 40],                        # mm (max aggregate size)
}

# Typical numeric ranges for each indicator:
RANGES = {
    # Percentage of circular content [0, 100]
    "circ_orig": (0, 100),
    # C2C health score: integer 0–6; 0 no, 1 unknown, 2 basic, 3 bronze and so on
    "health": (0, 6),
    # Global warming potential (kg CO₂-eq per kg): [0.05, 0.5]
    "gwp": (0.05, 0.5),
    # water depletion potential (m³ per kg): [0.03, 0.3]
    "wdp": (0.03, 0.3),
    # freshwater use (m³ per kg): [0.0005, 0.005]
    "fwu": (0.0005, 0.005),
    # Biodiversity impact score [0.0, 1.0]
    "b": (0.0, 1.0),
    # Lifecycle cost ranges:
    "c_p": (0.05, 0.2),     # product cost 
    "c_w": (0.01, 0.05),   # workforce/installation/demolition
    "c_m": (0.01, 0.05),     # lifecycle maintenance
}

# ─────────────── HELPER FUNCTIONS ───────────────

def maybe_missing(p=P_MISSING):
    """
    Returns True with probability p, indicating that the field should be set to None.
    """
    return random.random() < p

def generate_stakeholder_preference():
    num_preferences = random.choices([1, 2, 3], weights=[0.9, 0.075, 0.025])[0]
    selected = random.sample(STAKEHOLDER_PREFS, num_preferences)
    return selected 


def generate_concrete_scenarios():
    standard = "Standard structural application"
    others = [s for s in CONCRETE_SCENARIOS if s != standard]

    # Decide if this case is standard or non-standard
    mode = random.choices(["standard", "non_standard"], weights=[0.5, 0.5], k=1)[0]  

    if mode == "standard":
        return [standard]

    # Non-standard: pick 1 or 2 from the other three, with higher weight on 1
    k = random.choices([1, 2], weights=[0.9, 0.1], k=1)[0]
    selected = set(random.sample(others, k))

    # Return in stable order
    return [s for s in CONCRETE_SCENARIOS if s in selected]


# placeholder for category case generation when multiple categories are implemented
def generate_category_case():
    selected="concrete"
    return selected


def generate_end_of_life_percentages():
    """
    Generate end-of-life percentages for a concrete product.

    - fu_haz: 0–10%, but 0 most of the time
    - fu_incin: 0–10%, but 0 most of the time
    - fu_recyc + fu_inert take the remaining share
    All four are integers summing to 100.
    """

    # Hazardous: 0 most of the time, otherwise 1–10
    if random.random() < 0.9:  # 90% chance of 0
        haz = 0
    else:
        haz = random.randint(1, 10)  # inclusive [1,10] 

    # Incineration: 0 most of the time, otherwise 1–10
    if random.random() < 0.9:  # 90% chance of 0
        incin = 0
    else:
        incin = random.randint(1, 10)  # inclusive [1,10]

    # Split remaining between recycling and inert
    remaining = 100 - haz - incin

    frac_recyc = random.uniform(0, 1)
    recyc = int(round(remaining * frac_recyc))
    recyc = min(recyc, remaining)
    inert = remaining - recyc

    return {
        "fu_recyc": recyc,
        "fu_incin": incin,
        "fu_inert": inert,
        "fu_haz": haz,
    }

def generate_single_alternative(index: int, category:str) -> dict:
    """
    Returns a dictionary with keys matching:
      id_prod, circ_orig, dism, fu_recyc, fu_incin, fu_inert, fu_haz, 
      health, gwp, wdp, fwu, b, c (nested), performance
    Each indicator is set to None with probability P_MISSING.
    End-of-life percentages always sum to 100 (before missingness is applied).
    """
    alt = {}
    alt["id_prod"] = f"prod_{index}"

    # circ_orig (percentage of circular content)
    alt["circ_orig"] = None if maybe_missing() else random.randint(*RANGES["circ_orig"])

    # End-of-life percentages sum to 100; apply missingness per field afterward
    eol = generate_end_of_life_percentages()
    for key, val in eol.items():
        alt[key] = None if maybe_missing() else val

    # health (C2C score 0–6)
    alt["health"] = None if maybe_missing() else random.randint(*RANGES["health"])

 
    # gwp (kg CO2-eq per kg) and biodiversity impact score (0–1),
    # where ~49% of b is driven by gwp and 51% is "other stuff".
    if maybe_missing():
        alt["gwp"] = None
        alt["b"] = None
    else:
        # Sample gwp as before
        gwp = random.uniform(*RANGES["gwp"])

        # Normalise gwp to [0,1]
        gwp_norm = (gwp - RANGES["gwp"][0]) / (RANGES["gwp"][1] - RANGES["gwp"][0])

        # Other biodiversity drivers (abiotic depletion, acidification, etc.) as noise in [0,1]
        other = random.uniform(0.0, 1.0)

        # Combine: 49% gwp-driven, 51% other drivers
        b_raw = 0.49 * gwp_norm + 0.51 * other

        # Clip to [0,1] and round
        b = max(0.0, min(1.0, b_raw))

        alt["gwp"] = round(gwp, 3)
        alt["b"] = round(b, 3)


    # wdp (m3 water depletion potential)
    alt["wdp"] = None if maybe_missing() else round(random.uniform(*RANGES["wdp"]), 4)

    # fwu (m3 fresh water use)
    alt["fwu"] = None if maybe_missing() else round(random.uniform(*RANGES["fwu"]), 4)

  
    # c (lifecycle costs as a nested dict)
    cost = {}
    cost["c_p"] = None if maybe_missing() else round(random.uniform(*RANGES["c_p"]), 3)
    cost["c_w"] = None if maybe_missing() else round(random.uniform(*RANGES["c_w"]), 3)
    cost["c_m"] = None if maybe_missing() else round(random.uniform(*RANGES["c_m"]), 3)
    alt["c"] = cost

 
# compressive_strength (MPa)
    alt["compressive_strength"] = None if maybe_missing() else round(
        random.randint(*performance_indicator_values["compressive_strength"]), 1
    )

    if maybe_missing():
        alt["cement_content"] = None
        alt["SCM_content"] = None
    else:
        cement = random.uniform(*performance_indicator_values["cement_content"])  
        # pick a crude "cement family" to decide SCM replacement band
        family = random.choices(
            ["CEM_I_like", "CEM_II_like", "CEM_III_like"],
            weights=[0.4, 0.4, 0.2],
            k=1
        )[0]  

        if family == "CEM_I_like":
            r_scm = random.uniform(0.00, 0.15)
        elif family == "CEM_II_like":
            r_scm = random.uniform(0.10, 0.35)
        else:  
            r_scm = random.uniform(0.30, 0.70)

        scm = r_scm * cement

        alt["cement_content"] = int(round(cement))
        alt["SCM_content"] = int(round(scm))


    alt["water_to_cement_ratio"] = (
        None if maybe_missing()
        else round(random.uniform(*performance_indicator_values["water_to_cement_ratio"]), 3)
    )


    # slump (mm) - bias toward common 10–220 but allow up to 300
    if maybe_missing():
        alt["slump"] = None
    else:
        if random.random() < 0.9:
            alt["slump"] = round(random.uniform(10, 220), 1)
        else:
            alt["slump"] = round(random.uniform(220, performance_indicator_values["slump"][1]), 1)

 
    # density (kg/m3) - bias: most cases 2300-2600 but allow extremes
    if maybe_missing():
        alt["density"] = None
    else:
        if random.random() < 0.80:
            alt["density"] = int(round(random.uniform(2300, 2600)))
        else:
            # sample outside typical band
            low_part = random.random() < 0.5
            if low_part:
                alt["density"] = int(round(random.uniform(RANGES.get("density", (1600,3000))[0], 2299)))
            else:
                alt["density"] = int(round(random.uniform(2601, RANGES.get("density", (1600,3000))[1])))

    # d_max (mm)
    alt["d_max"] = (
        None if maybe_missing()
        else round(random.randint(*performance_indicator_values["d_max"]), 1)
    )


    return alt


def generate_scenarios(num_scenarios: int):
    """
    Produces a list of 'num_scenarios' scenario dictionaries.
    Each scenario contains:
      - id (string)
      - stakeholder_preference (string)
      - concrete situations
      - alternatives (list of 2–5 alternatives)
    """
    scenarios = []
    for i in range(1, num_scenarios):
        scenario = {
            "id": str(i),
            "stakeholder_preference": generate_stakeholder_preference(),
            "situations": generate_concrete_scenarios(),
#            "category":generate_category_case(),
            "alternatives": []
        }
        n_alts = random.randint(MIN_ALTS, MAX_ALTS)
        for j in range(1, n_alts + 1):
            scenario["alternatives"].append(generate_single_alternative(j,"concrete"))
        scenarios.append(scenario)
    return scenarios


if __name__ == "__main__":
    # Generate the dataset and print as a JSON array
    dataset = generate_scenarios(N_SCENARIOS)
  #  print(json.dumps(dataset, indent=2))
    with open(output_path, "w") as f:
     json.dump(dataset, f, indent=2)