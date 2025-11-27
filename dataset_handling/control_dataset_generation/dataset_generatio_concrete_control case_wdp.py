import json
import random
import os
import numpy as np

# ─────────────── CONFIGURATION ───────────────

output_dir = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_FILE = os.path.join(output_dir, "wdp_control_scenarios.json")
LABELS_FILE = os.path.join(output_dir, "wdp_control_labels.json")

# Total number of control cases to generate
N_SCENARIOS = 3000

# Alternatives per scenario
MIN_ALTS = 2
MAX_ALTS = 5

# WDP Range (m3 water depletion potential per kg)
WDP_MIN = 0.03
WDP_MAX = 0.3

# Preference Score Range (Higher score = Better)
PREF_MAX = 1.0
PREF_MIN = 0.60 

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

CONCRETE_SCENARIOS = [
    "Standard structural application",
    "Acoustic insulation",
    "Thermal insulation",
    "Architectural finish",
]

# ─────────────── GENERATION LOGIC ───────────────

def get_ideal_attributes():
    """
    Returns a dictionary of attributes set to 'Ideal' values with small random variation.
    NOTE: WDP is excluded here as it is the control variable.
    """
    
    # Circularity & Health (Ideal = High)
    circ_orig = random.uniform(95, 100) 
    health = random.randint(4, 6) # Silver, Gold, Platinum
    
    # End of Life (Ideal = High Recycling/Reuse)
    fu_recyc = random.randint(95, 100)
    remaining = 100 - fu_recyc
    fu_incin = 0
    fu_haz = 0
    fu_inert = remaining 

    # GWP (Fixed at Ideal/Low)
    gwp = random.uniform(0.05, 0.10)

    # Freshwater Use & Biodiversity (Ideal = Low impact)
    fwu = random.uniform(0.0005, 0.001)
    b = random.uniform(0.0, 0.1) 
    
    # Costs (Ideal = Low)
    c_p = random.uniform(0.05, 0.08) 
    c_w = random.uniform(0.01, 0.02) 
    c_m = random.uniform(0.01, 0.02) 

    # Technical Performance (Ideal = High Strength / Good Durability)
    compressive_strength = round(random.uniform(40, 60), 1)
    slump = round(random.uniform(150, 220), 1)
    w_c_ratio = round(random.uniform(0.30, 0.4), 3)
    cement_content = int(random.uniform(340, 380)) 
    scm_content = int(random.uniform(150, 180)) 
    density = int(random.uniform(2400, 2500))
    d_max = round(random.uniform(4, 8), 1)

    return {
        "circ_orig": int(circ_orig),
        "health": health,
        "fu_recyc": fu_recyc, "fu_incin": fu_incin, "fu_inert": fu_inert, "fu_haz": fu_haz,
        "gwp": round(gwp, 3), 
        "fwu": round(fwu, 5),
        "b": round(b, 3),
        "c": {"c_p": round(c_p, 3), "c_w": round(c_w, 3), "c_m": round(c_m, 3)},
        "compressive_strength": compressive_strength,
        "slump": slump,
        "water_to_cement_ratio": w_c_ratio,
        "cement_content": cement_content,
        "SCM_content": scm_content,
        "density": density,
        "d_max": d_max
    }

def generate_control_dataset(num_scenarios):
    scenarios = []
    all_labels = []

    for i in range(1, num_scenarios + 1):
        case_id = f"control_wdp_{i}"
        stakeholder = random.choice(STAKEHOLDER_PREFS)
        situation = random.choice(CONCRETE_SCENARIOS)
        
        n_alts = random.randint(MIN_ALTS, MAX_ALTS)
        
        # ──────── WDP MAPPING ────────
        # Evenly spaced WDP values across the range
        base_wdps = np.linspace(WDP_MIN, WDP_MAX, n_alts)
        np.random.shuffle(base_wdps)
        
        alternatives = []
        labels_for_case = []

        for idx, wdp_val in enumerate(base_wdps):
            # Add small noise to WDP input
            input_noise = random.uniform(-0.001, 0.001) 
            final_wdp = max(WDP_MIN, min(WDP_MAX, wdp_val + input_noise))
            final_wdp = round(final_wdp, 4)

            # Get Base Ideal Attributes (including low GWP)
            attrs = get_ideal_attributes()
            
            alt = {
                "id_prod": f"prod_{idx+1}",
                "circ_orig": attrs["circ_orig"],
                "fu_recyc": attrs["fu_recyc"],
                "fu_incin": attrs["fu_incin"],
                "fu_inert": attrs["fu_inert"],
                "fu_haz": attrs["fu_haz"],
                "health": attrs["health"],
                "gwp": attrs["gwp"], 
                "wdp": final_wdp,    # CONTROL VARIABLE
                "fwu": attrs["fwu"],
                "b": attrs["b"],
                "c": attrs["c"],
                "compressive_strength": attrs["compressive_strength"],
                "slump": attrs["slump"],
                "water_to_cement_ratio": attrs["water_to_cement_ratio"],
                "cement_content": attrs["cement_content"],
                "SCM_content": attrs["SCM_content"],
                "density": attrs["density"],
                "d_max": attrs["d_max"]
            }
            alternatives.append(alt)

            # ──────── CALCULATE LABEL ────────
            # Lower WDP = Higher Score
            wdp_range = WDP_MAX - WDP_MIN
            score_range = PREF_MAX - PREF_MIN
            
            normalized_position = (WDP_MAX - final_wdp) / wdp_range
            base_score = PREF_MIN + (score_range * normalized_position)
            
            # Add random noise to the score (+/- 0.01)
            score_noise = random.uniform(-0.01, 0.01)
            final_score = base_score + score_noise
            
            # Cap at 1.0
            final_score = min(1.0, final_score)
            
            pref_score = round(final_score, 3)

            labels_for_case.append({
                "id_prod": f"prod_{idx+1}",
                "pref": pref_score,
                "conf": 1.0,
                "reason": f"Control Case: WDP {final_wdp} results in base score {round(base_score,3)} + noise."
            })

        scenario_obj = {
            "id": case_id,
            "stakeholder_preference": [stakeholder],
            "situations": [situation],
            "alternatives": alternatives
        }
        
        scenarios.append(scenario_obj)
        all_labels.append({"id": case_id, "labelled_alternatives": labels_for_case})

    return scenarios, all_labels

if __name__ == "__main__":
    print(f"Generating {N_SCENARIOS} WDP control scenarios...")
    dataset, labels = generate_control_dataset(N_SCENARIOS)
    
    with open(SCENARIOS_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
        
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)
        
    print("Done.")
    print(f"Scenarios saved to: {SCENARIOS_FILE}")
    print(f"Labels saved to: {LABELS_FILE}")

    print("\n--- PREVIEW OF LOGIC (Scenario 1) ---")
    s1 = dataset[0]
    l1 = labels[0]
    print(f"Context: {s1['stakeholder_preference'][0]}")
    print("Alternatives:")
    for alt, lab in zip(s1['alternatives'], l1['labelled_alternatives']):
        print(f"  ID: {alt['id_prod']} | WDP: {alt['wdp']} -> Score: {lab['pref']}")
