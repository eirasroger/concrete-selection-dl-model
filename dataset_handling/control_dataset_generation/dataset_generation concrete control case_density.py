import json
import random
import os
import numpy as np

# ─────────────── CONFIGURATION ───────────────

output_dir = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_FILE = os.path.join(output_dir, "density_control_scenarios.json")
LABELS_FILE = os.path.join(output_dir, "density_control_labels.json")

N_SCENARIOS = 3000
MIN_ALTS = 2
MAX_ALTS = 5

# Shared density range (kg/m3) for both modes
DENSITY_MIN = 1700
DENSITY_MAX = 2600

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

# Keep EXACT scenario names
CONCRETE_SCENARIOS = [
    "Standard structural application",
    "Acoustic insulation",
    "Thermal insulation",
    "Architectural finish",
]

# ─────────────── BASE ATTRIBUTES (IDEAL) ───────────────

def get_ideal_attributes_without_density():
    circ_orig = random.uniform(95, 100)
    health = random.randint(4, 6)

    fu_recyc = random.randint(95, 100)
    remaining = 100 - fu_recyc
    fu_incin = 0
    fu_haz = 0
    fu_inert = remaining

    gwp = random.uniform(0.05, 0.10)
    wdp = random.uniform(0.03, 0.05)
    fwu = random.uniform(0.0005, 0.001)
    b = random.uniform(0.0, 0.1)

    c_p = random.uniform(0.05, 0.08)
    c_w = random.uniform(0.01, 0.02)
    c_m = random.uniform(0.01, 0.02)

    compressive_strength = round(random.uniform(40, 60), 1)
    slump = round(random.uniform(150, 220), 1)
    w_c_ratio = round(random.uniform(0.30, 0.40), 3)
    cement_content = int(random.uniform(340, 380))
    scm_content = int(random.uniform(150, 180))
    d_max = round(random.uniform(4, 16), 1)

    return {
        "circ_orig": int(circ_orig),
        "health": health,
        "fu_recyc": fu_recyc,
        "fu_incin": fu_incin,
        "fu_inert": fu_inert,
        "fu_haz": fu_haz,
        "gwp": round(gwp, 3),
        "wdp": round(wdp, 4),
        "fwu": round(fwu, 5),
        "b": round(b, 3),
        "c": {
            "c_p": round(c_p, 3),
            "c_w": round(c_w, 3),
            "c_m": round(c_m, 3),
        },
        "compressive_strength": compressive_strength,
        "slump": slump,
        "water_to_cement_ratio": w_c_ratio,
        "cement_content": cement_content,
        "SCM_content": scm_content,
        "d_max": d_max,
    }

# ─────────────── DENSITY SAMPLING PER SCENARIO ───────────────

def sample_densities_for_scenario(n_alts: int):
    """
    Ensure each scenario has:
      - one density near the minimum,
      - one density near the maximum,
      - remaining densities sampled randomly in [DENSITY_MIN, DENSITY_MAX].
    """
    densities = []

    # One near min
    near_min = random.uniform(DENSITY_MIN, DENSITY_MIN + 50)
    densities.append(int(round(near_min)))

    if n_alts > 1:
        # One near max
        near_max = random.uniform(DENSITY_MAX - 50, DENSITY_MAX)
        densities.append(int(round(near_max)))

    # Remaining random
    for _ in range(2, n_alts):
        val = random.uniform(DENSITY_MIN, DENSITY_MAX)
        densities.append(int(round(val)))

    random.shuffle(densities)
    return densities

# ─────────────── CONTROL DATASET GENERATION ───────────────

def generate_density_control_dataset(num_scenarios: int):
    scenarios = []
    all_labels = []

    for i in range(1, num_scenarios + 1):
        # thermal or acoustic case
        mode = random.choice(["thermal", "acoustic"])

        case_id = f"control_density_{i}"
        stakeholder = random.choice(STAKEHOLDER_PREFS)

        # Exact scenario names
        if mode == "acoustic":
            situation = "Acoustic insulation"
        else:
            situation = "Thermal insulation"

        n_alts = random.randint(MIN_ALTS, MAX_ALTS)

        # Preference direction:
        # - acoustic: higher density → better
        # - thermal: lower density → better
        increasing_pref = (mode == "acoustic")

        densities = sample_densities_for_scenario(n_alts)

        alternatives = []
        labels_for_case = []

        for idx, final_density in enumerate(densities):
            attrs = get_ideal_attributes_without_density()

            alt = {
                "id_prod": f"prod_{idx+1}",
                "circ_orig": attrs["circ_orig"],
                "fu_recyc": attrs["fu_recyc"],
                "fu_incin": attrs["fu_incin"],
                "fu_inert": attrs["fu_inert"],
                "fu_haz": attrs["fu_haz"],
                "health": attrs["health"],
                "gwp": attrs["gwp"],
                "wdp": attrs["wdp"],
                "fwu": attrs["fwu"],
                "b": attrs["b"],
                "c": attrs["c"],
                "compressive_strength": attrs["compressive_strength"],
                "slump": attrs["slump"],
                "water_to_cement_ratio": attrs["water_to_cement_ratio"],
                "cement_content": attrs["cement_content"],
                "SCM_content": attrs["SCM_content"],
                "density": final_density, # CONTROL VARIABLE
                "d_max": attrs["d_max"],
            }
            alternatives.append(alt)

            # Preference mapping based on density
            dens_range = DENSITY_MAX - DENSITY_MIN
            normalized = (final_density - DENSITY_MIN) / dens_range if dens_range > 0 else 0.5

            if increasing_pref:
                # Acoustic: higher density → higher preference
                base_score = PREF_MIN + (PREF_MAX - PREF_MIN) * normalized
            else:
                # Thermal: lower density → higher preference
                base_score = PREF_MIN + (PREF_MAX - PREF_MIN) * (1 - normalized)

            score_noise = random.uniform(-0.01, 0.01)
            final_score = min(1.0, base_score + score_noise)
            pref_score = round(final_score, 3)

            labels_for_case.append({
                "id_prod": f"prod_{idx+1}",
                "pref": pref_score,
                "conf": 1.0,
                "reason": (
                    f"Density control ({mode}): density={final_density} kg/m3, "
                    f"base score={round(base_score,3)} + noise."
                ),
            })

        scenario_obj = {
            "id": case_id,
            "stakeholder_preference": [stakeholder],
            "situations": [situation],
            "alternatives": alternatives,
        }

        scenarios.append(scenario_obj)
        all_labels.append({"id": case_id, "labelled_alternatives": labels_for_case})

    return scenarios, all_labels

if __name__ == "__main__":
    print(f"Generating {N_SCENARIOS} density control scenarios...")
    dataset, labels = generate_density_control_dataset(N_SCENARIOS)

    with open(SCENARIOS_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)

    print("Done.")
    print(f"Scenarios saved to: {SCENARIOS_FILE}")
    print(f"Labels saved to: {LABELS_FILE}")

    s0 = dataset[0]
    l0 = labels[0]
    print(f"\nPreview case: {s0['id']} | situation={s0['situations'][0]}")
    for alt, lab in zip(s0["alternatives"], l0["labelled_alternatives"]):
        print(f"  {alt['id_prod']} | density={alt['density']} kg/m3 -> pref={lab['pref']}")
