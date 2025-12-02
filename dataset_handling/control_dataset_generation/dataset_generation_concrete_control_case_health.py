import json
import random
import os

# ─────────────── CONFIGURATION ───────────────

output_dir = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_FILE = os.path.join(output_dir, "health_control_scenarios.json")
LABELS_FILE = os.path.join(output_dir, "health_control_labels.json")

N_SCENARIOS = 3000
MIN_ALTS = 2
MAX_ALTS = 5

# Preference target levels
LOW_HEALTH_PREF = 0.0     # for health = 0 (hazardous / non-compliant)
MID_HEALTH_PREF = 0.4     # for health = 1 (unknown or not possible to verify)
HIGH_HEALTH_MIN = 0.9     # for health 2–6 (basic, bronze, silver, gold, platinum)
HIGH_HEALTH_MAX = 1.0

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

# ─────────────── BASE ATTRIBUTES ───────────────

def get_ideal_attributes_except_health():
    """
    Near-ideal attributes for health >= 1.
    """
    circ_orig = random.uniform(80, 100)
    fu_recyc = random.randint(90, 100)
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
    density = int(random.uniform(2300, 2500))
    d_max = round(random.uniform(4, 16), 1)

    return {
        "circ_orig": int(circ_orig),
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
        "density": density,
        "d_max": d_max,
    }

def get_nonideal_attributes_for_health_zero():
    """
    Broader / typical ranges for health = 0.
    The model must still learn that health=0 => pref~0 regardless of how good/bad these are.
    """
    circ_orig = random.uniform(0, 100)

    fu_recyc = random.randint(0, 95)
    remaining = 100 - fu_recyc
    fu_haz = random.randint(0, min(10, remaining))
    remaining -= fu_haz
    fu_incin = random.randint(0, remaining)
    fu_inert = remaining - fu_incin

    gwp = random.uniform(0.05, 0.5)
    wdp = random.uniform(0.03, 0.3)
    fwu = random.uniform(0.0005, 0.005)
    b = random.uniform(0.0, 1.0)

    c_p = random.uniform(0.05, 0.30)
    c_w = random.uniform(0.01, 0.05)
    c_m = random.uniform(0.01, 0.05)

    compressive_strength = round(random.uniform(8, 60), 1)
    slump = round(random.uniform(10, 220), 1)
    w_c_ratio = round(random.uniform(0.30, 0.70), 3)
    cement_content = int(random.uniform(250, 450))
    scm_content = int(random.uniform(0, 200))
    density = int(random.uniform(1600, 2800))
    d_max = round(random.uniform(4, 40), 1)

    return {
        "circ_orig": int(circ_orig),
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
        "density": density,
        "d_max": d_max,
    }

# ─────────────── HEALTH SAMPLING PER SCENARIO ───────────────

def sample_health_for_scenario(n_alts: int):
    """
    Ensure each scenario has:
      - at least one alternative with health in {0,1}
      - at least one alternative with health in {5,6}
      - remaining health values random in [0,6]
    """
    values = []

    # One low health (0 or 1)
    low_health = random.choice([0, 1])
    values.append(low_health)

    if n_alts > 1:
        # One high health (5 or 6)
        high_health = random.choice([5, 6])
        values.append(high_health)

    # Remaining random
    for _ in range(2, n_alts):
        values.append(random.randint(0, 6))

    random.shuffle(values)
    return values

# ─────────────── HEALTH → BASE PREFERENCE MAPPING ───────────────

def health_to_base_pref(h: int) -> float:
    """
    Map health score (0–6) to base preference before noise:
      - 0 -> ~0.0
      - 1 -> ~0.4
      - 2..6 -> linear in [0.9, 1.0]
    """
    if h == 0:
        return LOW_HEALTH_PREF
    elif h == 1:
        return MID_HEALTH_PREF
    else:
        # h in [2,6] -> map to [0.9,1.0]
        t = (h - 2) / 4.0
        return HIGH_HEALTH_MIN + t * (HIGH_HEALTH_MAX - HIGH_HEALTH_MIN)

# ─────────────── CONTROL DATASET GENERATION ───────────────

def generate_health_control_dataset(num_scenarios: int):
    scenarios = []
    all_labels = []

    for i in range(1, num_scenarios + 1):
        case_id = f"control_health_{i}"
        stakeholder = random.choice(STAKEHOLDER_PREFS)
        situation = random.choice(CONCRETE_SCENARIOS)

        n_alts = random.randint(MIN_ALTS, MAX_ALTS)
        health_values = sample_health_for_scenario(n_alts)

        alternatives = []
        labels_for_case = []

        for idx, h in enumerate(health_values):
            # Choose attribute generator based on health
            if h == 0:
                attrs = get_nonideal_attributes_for_health_zero()
            else:
                attrs = get_ideal_attributes_except_health()

            alt = {
                "id_prod": f"prod_{idx+1}",
                "circ_orig": attrs["circ_orig"],
                "fu_recyc": attrs["fu_recyc"],
                "fu_incin": attrs["fu_incin"],
                "fu_inert": attrs["fu_inert"],
                "fu_haz": attrs["fu_haz"],
                "health": h,  # CONTROL VARIABLE
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
                "density": attrs["density"],
                "d_max": attrs["d_max"],
            }
            alternatives.append(alt)

            base_score = health_to_base_pref(h)

            # noise ±0.01
            score_noise = random.uniform(-0.01, 0.01)
            final_score = base_score + score_noise
            final_score = max(0.0, min(1.0, final_score))

            pref_score = round(final_score, 3)

            labels_for_case.append({
                "id_prod": f"prod_{idx+1}",
                "pref": pref_score,
                "conf": 1.0,
                "reason": f"Health control: health={h}, base score={round(base_score,3)} + noise.",
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
    print(f"Generating {N_SCENARIOS} health control scenarios...")
    dataset, labels = generate_health_control_dataset(N_SCENARIOS)

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
        print(f"  {alt['id_prod']} | health={alt['health']} -> pref={lab['pref']}")
