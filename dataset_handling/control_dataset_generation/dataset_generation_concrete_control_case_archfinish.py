import json
import random
import os

# ─────────────── CONFIGURATION ───────────────

output_dir = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_FILE = os.path.join(output_dir, "archfinish_control_scenarios.json")
LABELS_FILE = os.path.join(output_dir, "archfinish_control_labels.json")

N_SCENARIOS = 3000
MIN_ALTS = 2
MAX_ALTS = 5

PREF_MAX = 1.0
PREF_MIN = 0.60

# Shared ranges 
CIRC_MIN = 0
CIRC_MAX = 100

DMAX_MIN = 4   # mm
DMAX_MAX = 40  # mm

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

# IMPORTANT: keep scenario names identical to your main code
CONCRETE_SCENARIOS = [
    "Standard structural application",
    "Acoustic insulation",
    "Thermal insulation",
    "Architectural finish",
]

# ─────────────── BASE ATTRIBUTES (IDEAL EXCEPT circ_orig/d_max) ───────────────

def get_ideal_attributes_except_circ_and_dmax():
    """
    All indicators near-ideal, except circ_orig and d_max which will be controlled separately.
    """
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
    density = int(random.uniform(2300, 2500))

    return {
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
        "density": density,
    }

# ─────────────── SAMPLERS FOR CONTROL VARIABLES ───────────────

def sample_circ_for_scenario(n_alts: int):
    """
    circ_orig control:
      - one near minimum (0–10%)  -> ideal for finish (more 'pure' concrete)
      - one near maximum (90–100%) -> worst
      - rest random in [0,100]
    """
    vals = []
    near_min = random.uniform(CIRC_MIN, CIRC_MIN + 10)
    vals.append(int(round(near_min)))

    if n_alts > 1:
        near_max = random.uniform(CIRC_MAX - 10, CIRC_MAX)
        vals.append(int(round(near_max)))

    for _ in range(2, n_alts):
        vals.append(int(round(random.uniform(CIRC_MIN, CIRC_MAX))))

    random.shuffle(vals)
    return vals

def sample_dmax_for_scenario(n_alts: int):
    """
    d_max control:
      - one near minimum (4–8 mm)  -> ideal for smooth finish
      - one near maximum (32–40 mm) -> worst
      - rest random in [4,40]
    """
    vals = []
    near_min = random.uniform(DMAX_MIN, DMAX_MIN + 4)
    vals.append(round(near_min, 1))

    if n_alts > 1:
        near_max = random.uniform(DMAX_MAX - 8, DMAX_MAX)
        vals.append(round(near_max, 1))

    for _ in range(2, n_alts):
        vals.append(round(random.uniform(DMAX_MIN, DMAX_MAX), 1))

    random.shuffle(vals)
    return vals

# ─────────────── CONTROL DATASET GENERATION ───────────────

def generate_archfinish_control_dataset(num_scenarios: int):
    scenarios = []
    all_labels = []

    for i in range(1, num_scenarios + 1):
        # Mode: either circ_orig-controlled or d_max-controlled
        mode = random.choice(["circ", "dmax"])

        case_id = f"control_archfinish_{i}"
        stakeholder = random.choice(STAKEHOLDER_PREFS)

        # Always architectural finish situation
        situation = "Architectural finish"

        n_alts = random.randint(MIN_ALTS, MAX_ALTS)

        if mode == "circ":
            circ_values = sample_circ_for_scenario(n_alts)
            dmax_values = [round(random.uniform(4, 12), 1) for _ in range(n_alts)]  # keep d_max small/ideal-ish
        else:
            dmax_values = sample_dmax_for_scenario(n_alts)
            circ_values = [int(round(random.uniform(0, 20))) for _ in range(n_alts)]  # keep circ near low/ideal-ish

        alternatives = []
        labels_for_case = []

        for idx in range(n_alts):
            circ_val = circ_values[idx]
            dmax_val = dmax_values[idx]

            attrs = get_ideal_attributes_except_circ_and_dmax()

            alt = {
                "id_prod": f"prod_{idx+1}",
                "circ_orig": circ_val,
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
                "density": attrs["density"],
                "d_max": dmax_val,
            }
            alternatives.append(alt)

            # Preference mapping
            if mode == "circ":
                # Lower circ_orig -> better for finish (more 'pure' mix)
                span = CIRC_MAX - CIRC_MIN
                norm = (circ_val - CIRC_MIN) / span if span > 0 else 0.5
                base_score = PREF_MIN + (PREF_MAX - PREF_MIN) * (1 - norm)
                reason = f"Architectural finish control (circ_orig): circ_orig={circ_val}%, base score={round(base_score,3)} + noise."
            else:
                # Lower d_max -> smoother finish -> higher preference
                span = DMAX_MAX - DMAX_MIN
                norm = (dmax_val - DMAX_MIN) / span if span > 0 else 0.5
                base_score = PREF_MIN + (PREF_MAX - PREF_MIN) * (1 - norm)
                reason = f"Architectural finish control (d_max): d_max={dmax_val} mm, base score={round(base_score,3)} + noise."

            score_noise = random.uniform(-0.01, 0.01)
            final_score = min(1.0, base_score + score_noise)
            pref_score = round(final_score, 3)

            labels_for_case.append({
                "id_prod": f"prod_{idx+1}",
                "pref": pref_score,
                "conf": 1.0,
                "reason": reason,
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
    print(f"Generating {N_SCENARIOS} architectural finish control scenarios...")
    dataset, labels = generate_archfinish_control_dataset(N_SCENARIOS)

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
        print(f"  {alt['id_prod']} | circ_orig={alt['circ_orig']}% | d_max={alt['d_max']} mm -> pref={lab['pref']}")
