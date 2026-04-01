import json
import random
import os

# ─────────────── CONFIGURATION ───────────────

output_dir = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_FILE = os.path.join(output_dir, "archfinish_slump_control_scenarios.json")
LABELS_FILE    = os.path.join(output_dir, "archfinish_slump_control_labels.json")

N_SCENARIOS = 3000
MIN_ALTS = 2
MAX_ALTS = 5

PREF_MAX = 1.0
PREF_MIN = 0.60

# Slump control range (mm) — from stiff/unusable to highly fluid
SLUMP_MIN = 10    # very stiff, poor workability, poor surface finish
SLUMP_MAX = 260   # highly fluid, excellent workability, polished surfaces

# Fixed near-ideal ranges for the variables kept constant in this control
CIRC_IDEAL_MIN = 0
CIRC_IDEAL_MAX = 10   # low circular origin → more 'pure' mix → good for finish

DMAX_IDEAL_MIN = 4    # mm — small aggregate → smooth finish
DMAX_IDEAL_MAX = 12    # mm

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

# ─────────────── BASE ATTRIBUTES  ───────────────

def get_ideal_attributes_except_slump():
    """
    All indicators near-ideal. circ_orig, d_max, and slump are injected
    externally so this function does NOT set them.
    """
    health = random.randint(4, 6)

    fu_recyc  = random.randint(95, 100)
    remaining = 100 - fu_recyc
    fu_incin  = 0
    fu_haz    = 0
    fu_inert  = remaining

    gwp = random.uniform(0.05, 0.10)
    wdp = random.uniform(0.03, 0.05)
    fwu = random.uniform(0.0005, 0.001)
    b   = random.uniform(0.0, 0.1)

    c_p = random.uniform(0.05, 0.08)
    c_w = random.uniform(0.01, 0.02)
    c_m = random.uniform(0.01, 0.02)

    compressive_strength = round(random.uniform(40, 60), 1)
    w_c_ratio            = round(random.uniform(0.30, 0.40), 3)
    cement_content       = int(random.uniform(340, 380))
    scm_content          = int(random.uniform(150, 180))
    density              = int(random.uniform(2300, 2500))

    return {
        "health":    health,
        "fu_recyc":  fu_recyc,
        "fu_incin":  fu_incin,
        "fu_inert":  fu_inert,
        "fu_haz":    fu_haz,
        "gwp":       round(gwp, 3),
        "wdp":       round(wdp, 4),
        "fwu":       round(fwu, 5),
        "b":         round(b, 3),
        "c": {
            "c_p": round(c_p, 3),
            "c_w": round(c_w, 3),
            "c_m": round(c_m, 3),
        },
        "compressive_strength":  compressive_strength,
        "water_to_cement_ratio": w_c_ratio,
        "cement_content":        cement_content,
        "SCM_content":           scm_content,
        "density":               density,
    }

# ─────────────── SAMPLER FOR SLUMP (CONTROL VARIABLE) ───────────────

def sample_slump_for_scenario(n_alts: int):
    """
    Slump control:
      - one near minimum (10–30 mm)   → stiff mix, poor workability → low preference
      - one near maximum (230–260 mm) → fluid mix, excellent finish  → high preference
      - remaining alternatives: random in [SLUMP_MIN, SLUMP_MAX]
    Values are shuffled so position does not encode rank.
    """
    vals = []

    near_min = random.uniform(SLUMP_MIN, SLUMP_MIN + 20)
    vals.append(round(near_min, 1))

    if n_alts > 1:
        near_max = random.uniform(SLUMP_MAX - 30, SLUMP_MAX)
        vals.append(round(near_max, 1))

    for _ in range(2, n_alts):
        vals.append(round(random.uniform(SLUMP_MIN, SLUMP_MAX), 1))

    random.shuffle(vals)
    return vals

# ─────────────── CONTROL DATASET GENERATION ───────────────

def generate_archfinish_slump_control_dataset(num_scenarios: int):
    scenarios  = []
    all_labels = []

    for i in range(1, num_scenarios + 1):
        case_id     = f"control_archfinish_slump_{i}"
        stakeholder = random.choice(STAKEHOLDER_PREFS)
        situation   = "Architectural finish"
        n_alts      = random.randint(MIN_ALTS, MAX_ALTS)

        slump_values = sample_slump_for_scenario(n_alts)

        # circ_orig and d_max are fixed near ideal throughout
        circ_values = [int(round(random.uniform(CIRC_IDEAL_MIN, CIRC_IDEAL_MAX))) for _ in range(n_alts)]
        dmax_values = [round(random.uniform(DMAX_IDEAL_MIN, DMAX_IDEAL_MAX), 1)  for _ in range(n_alts)]

        alternatives    = []
        labels_for_case = []

        for idx in range(n_alts):
            slump_val = slump_values[idx]
            circ_val  = circ_values[idx]
            dmax_val  = dmax_values[idx]

            attrs = get_ideal_attributes_except_slump()

            alt = {
                "id_prod":               f"prod_{idx+1}",
                "circ_orig":             circ_val,
                "fu_recyc":              attrs["fu_recyc"],
                "fu_incin":              attrs["fu_incin"],
                "fu_inert":              attrs["fu_inert"],
                "fu_haz":                attrs["fu_haz"],
                "health":                attrs["health"],
                "gwp":                   attrs["gwp"],
                "wdp":                   attrs["wdp"],
                "fwu":                   attrs["fwu"],
                "b":                     attrs["b"],
                "c":                     attrs["c"],
                "compressive_strength":  attrs["compressive_strength"],
                "slump":                 slump_val,
                "water_to_cement_ratio": attrs["water_to_cement_ratio"],
                "cement_content":        attrs["cement_content"],
                "SCM_content":           attrs["SCM_content"],
                "density":               attrs["density"],
                "d_max":                 dmax_val,
            }
            alternatives.append(alt)

            # Preference: higher slump → higher workability → better surface finish → higher preference
            span      = SLUMP_MAX - SLUMP_MIN
            norm      = (slump_val - SLUMP_MIN) / span if span > 0 else 0.5
            base_score = PREF_MIN + (PREF_MAX - PREF_MIN) * norm

            score_noise = random.uniform(-0.01, 0.01)
            pref_score  = round(min(1.0, base_score + score_noise), 3)

            labels_for_case.append({
                "id_prod": f"prod_{idx+1}",
                "pref":    pref_score,
                "conf":    1.0,
                "reason":  (
                    f"Architectural finish control (slump): slump={slump_val} mm, "
                    f"base score={round(base_score, 3)} + noise. "
                    f"circ_orig={circ_val}% (near-ideal), d_max={dmax_val} mm (near-ideal)."
                ),
            })

        scenario_obj = {
            "id":                     case_id,
            "stakeholder_preference": [stakeholder],
            "situations":             [situation],
            "alternatives":           alternatives,
        }

        scenarios.append(scenario_obj)
        all_labels.append({"id": case_id, "labelled_alternatives": labels_for_case})

    return scenarios, all_labels


if __name__ == "__main__":
    print(f"Generating {N_SCENARIOS} architectural finish slump control scenarios...")
    dataset, labels = generate_archfinish_slump_control_dataset(N_SCENARIOS)

    with open(SCENARIOS_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)

    print("Done.")
    print(f"Scenarios saved to: {SCENARIOS_FILE}")
    print(f"Labels saved to:    {LABELS_FILE}")

    s0 = dataset[0]
    l0 = labels[0]
    print(f"\nPreview case: {s0['id']} | situation={s0['situations'][0]}")
    for alt, lab in zip(s0["alternatives"], l0["labelled_alternatives"]):
        print(
            f"  {alt['id_prod']} | circ_orig={alt['circ_orig']}% "
            f"| d_max={alt['d_max']} mm | slump={alt['slump']} mm "
            f"-> pref={lab['pref']}"
        )