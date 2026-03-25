import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9

from data.config import (
    STAKEHOLDER_PREFS,
    SCENARIO_PREFS,
    PERFORMANCE_INDICATOR_RELEVANCE_MAPPING as PERF_MAP,
    FROZEN_PATH,
    HIDDEN_DIM,
    DROPOUT
)
from data.loader import encode_stakeholder_pref, encode_scenario_pref, _extract_features_from_alt
from model.architecture import SetRanker

# --------------------
# Settings
# --------------------
DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
WEIGHTS               = os.path.join(BASE_DIR, "stored_models", "ranking_model.pt")
EXPERT_SCENARIOS_PATH = os.path.join(BASE_DIR, "expert_validation_dataset.json")
EXPERT_LABELS_PATH    = os.path.join(BASE_DIR, "expert_validation_labels_dataset.json")
PRODUCT_LABELS        = ['A', 'B', 'C', 'D', 'E']
N_PRODUCTS            = 5

# --------------------
# Load frozen dataset for scaling
# --------------------
with open(FROZEN_PATH, 'r', encoding='utf-8') as f:
    frozen_raw = json.load(f)

feature_keys_set = set()
for sc in frozen_raw:
    for alt in sc.get('alternatives', []):
        for k in alt.keys():
            if k in ('id_prod', 'c', 'wb'):
                continue
            feature_keys_set.add(k)
feature_keys       = sorted(feature_keys_set)
final_feature_keys = feature_keys + ['total_cost']

all_values, all_present, all_relevant = [], [], []
for sc in frozen_raw:
    situations = sc.get('situations', []) or []
    relevant_indicators = set()
    for sit in situations:
        relevant_indicators.update(PERF_MAP.get(sit, []))
    for alt in sc.get('alternatives', []):
        feats = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
        all_values.append(feats[0::3])
        all_present.append(feats[1::3])
        all_relevant.append(feats[2::3])

all_values   = np.array(all_values,   dtype=np.float32)
all_present  = np.array(all_present,  dtype=np.int8)
all_relevant = np.array(all_relevant, dtype=np.int8)

n_cols  = all_values.shape[1]
col_min = []
col_max = []
for i in range(n_cols):
    valid_mask = (all_present[:, i] == 1) & (all_relevant[:, i] == 1)
    vals = all_values[valid_mask, i]
    col_min.append(float(vals.min()) if vals.size > 0 else 0.0)
    col_max.append(float(vals.max()) if vals.size > 0 else 0.0)

# --------------------
# Load model
# --------------------
scenario_vector_size = len(STAKEHOLDER_PREFS) + len(SCENARIO_PREFS)

_sample_relevant = set()
_sample_feats    = _extract_features_from_alt(
    frozen_raw[0]['alternatives'][0], feature_keys, _sample_relevant
)
_sample_stk = encode_stakeholder_pref([STAKEHOLDER_PREFS[0]])
_sample_sc  = encode_scenario_pref([SCENARIO_PREFS[0]])
INPUT_DIM   = len(_sample_feats) + len(_sample_stk) + len(_sample_sc)

model = SetRanker(
    feat_dim     = INPUT_DIM,
    scenario_dim = scenario_vector_size,
    hidden_dims  = HIDDEN_DIM,
    dropout      = DROPOUT
).to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# --------------------
# Helper: encode and scale one alternative
# --------------------
def encode_alt(alt, relevant_indicators, stakeholder_vec, scenario_vec):
    feats    = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
    values   = np.array(feats[0::3], dtype=np.float32)
    present  = np.array(feats[1::3], dtype=np.int8)
    relevant = np.array(feats[2::3], dtype=np.int8)
    scaled   = np.zeros_like(values)
    for i in range(len(values)):
        if present[i] == 1 and relevant[i] == 1 and col_max[i] > col_min[i]:
            scaled[i] = (values[i] - col_min[i]) / (col_max[i] - col_min[i])
    combined = []
    for v, p, r in zip(scaled, present, relevant):
        combined.extend([float(v), int(p), int(r)])
    combined.extend(stakeholder_vec.tolist())
    combined.extend(scenario_vec.tolist())
    return combined

# --------------------
# Load expert scenarios and labels
# --------------------
with open(EXPERT_SCENARIOS_PATH, 'r', encoding='utf-8') as f:
    expert_scenarios = json.load(f)

with open(EXPERT_LABELS_PATH, 'r', encoding='utf-8') as f:
    expert_labels_raw = json.load(f)

expert_labels_lookup = {}
for entry in expert_labels_raw:
    vid   = entry['id']
    prods = {alt['id_prod']: alt['pref'] for alt in entry['labelled_alternatives']}
    expert_labels_lookup[vid] = prods

# --------------------
# Collect model and expert scores per product
# --------------------
model_scores  = [[] for _ in range(N_PRODUCTS)]
expert_scores = [[] for _ in range(N_PRODUCTS)]

with torch.no_grad():
    for sc in expert_scenarios:
        vid        = sc['id']
        situations = sc.get('situations', []) or []

        relevant_indicators = set()
        for sit in situations:
            relevant_indicators.update(PERF_MAP.get(sit, []))

        stakeholder_vec = encode_stakeholder_pref(sc.get('stakeholder_preference', []))
        scenario_vec    = encode_scenario_pref(sc.get('situations', []))

        alts         = sc.get('alternatives', [])
        feats_scaled = [
            encode_alt(alt, relevant_indicators, stakeholder_vec, scenario_vec)
            for alt in alts
        ]

        X      = torch.tensor(np.array([feats_scaled], dtype=np.float32)).to(DEVICE)
        mask   = torch.ones(1, len(feats_scaled), dtype=torch.bool, device=DEVICE)
        scores = model(X, mask).squeeze(0).cpu().numpy()

        for i, alt in enumerate(alts):
            prod_id  = alt.get('id_prod', f'prod_{i+1}')
            prod_idx = int(prod_id.split('_')[1]) - 1
            if prod_idx < N_PRODUCTS:
                model_scores[prod_idx].append(float(scores[i]))

        if vid in expert_labels_lookup:
            for prod_id, pref in expert_labels_lookup[vid].items():
                prod_idx = int(prod_id.split('_')[1]) - 1
                if prod_idx < N_PRODUCTS:
                    expert_scores[prod_idx].append(float(pref))

# --------------------
# Sanity check
# --------------------
print(f"{'Product':<10} {'Model scores':<15} {'Expert scores':<15} {'Match'}")
print("-" * 45)
for i in range(N_PRODUCTS):
    n_model  = len(model_scores[i])
    n_expert = len(expert_scores[i])
    match    = "OK" if n_model == n_expert else "MISMATCH"
    print(f"{'Prod ' + PRODUCT_LABELS[i]:<10} {n_model:<15} {n_expert:<15} {match}")

print(f"\nTotal validation scenarios loaded: {len(expert_scenarios)}")
print(f"Total expert label entries loaded: {len(expert_labels_raw)}")

# --------------------
# Plot: violin plots
# --------------------
cmap          = cm.get_cmap('Blues', 5)
colour_model  = cmap(2)
colour_expert = cmap(4)

fig, ax = plt.subplots(figsize=(6, 5))

group_width   = 1.0
violin_width  = 0.35
gap           = 0.05
group_centres = np.arange(N_PRODUCTS) * group_width

positions_model  = group_centres - (violin_width / 2 + gap / 2)
positions_expert = group_centres + (violin_width / 2 + gap / 2)

for i in range(N_PRODUCTS):
    for data, pos, colour in [
        (model_scores[i],  positions_model[i],  colour_model),
        (expert_scores[i], positions_expert[i], colour_expert)
    ]:
        parts = ax.violinplot(
            data,
            positions=[pos],
            widths=violin_width,
            showmedians=True,
            showextrema=True
        )
        for pc in parts['bodies']:
            pc.set_facecolor(colour)
            pc.set_edgecolor(colour)
            pc.set_alpha(0.4)
        for partname in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if partname in parts:
                parts[partname].set_edgecolor(colour)
                parts[partname].set_linewidth(0.8)

ax.set_xticks(group_centres)
ax.set_xticklabels(
    [f'Product {p}' for p in PRODUCT_LABELS],
    fontname="Times New Roman", fontsize=9
)
ax.set_ylabel("Preference score", fontname="Times New Roman", fontsize=9)
ax.set_xlabel("Product",          fontname="Times New Roman", fontsize=9)
ax.set_ylim(0, 1.05)

patch_model  = mpatches.Patch(facecolor=colour_model,  alpha=0.4, label='Model output')
patch_expert = mpatches.Patch(facecolor=colour_expert, alpha=0.4, label='Expert output')
ax.legend(handles=[patch_model, patch_expert], frameon=False, fontsize=8, loc='upper right')

plt.tight_layout()
output_path = os.path.join(BASE_DIR, "validator_comparison.png")
plt.savefig(output_path, dpi=300)
plt.show()
print(f"Figure saved to {output_path}")

