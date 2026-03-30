import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9
from data.config import (
    STAKEHOLDER_PREFS, 
    STAKEHOLDER_PREFS as STAKEHOLDERS,
    SCENARIO_PREFS, 
    PERFORMANCE_INDICATOR_RELEVANCE_MAPPING,
    PERFORMANCE_INDICATOR_RELEVANCE_MAPPING as PERF_MAP 
)
# --- project imports ---
from data.config import FROZEN_PATH, HIDDEN_DIM, DROPOUT
from data.loader import encode_stakeholder_pref, encode_scenario_pref, _extract_features_from_alt
from model.architecture import SetRanker

# --------------------
# Settings
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE_DIR, "stored_models", "ranking_model.pt")
EVALUATOR_JSON = os.path.join(BASE_DIR, "evaluator_dataset.json")

# --------------------
# Load frozen dataset to compute scaling
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
feature_keys = sorted(feature_keys_set)
final_feature_keys = feature_keys + ['total_cost']

all_values = []
all_present = []
all_relevant = []
for sc in frozen_raw:
    situations = sc.get('situations', []) or []
    from data.config import PERFORMANCE_INDICATOR_RELEVANCE_MAPPING as PERF_MAP
    relevant_indicators = set()
    for sit in situations:
        relevant_indicators.update(PERF_MAP.get(sit, []))
    for alt in sc.get('alternatives', []):
        feats = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
        values = feats[0::3]
        present = feats[1::3]
        relevant = feats[2::3]
        all_values.append(values)
        all_present.append(present)
        all_relevant.append(relevant)

all_values = np.array(all_values, dtype=np.float32)
all_present = np.array(all_present, dtype=np.int8)
all_relevant = np.array(all_relevant, dtype=np.int8)

col_min = []
col_max = []
n_cols = all_values.shape[1]
for i in range(n_cols):
    valid_mask = (all_present[:, i] == 1) & (all_relevant[:, i] == 1)
    vals = all_values[valid_mask, i]
    if vals.size == 0:
        col_min.append(0.0)
        col_max.append(0.0)
    else:
        col_min.append(float(vals.min()))
        col_max.append(float(vals.max()))

# --------------------
# Load evaluator dataset
# --------------------
with open(EVALUATOR_JSON, 'r', encoding='utf-8') as f:
    evaluator_raw = json.load(f)

scenarios = []
for sc in evaluator_raw:
    # Encode stakeholders and scenario once per scenario
    stakeholder_vec = encode_stakeholder_pref(sc.get('stakeholder_preference', []))
    scenario_vec = encode_scenario_pref(sc.get('scenario_preference', []))
    situations = sc.get('situations', []) or []
    from data.config import PERFORMANCE_INDICATOR_RELEVANCE_MAPPING as PERF_MAP
    relevant_indicators = set()
    for sit in situations:
        relevant_indicators.update(PERF_MAP.get(sit, []))
    feats_scaled = []
    for alt in sc.get('alternatives', []):
        feats = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
        values = np.array(feats[0::3], dtype=np.float32)
        present = np.array(feats[1::3], dtype=np.int8)
        relevant = np.array(feats[2::3], dtype=np.int8)
        scaled = np.zeros_like(values)
        for i in range(len(values)):
            if present[i] == 1 and relevant[i] == 1:
                if col_max[i] > col_min[i]:
                    scaled[i] = (values[i] - col_min[i]) / (col_max[i] - col_min[i])
                else:
                    scaled[i] = 0.0
            else:
                scaled[i] = 0.0
        combined = []
        for v, p, r in zip(scaled, present, relevant):
            combined.extend([float(v), int(p), int(r)])
        combined.extend(stakeholder_vec.tolist())
        combined.extend(scenario_vec.tolist())
        feats_scaled.append(combined)
    scenarios.append({
        'id': sc.get('id'),
        'features': np.array(feats_scaled, dtype=np.float32),
        'raw_alternatives': sc.get('alternatives', []),  # keep original for perturbation
        'feature_keys': final_feature_keys,
        'stakeholder_vec': stakeholder_vec,
        'scenario_vec': scenario_vec,
        'relevant_indicators': relevant_indicators
    })

if len(scenarios) < 1:
    raise ValueError("No scenarios to evaluate.")

INPUT_DIM = scenarios[0]['features'].shape[1]

# --------------------
# Load model
# --------------------
scenario_vector_size = len(STAKEHOLDER_PREFS) + len(SCENARIO_PREFS)

model = SetRanker(
    feat_dim=INPUT_DIM, 
    scenario_dim=scenario_vector_size,  
    hidden_dims=HIDDEN_DIM, 
    dropout=DROPOUT
).to(DEVICE)

state = torch.load(WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# --------------------
# Robustness evaluation and boxplots
# --------------------
PERT_SAMPLES = 5000  # number of perturbations per alternative
PRODUCT_LABELS = ['A', 'B', 'C', 'D', 'E']


def perturb_feature(value, feature_name, minval, maxval):
    # ±10% perturbation for continuous features, clipped to min/max
    if feature_name in ['circ_orig', 'fu_recyc', 'fu_incin', 'fu_inert', 'fu_haz']:
        pert = value * np.random.uniform(0.9, 1.1)
        return np.clip(pert, minval, maxval)
    if feature_name == 'health':
        pert = int(round(value + np.random.randint(-1, 1)))
        return np.clip(pert, minval, maxval)
    if feature_name in ['wdp', 'fwu', 'c_p', 'c_w', 'c_m', 'gwp']:
        spread = max(0.001, abs(value) * 0.1)
        pert = value + np.random.uniform(-spread, spread)
        return np.clip(pert, minval, maxval)
    return value  


num_scenarios = len(scenarios)
ncols = math.ceil(num_scenarios / 2)
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharey=True)
axes = axes.flatten()
subplot_titles = ["(a)", "(b)", "(c)", "(d)"]

with torch.no_grad():
    for s_idx, sc in enumerate(scenarios):
        alt_scores = [[] for _ in range(len(sc['features']))]
        stakeholder_vec = sc['stakeholder_vec']
        scenario_vec = sc['scenario_vec']
        relevant_indicators = sc['relevant_indicators']

        for _ in range(PERT_SAMPLES):
            pert_feats = []
            for a_idx, alt_raw in enumerate(sc['raw_alternatives']):
                alt_vals = alt_raw.copy()
                # Evaluate present and relevant flags only once per original alternative
                feats_orig = _extract_features_from_alt(alt_raw, feature_keys, relevant_indicators)
                present = np.array(feats_orig[1::3], dtype=np.int8)
                relevant = np.array(feats_orig[2::3], dtype=np.int8)
                # Perturb only present and relevant features per feature_keys order
                for idx, feat in enumerate(feature_keys):
                    minval = col_min[idx]
                    maxval = col_max[idx]
                    if feat in alt_vals and present[idx] == 1 and relevant[idx] == 1:
                        alt_vals[feat] = perturb_feature(alt_vals[feat], feat, minval, maxval)
                # Extract features from perturbed alternative
                feats = _extract_features_from_alt(alt_vals, feature_keys, relevant_indicators)
                values = np.array(feats[0::3], dtype=np.float32)
                present2 = np.array(feats[1::3], dtype=np.int8)
                relevant2 = np.array(feats[2::3], dtype=np.int8)
                # Clip values before scaling
                values = np.array([
                    np.clip(v, col_min[i], col_max[i]) if present2[i] == 1 and relevant2[i] == 1 else v
                    for i, v in enumerate(values)
                ], dtype=np.float32)
                scaled = np.zeros_like(values)
                for i in range(len(values)):
                    if present2[i] == 1 and relevant2[i] == 1:
                        if col_max[i] > col_min[i]:
                            scaled[i] = (values[i] - col_min[i]) / (col_max[i] - col_min[i])
                        else:
                            scaled[i] = 0.0
                    else:
                        scaled[i] = 0.0
                combined = []
                for v, p, r in zip(scaled, present2, relevant2):
                    combined.extend([float(v), int(p), int(r)])
                combined.extend(stakeholder_vec.tolist())
                combined.extend(scenario_vec.tolist())
                pert_feats.append(combined)
            X_pert = torch.tensor(np.array([pert_feats], dtype=np.float32)).to(DEVICE)
            mask_pert = torch.ones(1, len(pert_feats), dtype=torch.bool, device=DEVICE)
            scores = model(X_pert, mask_pert).squeeze(0).cpu().numpy()
            for i, s in enumerate(scores):
                alt_scores[i].append(s)
        cmap = cm.get_cmap('Blues', 3)
        median_color = cmap(1)
        bp = axes[s_idx].boxplot(alt_scores, patch_artist=True, tick_labels=PRODUCT_LABELS[:len(alt_scores)])
        for box in bp['boxes']:
            box.set(facecolor='white')
        for median_line in bp['medians']:
            median_line.set_color(median_color)
            median_line.set_linewidth(1)
        axes[s_idx].set_title(subplot_titles[s_idx], fontname="Times New Roman", fontsize=9, fontweight="bold")
        axes[s_idx].set_xlabel("Product", fontname="Times New Roman", fontsize=9)
        axes[s_idx].set_ylabel("Preference score", fontname="Times New Roman", fontsize=9)
    for i in range(num_scenarios, nrows * ncols):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()





# ---------------------------------------------------------------------
# Stakeholder sensitivity analysis 
# ---------------------------------------------------------------------

from itertools import combinations
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Choose the scenario to analyse 
scenario_to_analyse = scenarios[3]

# Prepare list of stakeholder sets
stakeholder_sets = [[s] for s in STAKEHOLDERS]  # + list(combinations(STAKEHOLDERS, 2))

scores_by_pref = []
labels = []

with torch.no_grad():
    for prefs in stakeholder_sets:
        sh_vec = encode_stakeholder_pref(list(prefs)).tolist()
        sc_vec = encode_scenario_pref([]).tolist()
        feats_scaled = []
        for alt in scenario_to_analyse["raw_alternatives"]:
            feats = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
            values = np.array(feats[0::3], dtype=np.float32)
            present = np.array(feats[1::3], dtype=np.int8)
            relevant = np.array(feats[2::3], dtype=np.int8)
            scaled = np.zeros_like(values)
            for i in range(len(values)):
                if present[i] == 1 and relevant[i] == 1:
                    if col_max[i] > col_min[i]:
                        scaled[i] = (values[i] - col_min[i]) / (col_max[i] - col_min[i])
            combined = []
            for v, p, r in zip(scaled, present, relevant):
                combined.extend([float(v), int(p), int(r)])
            combined.extend(sh_vec)
            combined.extend(sc_vec)
            feats_scaled.append(combined)
        X = torch.tensor(np.array([feats_scaled], dtype=np.float32)).to(DEVICE)
        mask = torch.ones(1, len(feats_scaled), dtype=torch.bool, device=DEVICE)
        out = model(X, mask).squeeze(0).cpu().numpy()
        scores_by_pref.append(out)

        # Shorten long names for x-axis
        if len(prefs) == 1:
            labels.append(f"S{STAKEHOLDERS.index(prefs[0]) + 1}")
        else:
            i1 = STAKEHOLDERS.index(prefs[0]) + 1
            i2 = STAKEHOLDERS.index(prefs[1]) + 1
            labels.append(f"S{i1}&S{i2}")

scores_by_pref = np.array(scores_by_pref)  # shape: [n_prefs, n_alternatives]
# ---- Plot grouped bars ----
fig2, ax2 = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 4))

bar_width = 0.15
x = np.arange(len(labels))

cmap = cm.get_cmap("Blues", scores_by_pref.shape[1] + 2)
colours = [cmap(i + 1) for i in range(scores_by_pref.shape[1])]

for i, alt_label in enumerate(PRODUCT_LABELS[:scores_by_pref.shape[1]]):
    ax2.bar(
        x + i * bar_width,
        scores_by_pref[:, i],
        width=bar_width,
        label=f"Product {alt_label}",
        color=colours[i],
        edgecolor="black",
        linewidth=0.3,
    )

ax2.set_xticks(x + bar_width * (scores_by_pref.shape[1] - 1) / 2)
ax2.set_xticklabels(labels, rotation=0, ha="right",
                    fontname="Times New Roman", fontsize=9)
ax2.set_ylabel("Preference score", fontname="Times New Roman", fontsize=9)
ax2.set_xlabel("Stakeholder priority",
               fontname="Times New Roman", fontsize=9)


ax2.legend(
    frameon=False, fontsize=8,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout(rect=[0, 0, 0.9, 1])  
plt.show()




# ---------------------------------------------------------------------
#  Concrete situation sensitivity analysis
# ---------------------------------------------------------------------
import seaborn as sns


# Choose a fixed stakeholder preference for this analysis
STAKEHOLDER_PREF = ["Balanced optimizer: tends to seek well-rounded performance across all indicators without extreme trade-offs."]

# ---------------------------------------------------------------------
# Concrete situation sensitivity analysis 
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
#  Analysis Loop
# ---------------------------------------------------------------------

# 1. Set model to evaluation mode
model.eval() 

scores_matrix = np.zeros((len(SCENARIO_PREFS), len(scenario_to_analyse["raw_alternatives"])))

with torch.no_grad():
    for idx, scenario_name in enumerate(SCENARIO_PREFS):
        
        # Encode scenario (one-hot)
        sc_vec = encode_scenario_pref([scenario_name], all_possible_prefs=SCENARIO_PREFS).tolist()
        sh_vec = encode_stakeholder_pref(STAKEHOLDER_PREF).tolist()
        
        
        # Get indicators
        relevant_indicators = set(PERFORMANCE_INDICATOR_RELEVANCE_MAPPING.get(scenario_name, []))

        feats_scaled = []
        for alt in scenario_to_analyse["raw_alternatives"]:
            feats = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
            
            values = np.array(feats[0::3], dtype=np.float32)
            present = np.array(feats[1::3], dtype=np.int8)
            relevant = np.array(feats[2::3], dtype=np.int8)
            
            scaled = np.zeros_like(values)
            for i in range(len(values)):
                if present[i] == 1 and relevant[i] == 1:
                    if col_max[i] > col_min[i]:
                        scaled[i] = (values[i] - col_min[i]) / (col_max[i] - col_min[i])
            
            combined = []
            for v, p, r in zip(scaled, present, relevant):
                combined.extend([float(v), int(p), int(r)])
            
            # Append vectors
            combined.extend(sh_vec)
            combined.extend(sc_vec)
            feats_scaled.append(combined)

        # Convert to tensor
        X = torch.tensor(np.array([feats_scaled], dtype=np.float32)).to(DEVICE)
        mask = torch.ones(1, len(feats_scaled), dtype=torch.bool, device=DEVICE)
        
        # Run model
        out = model(X, mask).squeeze(0).cpu().numpy()
        scores_matrix[idx, :] = out


# ---------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.heatmap(
    scores_matrix,
    annot=True,
    fmt=".2f",
    yticklabels=SCENARIO_PREFS,
    xticklabels=PRODUCT_LABELS[:scores_matrix.shape[1]],
    cmap="Blues",
    cbar_kws={"label": "Preference score"}
)
plt.xlabel("Product", fontname="Times New Roman", fontsize=9)
plt.ylabel("Concrete Situation", fontname="Times New Roman", fontsize=9)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Inspect the first layer weights
first_layer = model.input_encoder[0] # Linear(Input_Dim -> 128)
weights = first_layer.weight.data.cpu().numpy() # Shape [128, Input_Dim]

# Calculate average magnitude of weights for Features vs Scenario
scenario_weights = np.abs(weights[:, -4:]).mean()
feature_weights = np.abs(weights[:, :-4]).mean()

print(f"Average Feature Weight Magnitude: {feature_weights:.6f}")
print(f"Average Scenario Weight Magnitude: {scenario_weights:.6f}")



if scenario_weights < 1e-4:
    print("CRITICAL: The model has effectively zeroed out the scenario inputs!") #debug print, this should not happen with proper training and regularization

stakeholder_weights = np.abs(weights[:, -12:-4]).mean()
print(f"Average Stakeholder Weight: {stakeholder_weights:.6f}")



# ---------------------------------------------------------------------
# FULL FINAL SHAP SCRIPT 
# ---------------------------------------------------------------------
import shap
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 0. MODEL WRAPPER
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_predict(data_numpy):
    X_tensor = torch.tensor(data_numpy, dtype=torch.float32).to(DEVICE)
    X_unsqueezed = X_tensor.unsqueeze(1)
    mask = torch.ones(X_tensor.shape[0], 1, dtype=torch.bool, device=DEVICE)
    
    model.eval()
    with torch.no_grad():
        scores = model(X_unsqueezed, mask)
    
    if scores.dim() > 1:
        scores = scores.squeeze(-1)
        
    return scores.cpu().numpy()

# ---------------------------------------------------------------------
# 1. FEATURE NAMES
# ---------------------------------------------------------------------
NAME_MAP = {
    "circ_orig": "Circular origin percentage",
    "fu_recyc": "Future use - recycling",
    "fu_incin": "Future use - incineration",
    "fu_inert": "Future use - inert landfilling",
    "fu_haz": "Future use - hazardous waste",
    "health": "Health scoring",
    "gwp": "Global warming potential",
    "wdp": "Water depletion potential",
    "fwu": "Freshwater use",
    "b": "Biodiversity scoring",
    "total_cost": "Total costs",
    "compressive_strength": "Compressive strength",
    "slump": "Consistency - slump",
    "water_to_cement_ratio": "Water to cement ratio",
    "cement_content": "Cement content",
    "SCM_content": "SCM content",
    "density": "Density",
    "d_max": "Max aggregate size"
}

shap_feature_names = []
for key in final_feature_keys:
    base_name = NAME_MAP.get(key, key)
    shap_feature_names.extend([f"{base_name} (Value)", f"{base_name} (Present)", f"{base_name} (Relevant)"])
for s in STAKEHOLDERS:
    shap_feature_names.append(f"SH: {s}")
for s in SCENARIO_PREFS:
    shap_feature_names.append(f"SC: {s}")

# ---------------------------------------------------------------------
# 2. HELPER: EXTRACT DATA SLICE 
# ---------------------------------------------------------------------
def get_data_slice(target_sh_idx, target_sc_idx):
    target_sh_text = STAKEHOLDER_PREFS[target_sh_idx]
    target_sc_text = SCENARIO_PREFS[target_sc_idx]
    
    print(f"\nFiltering for:\n  SH: '{target_sh_text[:30]}...'\n  SC: '{target_sc_text}'...")
    
    extracted_data = []
    
    for sc in frozen_raw:
        # Check situation
        sc_sits = [s.strip().lower() for s in sc.get('situations', [])]
        if target_sc_text.strip().lower() not in sc_sits:
            continue

        # Check stakeholder
        sc_shs = [s.strip().lower() for s in sc.get('stakeholder_preference', [])]
        if target_sh_text.strip().lower() not in sc_shs:
            continue

        # Process alternatives (control + non-control)
        stakeholder_vec = encode_stakeholder_pref(sc.get('stakeholder_preference', []))
        scenario_vec = encode_scenario_pref(sc.get('scenario_preference', []))
        
        relevant_indicators = set()
        for sit in sc.get('situations', []):
             relevant_indicators.update(PERF_MAP.get(sit, []))
        
        for alt in sc.get('alternatives', []):
            
            feats = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
            values = np.array(feats[0::3], dtype=np.float32)
            present = np.array(feats[1::3], dtype=np.int8)
            relevant = np.array(feats[2::3], dtype=np.int8)

            scaled = np.zeros_like(values)
            for i in range(len(values)):
                if present[i] == 1 and relevant[i] == 1:
                    if col_max[i] > col_min[i]:
                        scaled[i] = (values[i] - col_min[i]) / (col_max[i] - col_min[i])
            
            combined = []
            for v, p, r in zip(scaled, present, relevant):
                combined.extend([float(v), int(p), int(r)])
            
            combined.extend(stakeholder_vec.tolist())
            combined.extend(scenario_vec.tolist())
            
            extracted_data.append(combined)
            
    return np.array(extracted_data, dtype=np.float32)

# ---------------------------------------------------------------------
# 3. HELPER: RUN SHAP & PLOT 
# ---------------------------------------------------------------------
def run_shap_on_slice(slice_name, target_sh_idx, target_sc_idx):
    X_slice = get_data_slice(target_sh_idx, target_sc_idx)
    
    if len(X_slice) < 10:
        print(f"⚠️ Skipping {slice_name}: Found {len(X_slice)} samples.") #If the model was trained appropriately, it should have enough data to compute SHAP values even for specific slices, but we set a minimum threshold to avoid unreliable explanations.
        return

    print(f"  -> Found {len(X_slice)} samples. Running SHAP...")

    X_bg_summary = shap.kmeans(X_slice, 50)
    n_explain = min(len(X_slice), 50)
    indices = np.random.choice(len(X_slice), n_explain, replace=False)
    X_explain_slice = X_slice[indices]

    explainer = shap.KernelExplainer(model_predict, X_bg_summary)
    shap_values = explainer.shap_values(X_explain_slice, nsamples=500, silent=True)
    if isinstance(shap_values, list): shap_values = shap_values[0]

    # --- FILTER FEATURES FOR PLOT ---
    keep_indices = []
    clean_names = []
    
    for i, name in enumerate(shap_feature_names):
        # Hide context flags (SH/SC)
        if name.startswith("SH:") or name.startswith("SC:"):
            continue
        keep_indices.append(i)
        clean_names.append(name)
    
    shap_values_plot = shap_values[:, keep_indices]
    X_explain_plot   = X_explain_slice[:, keep_indices]

    # Plot
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_plot, 
        X_explain_plot, 
        feature_names=clean_names,
        max_display=15, 
        plot_type="dot",
        show=False,
        plot_size=None
    )
    ax = plt.gca()
    ax.set_xlim(left=-0.2)
    ax.set_title(f"Drivers for: {slice_name}", fontname="Times New Roman", fontsize=14)
    ax.set_xlabel("Impact on Preference Score", fontname="Times New Roman", fontsize=12)
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(10)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 4. EXECUTE SCENARIOS (some examples, can be extended to all combinations of interest)
# ---------------------------------------------------------------------
# 1. Cost-Conscious (1) + Standard (0)
run_shap_on_slice("Cost-Conscious (standard application)", 1, 0)
# 2. Sustainability (0) + Standard (0)
#run_shap_on_slice("Sustainability (standard application)", 0, 0)
# 3. Balanced (6) + Standard (0)
#run_shap_on_slice("Balanced (standard application)", 6, 0)
# 3. Regulatory (5) + Standard (0)
#run_shap_on_slice("Regulatory (standard application)", 5, 0)

# 1. Cost-Conscious (1) + Architectural (3)
run_shap_on_slice("Cost-Conscious (Arch)", 1, 3)

# 2. Sustainability (0) + Architectural (3)
#run_shap_on_slice("Sustainability (Arch)", 0, 3)

# 3. Balanced (6) + Acoustic (1)
run_shap_on_slice("Balanced (Acoustic)", 6, 1)

# 4. Balanced (6) + Thermal (2)
run_shap_on_slice("Balanced (Thermal)", 6, 2)
