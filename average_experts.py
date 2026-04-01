import json, os, glob
from collections import defaultdict

home = os.path.dirname(os.path.abspath(__file__))
all_json = glob.glob(os.path.join(home, '**', '*.json'), recursive=True)

validation_file = next((f for f in all_json if 'expert_validation_dataset' in f and 'label' not in f), None)
labels_file = next((f for f in all_json if 'expert_validation_labels' in f), None)

with open(validation_file) as f:
    validation_data = json.load(f)

with open(labels_file) as f:
    labels_data = json.load(f)

target_pref = "Balanced optimizer"
target_situation = "Standard structural application"

target_ids = [
    v["id"] for v in validation_data
    if any(target_pref in p for p in v["stakeholder_preference"])
    and any(target_situation in s for s in v["situations"])
]

pref_by_prod = defaultdict(list)
conf_by_prod = defaultdict(list)

for vid in target_ids:
    labels = next(l for l in labels_data if l["id"] == vid)
    for a in labels["labelled_alternatives"]:
        pref_by_prod[a["id_prod"]].append(a["pref"])
        conf_by_prod[a["id_prod"]].append(a["conf"])

print(f"{'Product':<10} {'Mean pref':>10} {'Mean conf':>10}")
for prod in sorted(pref_by_prod):
    mp = sum(pref_by_prod[prod]) / len(pref_by_prod[prod])
    mc = sum(conf_by_prod[prod]) / len(conf_by_prod[prod])
    print(f"{prod:<10} {mp:>10.4f} {mc:>10.4f}")