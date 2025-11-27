import json
import os

# Base directory where all input JSON files reside and where outputs will be saved
dir_path = os.path.dirname(os.path.abspath(__file__))

# Input control case and label files
control_case_files = [
    "cost_control_scenarios.json", "cost_control_labels.json",
    "fwu_control_scenarios.json", "fwu_control_labels.json",
    "gwp_control_scenarios.json", "gwp_control_labels.json",
    "wdp_control_scenarios.json", "wdp_control_labels.json",
    "expert_scenarios.json", "expert_labels.json"
]

# LLM-generated dataset files
llm_cases_file = "dataset.json"
llm_labels_file = "labelled_dataset_LLM.json"

# Output combined files
FROZEN_PATH = os.path.join(dir_path, 'frozen_dataset.json')
LABELED_PATH = os.path.join(dir_path, 'labelled_dataset.json')

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

all_scenarios = []
all_labelled = []

# Load control cases and labels
for filename in control_case_files:
    path = os.path.join(dir_path, filename)
    if os.path.isfile(path):
        data = load_json(path)
        if 'scenarios' in filename:
            all_scenarios.extend(data)
        elif 'labels' in filename:
            all_labelled.extend(data)

# Load LLM-generated dataset and labels
llm_cases_path = os.path.join(dir_path, llm_cases_file)
llm_labels_path = os.path.join(dir_path, llm_labels_file)

if os.path.isfile(llm_cases_path):
    all_scenarios.extend(load_json(llm_cases_path))

if os.path.isfile(llm_labels_path):
    all_labelled.extend(load_json(llm_labels_path))

# Save combined datasets
with open(FROZEN_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_scenarios, f, indent=2)

with open(LABELED_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_labelled, f, indent=2)

print(f"Combined scenarios saved to: {FROZEN_PATH}")
print(f"Combined labels saved to: {LABELED_PATH}")
