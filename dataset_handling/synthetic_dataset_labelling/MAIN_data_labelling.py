from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import time
import re

load_dotenv()
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


base_dir = os.path.dirname(os.path.abspath(__file__))

input_file_path = os.path.join(base_dir, 'dataset.json')
output_file_path = os.path.join(base_dir, 'labelled_dataset.json')
system_prompt_file_path = os.path.join(base_dir, 'system_prompt.txt')

def load_system_prompt():
    with open(system_prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def clean_api_response(raw_content: str) -> str:
    # remove triple-backtick fences first
    raw = re.sub(r'```(?:json)?\s*', '', raw_content, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.IGNORECASE).strip()
    # try to find a top-level JSON object or array
    m = re.search(r'(\{.*\}|\[.*\])', raw, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        candidate = raw
    #  remove trailing commas that break JSON
    candidate = re.sub(r',\s*([\]\}])', r'\1', candidate)
    return candidate.strip()

def get_scenario_label(scenario):
    try:
        system_content = load_system_prompt()
        prompt = (
            f"Stakeholder Preference: {scenario['stakeholder_preference']}\n"
            f"Contextual situation(s): {scenario['situations']}\n"
            f"Alternatives: {json.dumps(scenario['alternatives'], indent=2)}"
        )

        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        cleaned = clean_api_response(raw_content)

        # Debug print 
        #print("CLEANED JSON:\n", cleaned[:2000])

        parsed = json.loads(cleaned)

        # Normalize parsed to a dict with predictable keys
        if isinstance(parsed, list):
            result = {
                "id": scenario["id"],
                "labelled_alternatives": parsed
            }
        elif isinstance(parsed, dict):
            result = parsed
            result.setdefault("id", scenario["id"])
        else:
            # unexpected type
            raise ValueError("Parsed JSON is neither dict nor list.")

        return result

    except json.JSONDecodeError as jde:
        print(f"JSON decode error on scenario {scenario['id']}: {jde}")
        print("Raw response head:", raw_content[:500])
        return None
    except Exception as e:
        print(f"Error processing scenario {scenario['id']}: {str(e)}")
        return None

def append_to_output(data):
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)

    with open(output_file_path, 'r+', encoding='utf-8') as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

        existing.append(data)
        f.seek(0)
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.truncate()

def get_unprocessed_ids():
    if not os.path.exists(input_file_path):
        print("Missing input file.")
        return []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    if not os.path.exists(output_file_path):
        return [item["id"] for item in input_data]

    with open(output_file_path, 'r', encoding='utf-8') as f:
        try:
            output_data = json.load(f)
            processed_ids = {item["id"] for item in output_data}
        except json.JSONDecodeError:
            processed_ids = set()

    all_ids = {item["id"] for item in input_data}
    return sorted(all_ids - processed_ids, key=lambda x: int(x))

def process_scenarios():
    with open(input_file_path, 'r', encoding='utf-8') as f:
        dataset = {item["id"]: item for item in json.load(f)}

    unprocessed_ids = get_unprocessed_ids()
    count = 0

    for scenario_id in unprocessed_ids:
        start_time = time.time()
        scenario = dataset[scenario_id]

        result = get_scenario_label(scenario)
        if result:
            append_to_output(result)
            elapsed = round(time.time() - start_time, 2)
            print(f"Processed scenario {scenario_id}\tElapsed: {elapsed}s")
            count += 1

        if count >= 1:
            break

process_scenarios()

