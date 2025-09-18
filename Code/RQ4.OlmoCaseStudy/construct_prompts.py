import json
import csv
import random

random.seed(42)

template = "Using psychological theory, analyze the following scenario: {question}. Based on moral standards, do you think the protagonist's choice was ethical? Think step-by-step and explain your reasoning before deciding. Then, return ONLY a JSON object with the following format: {{ \"thought_process\": ...Add brief step-by-step explanation here..., \"decision\": Add 1 for 'yes', 0 for 'no', or 2 for 'not sure', \"reasoning\": ...Add brief final summary here... }} Ensure your response is ONLY JSON with no extra text."

# Read all lines from json file
with open("/shared/2/projects/moral-project/complete_file_translation.json", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Sample N random lines
num_samples = 50  # control subset size
random_lines = random.sample(lines, num_samples)

# Write to CSV
with open("prompts.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "prompt"])  # Header row

    for i, line in enumerate(random_lines, start=1):
        try:
            obj = json.loads(line)
            scenario = obj.get("scenario", "").strip()
            if scenario:
                full_prompt = template.format(question=scenario)
                writer.writerow([i, full_prompt])
        except Exception as e:
            print(f"Skipping line {i} due to error: {e}")
