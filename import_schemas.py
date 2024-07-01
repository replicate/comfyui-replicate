#!/usr/bin/env python3
import replicate
import json
import os


def format_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            data["run_count"] = 0

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except json.JSONDecodeError:
        print(f"Error: {file_path} contains invalid JSON")
    except IOError:
        print(f"Error: Could not read or write to {file_path}")


def format_json_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            format_json_file(file_path)


with open("supported_models.json", "r", encoding="utf-8") as f:
    supported_models = json.load(f)

for model in supported_models["models"]:
    m = replicate.models.get(model)
    with open(f"schemas/{model.replace('/', '_')}.json", "w", encoding="utf-8") as f:
        f.write(m.json())

schemas_directory = "schemas"
format_json_files_in_directory(schemas_directory)
