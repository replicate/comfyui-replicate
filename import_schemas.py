#!/usr/bin/env python3
import replicate
import json
import os
import argparse


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


def update_schemas(update=False):
    with open("supported_models.json", "r", encoding="utf-8") as f:
        supported_models = json.load(f)

    schemas_directory = "schemas"
    existing_schemas = set(os.listdir(schemas_directory))

    for model in supported_models["models"]:
        schema_filename = f"{model.replace('/', '_')}.json"
        schema_path = os.path.join(schemas_directory, schema_filename)

        if update or schema_filename not in existing_schemas:
            try:
                m = replicate.models.get(model)
                with open(schema_path, "w", encoding="utf-8") as f:
                    f.write(m.json())
                print(f"{'Updated' if update else 'Added'} schema for {model}")
            except replicate.exceptions.ReplicateError as e:
                print(f"Error fetching schema for {model}: {str(e)}")
                continue

    format_json_files_in_directory(schemas_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update model schemas")
    parser.add_argument("--update", action="store_true", help="Update all schemas, not just new ones")
    args = parser.parse_args()

    update_schemas(update=args.update)
