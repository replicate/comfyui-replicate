import replicate
import json
import os

models_to_import = [
    "andreasjansson/blip-2",
    "bytedance/sdxl-lightning-4step",
    "fofr/face-to-many",
    "fofr/sd3-with-chaos",
    "meta/llama-2-7b-chat",
    "meta/llama-2-70b-chat",
    # "meta/meta-llama-3-70b-instruct",
    # "meta/meta-llama-3-8b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "philz1337x/clarity-upscaler",
    "salesforce/blip",
    "smoretalk/rembg-enhance",
    "stability-ai/sdxl",
    "stability-ai/stable-diffusion-3",
    "yorickvp/llava-13b",
    "yorickvp/llava-v1.6-34b",
    "yorickvp/llava-v1.6-mistral-7b",
    "snowflake/snowflake-arctic-instruct",
    "batouresearch/high-resolution-controlnet-tile",
    "batouresearch/magic-style-transfer",
    "batouresearch/magic-image-refiner",
    "lucataco/pasd-magnify",
    "cjwbw/supir",
    "lucataco/qwen-vl-chat",
    "omniedgeio/face-swap",
    "ai-forever/kandinsky-2.2",
    "pharmapsychotic/clip-interrogator",
    "cuuupid/idm-vton",
    "cuuupid/glm-4v-9b",
    "lucataco/moondream2",
]


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


for model in models_to_import:
    m = replicate.models.get(model)
    with open(f"schemas/{model.replace('/', '_')}.json", "w") as f:
        f.write(m.json())

schemas_directory = "schemas"
format_json_files_in_directory(schemas_directory)
