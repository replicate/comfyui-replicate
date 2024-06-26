import os
import json
import replicate
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch


def convert_to_comfyui_input_type(openapi_type, openapi_format=None):
    type_mapping = {
        "string": "STRING",
        "integer": "INT",
        "number": "FLOAT",
        "boolean": "BOOLEAN",
    }
    if openapi_type == "string" and openapi_format == "uri":
        return "IMAGE"
    return type_mapping.get(openapi_type, "STRING")


def resolve_schema(prop_data, schemas):
    if "$ref" in prop_data:
        ref_path = prop_data["$ref"].split("/")
        current = schemas
        for path in ref_path[1:]:  # Skip the first '#' element
            current = current[path]
        return current
    return prop_data


def convert_schema_to_comfyui(schema, schemas):
    input_types = {"required": {}, "optional": {}}

    required_props = schema.get("required", [])

    for prop_name, prop_data in schema["properties"].items():
        prop_data = resolve_schema(prop_data, schemas)

        if "allOf" in prop_data:
            prop_data = resolve_schema(prop_data["allOf"][0], schemas)

        if "enum" in prop_data:
            input_type = prop_data["enum"]
        elif "type" in prop_data:
            input_type = convert_to_comfyui_input_type(
                prop_data["type"], prop_data.get("format")
            )
        else:
            input_type = "STRING"

        default_value = prop_data.get("default", None)

        input_config = {"default": default_value}

        if "minimum" in prop_data:
            input_config["min"] = prop_data["minimum"]
        if "maximum" in prop_data:
            input_config["max"] = prop_data["maximum"]

        if "prompt" in prop_name and prop_data.get("type") == "string":
            input_config["multiline"] = True

            # Meta prompt_template needs `{prompt}` to be sent through
            if "template" not in prop_name:
                input_config["dynamicPrompts"] = True

        if prop_name in required_props:
            input_types["required"][prop_name] = (input_type, input_config)
        else:
            input_types["optional"][prop_name] = (input_type, input_config)

    return reorder_input_types(input_types, schema)


def reorder_input_types(input_types, schema):
    ordered_input_types = {"required": {}, "optional": {}}

    # Sort properties based on "x-order" if available
    sorted_properties = sorted(
        schema["properties"].items(), key=lambda x: x[1].get("x-order", float("inf"))
    )

    for prop_name, _ in sorted_properties:
        if prop_name in input_types["required"]:
            ordered_input_types["required"][prop_name] = input_types["required"][
                prop_name
            ]
        elif prop_name in input_types["optional"]:
            ordered_input_types["optional"][prop_name] = input_types["optional"][
                prop_name
            ]

    return ordered_input_types


def get_return_type(schemas, model_info):
    output_schema = schemas["components"]["schemas"].get("Output")
    default_example_output = model_info.get("default_example", {}).get("output", [])

    if (
        output_schema
        and output_schema.get("type") == "array"
        and output_schema["items"].get("type") == "string"
        and output_schema["items"].get("format") == "uri"
        and default_example_output
        and default_example_output[0]
        and default_example_output[0]
        .lower()
        .endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
    ):
        return "IMAGE"
    return "STRING"


def create_comfyui_node(schemas, model_info):
    author = model_info["owner"]
    name = model_info["name"]
    version = model_info["latest_version"]["id"]

    replicate_model = f"{author}/{name}:{version}"
    node_name = f"Replicate {author}/{name}"
    input_schema = schemas["components"]["schemas"]["Input"]
    return_type = get_return_type(schemas, model_info)
    print(f"{node_name} - {return_type}")

    class ReplicateToComfyUI:
        @classmethod
        def INPUT_TYPES(cls):
            return convert_schema_to_comfyui(input_schema, schemas)

        RETURN_TYPES = (return_type,)
        FUNCTION = "run_openapi_to_comfyui"
        CATEGORY = "Replicate"

        def run_openapi_to_comfyui(self, **kwargs):
            print(f"Running {replicate_model} with {kwargs}")
            output = replicate.run(replicate_model, input=kwargs)
            print(f"Output: {output}")

            if return_type == "IMAGE":
                # Convert generator to list
                output_list = list(output)
                if output_list:
                    output_tensors = []
                    transform = transforms.ToTensor()
                    for image_url in output_list:
                        # Download the image from the URL
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            image = Image.open(BytesIO(response.content))
                            # Convert image to RGB if it's not already
                            if image.mode != "RGB":
                                image = image.convert("RGB")
                            # Convert to tensor and reshape
                            tensor_image = transform(image)
                            tensor_image = tensor_image.unsqueeze(0)
                            tensor_image = (
                                tensor_image.permute(0, 2, 3, 1).cpu().float()
                            )
                            output_tensors.append(tensor_image)
                        else:
                            print(
                                f"Failed to download image. Status code: {response.status_code}"
                            )
                    # Combine all tensors into a single batch if multiple images
                    output = (
                        torch.cat(output_tensors, dim=0)
                        if len(output_tensors) > 1
                        else output_tensors[0]
                    )
                else:
                    print("No output received from the model")
                    output = None
            else:
                output = "".join(list(output))

            return (output,)

    return node_name, ReplicateToComfyUI


def create_comfyui_nodes_from_schemas(schemas_dir):
    nodes = {}
    current_path = os.path.dirname(os.path.abspath(__file__))
    schemas_dir_path = os.path.join(current_path, schemas_dir)
    for schema_file in os.listdir(schemas_dir_path):
        if schema_file.endswith(".json"):
            with open(os.path.join(schemas_dir_path, schema_file), "r") as f:
                schema = json.load(f)
                openapi_schema = schema["latest_version"]["openapi_schema"]
                model_info = schema
                node_name, node_class = create_comfyui_node(openapi_schema, model_info)
                nodes[node_name] = node_class
    return nodes


NODE_CLASS_MAPPINGS = create_comfyui_nodes_from_schemas("schemas")
print(NODE_CLASS_MAPPINGS)

# # Load the schema
# with open("schema.json", "r") as f:
#     schema = json.load(f)
#     openapi_schema = schema["latest_version"]["openapi_schema"]

# # Create the ComfyUI node
# ComfyUINode = create_comfyui_node(openapi_schema, schema)

# # Print the resulting node class
# print(ComfyUINode.INPUT_TYPES())

# # Create an instance of the node and pass in defaults
# node_instance = ComfyUINode()
# defaults = {
#     "seed": 0,
#     "image": "https://example.com/default_image.png",
#     "style": "3D",
#     "prompt": "a person",
#     "lora_scale": 1.0
# }

# # Run the node with the defaults
# result = node_instance.run_openapi_to_comfyui(**defaults)
# print(result)
