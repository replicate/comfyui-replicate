import os
import json
import replicate


def convert_type(openapi_type):
    type_mapping = {
        "string": "STRING",
        "integer": "INT",
        "number": "FLOAT",
        "boolean": "BOOL",
    }
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
    input_types = {"required": {}}

    for prop_name, prop_data in schema["properties"].items():
        prop_data = resolve_schema(prop_data, schemas)

        if "allOf" in prop_data:
            prop_data = resolve_schema(prop_data["allOf"][0], schemas)

        if "enum" in prop_data:
            input_type = prop_data["enum"]
        elif "type" in prop_data:
            input_type = convert_type(prop_data["type"])
        else:
            input_type = "STRING"

        default_value = prop_data.get("default", "")

        input_config = {"default": default_value}

        if "minimum" in prop_data:
            input_config["min"] = prop_data["minimum"]
        if "maximum" in prop_data:
            input_config["max"] = prop_data["maximum"]

        if prop_data.get("type") == "string" and prop_data.get("format") == "uri":
            input_config["multiline"] = True

        if "prompt" in prop_name and prop_data.get("type") == "string":
            input_config["multiline"] = True

            if "template" not in prop_name:
                input_config["dynamicPrompts"] = True

        input_types["required"][prop_name] = (input_type, input_config)

    # Reorder input_types to put prompt and negative_prompt first
    ordered_input_types = {"required": {}}
    for key in ["prompt", "negative_prompt"]:
        if key in input_types["required"]:
            ordered_input_types["required"][key] = input_types["required"][key]
    for key in list(input_types["required"].keys()):
        if "prompt" in key:
            ordered_input_types["required"][key] = input_types["required"][key]
    for key in list(input_types["required"].keys()):
        if key not in ordered_input_types["required"] and key != "seed":
            ordered_input_types["required"][key] = input_types["required"][key]
    if "seed" in input_types["required"]:
        ordered_input_types["required"]["seed"] = input_types["required"]["seed"]

    return ordered_input_types


def create_comfyui_node(schemas, model_info):
    author = model_info["owner"]
    name = model_info["name"]
    version = model_info["latest_version"]["id"]

    replicate_model = f"{author}/{name}:{version}"
    node_name = f"Replicate {author}/{name}"
    input_schema = schemas["components"]["schemas"]["Input"]

    class ReplicateToComfyUI:
        @classmethod
        def INPUT_TYPES(cls):
            return convert_schema_to_comfyui(input_schema, schemas)

        RETURN_TYPES = ("STRING",)
        FUNCTION = "run_openapi_to_comfyui"
        CATEGORY = "Replicate"

        def run_openapi_to_comfyui(self, **kwargs):
            print(f"Running {replicate_model} with {kwargs}")
            output = replicate.run(replicate_model, input=kwargs)
            output = "".join(output).strip()
            # print(f"Output: {output}")
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


# Create ComfyUI nodes for all schema files in the "schemas" directory
comfyui_nodes = create_comfyui_nodes_from_schemas("schemas")

# Print the resulting node classes
for schema_file, node_class in comfyui_nodes.items():
    print(f"Node class for {schema_file}:")
    print(node_class.INPUT_TYPES())


NODE_CLASS_MAPPINGS = comfyui_nodes
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
