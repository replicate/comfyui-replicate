DEFAULT_STEP = 0.01
DEFAULT_ROUND = 0.001

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".mpg", ".mpeg")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".mpga", ".m4a")

TYPE_MAPPING = {
    "string": "STRING",
    "integer": "INT",
    "number": "FLOAT",
    "boolean": "BOOLEAN",
}


def convert_to_comfyui_input_type(
    input_name, openapi_type, openapi_format=None, default_example_input=None
):
    if openapi_type == "string" and openapi_format == "uri":
        if (
            default_example_input
            and isinstance(default_example_input, dict)
            and input_name in default_example_input
        ):
            if is_type(default_example_input[input_name], IMAGE_EXTENSIONS):
                return "IMAGE"
            elif is_type(default_example_input[input_name], VIDEO_EXTENSIONS):
                return "VIDEO"
            elif is_type(default_example_input[input_name], AUDIO_EXTENSIONS):
                return "AUDIO"
        elif any(x in input_name.lower() for x in ["image", "mask"]):
            return "IMAGE"
        elif "audio" in input_name.lower():
            return "AUDIO"
        else:
            return "STRING"

    return TYPE_MAPPING.get(openapi_type, "STRING")


def name_and_version(schema):
    author = schema["owner"]
    name = schema["name"]
    version = schema["latest_version"]["id"]
    replicate_model = f"{author}/{name}:{version}"
    node_name = f"Replicate {author}/{name}"
    return replicate_model, node_name


def resolve_schema(prop_data, openapi_schema):
    if "$ref" in prop_data:
        ref_path = prop_data["$ref"].split("/")
        current = openapi_schema
        for path in ref_path[1:]:  # Skip the first '#' element
            if path not in current:
                return prop_data  # Return original if path is invalid
            current = current[path]
        return current
    return prop_data


def schema_to_comfyui_input_types(schema):
    openapi_schema = schema["latest_version"]["openapi_schema"]
    input_schema = openapi_schema["components"]["schemas"]["Input"]
    input_types = {"required": {}, "optional": {}}
    default_example_input = get_default_example_input(schema)

    required_props = input_schema.get("required", [])

    for prop_name, prop_data in input_schema["properties"].items():
        prop_data = resolve_schema(prop_data, openapi_schema)
        default_value = prop_data.get("default", None)

        if "allOf" in prop_data:
            prop_data = resolve_schema(prop_data["allOf"][0], openapi_schema)

        if "enum" in prop_data:
            input_type = prop_data["enum"]
        elif "type" in prop_data:
            input_type = convert_to_comfyui_input_type(
                prop_name,
                prop_data["type"],
                prop_data.get("format"),
                default_example_input,
            )
        else:
            input_type = "STRING"

        input_config = {"default": default_value} if default_value is not None else {}

        if "minimum" in prop_data:
            input_config["min"] = prop_data["minimum"]
        if "maximum" in prop_data:
            input_config["max"] = prop_data["maximum"]
        if input_type == "FLOAT":
            input_config["step"] = DEFAULT_STEP
            input_config["round"] = DEFAULT_ROUND

        if "prompt" in prop_name and prop_data.get("type") == "string":
            input_config["multiline"] = True

            # Meta prompt_template needs `{prompt}` to be sent through
            # dynamicPrompts would strip it out
            if "template" not in prop_name:
                input_config["dynamicPrompts"] = True

        if prop_name in required_props:
            input_types["required"][prop_name] = (input_type, input_config)
        else:
            input_types["optional"][prop_name] = (input_type, input_config)

    input_types["optional"]["force_rerun"] = ("BOOLEAN", {"default": False})

    return order_inputs(input_types, input_schema)


def order_inputs(input_types, input_schema):
    ordered_input_types = {"required": {}, "optional": {}}
    sorted_properties = sorted(
        input_schema["properties"].items(),
        key=lambda x: x[1].get("x-order", float("inf")),
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

    ordered_input_types["optional"]["force_rerun"] = input_types["optional"][
        "force_rerun"
    ]

    return ordered_input_types


def inputs_that_need_arrays(schema):
    openapi_schema = schema["latest_version"]["openapi_schema"]
    input_schema = openapi_schema["components"]["schemas"]["Input"]
    array_inputs = []
    for prop_name, prop_data in input_schema["properties"].items():
        if prop_data.get("type") == "array":
            array_inputs.append(prop_name)

    return array_inputs


def is_type(default_example_output, extensions):
    if isinstance(
        default_example_output, str
    ) and default_example_output.lower().endswith(extensions):
        return True
    elif (
        isinstance(default_example_output, list)
        and default_example_output
        and isinstance(default_example_output[0], str)
        and default_example_output[0].lower().endswith(extensions)
    ):
        return True
    return False


def get_default_example(schema):
    default_example = schema.get("default_example")
    return default_example if default_example else None


def get_default_example_input(schema):
    default_example = get_default_example(schema)
    return default_example.get("input") if default_example else None


def get_default_example_output(schema):
    default_example = get_default_example(schema)
    return default_example.get("output") if default_example else None


def get_return_type(schema):
    openapi_schema = schema["latest_version"]["openapi_schema"]
    output_schema = (
        openapi_schema.get("components", {}).get("schemas", {}).get("Output")
    )
    default_example_output = get_default_example_output(schema)

    if output_schema and "$ref" in output_schema:
        output_schema = resolve_schema(output_schema, openapi_schema)

    if isinstance(output_schema, dict) and output_schema.get("properties"):
        return_types = {}
        for prop_name, prop_data in output_schema["properties"].items():
            if isinstance(default_example_output, dict):
                prop_value = default_example_output.get(prop_name)

                if is_type(prop_value, IMAGE_EXTENSIONS):
                    return_types[prop_name] = "IMAGE"
                elif is_type(prop_value, AUDIO_EXTENSIONS):
                    return_types[prop_name] = "AUDIO"
                elif is_type(prop_value, VIDEO_EXTENSIONS):
                    return_types[prop_name] = "VIDEO_URI"
                else:
                    return_types[prop_name] = "STRING"
            elif prop_data.get("format") == "uri":
                if "audio" in prop_name.lower():
                    return_types[prop_name] = "AUDIO"
                elif "image" in prop_name.lower():
                    return_types[prop_name] = "IMAGE"
                else:
                    return_types[prop_name] = "STRING"
            elif prop_data.get("type") == "string":
                return_types[prop_name] = "STRING"
            else:
                return_types[prop_name] = "STRING"

        return return_types

    if is_type(default_example_output, IMAGE_EXTENSIONS):
        return "IMAGE"
    elif is_type(default_example_output, VIDEO_EXTENSIONS):
        return "VIDEO_URI"
    elif is_type(default_example_output, AUDIO_EXTENSIONS):
        return "AUDIO"

    if output_schema:
        if (
            output_schema.get("type") == "string"
            and output_schema.get("format") == "uri"
        ):
            # Handle single image output
            return "IMAGE"
        elif (
            output_schema.get("type") == "array"
            and output_schema.get("items", {}).get("type") == "string"
            and output_schema.get("items", {}).get("format") == "uri"
        ):
            # Handle multiple image output
            return "IMAGE"

    return "STRING"
