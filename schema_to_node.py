DEFAULT_STEP = 0.01
DEFAULT_ROUND = 0.001


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


def name_and_version(schema):
    author = schema["owner"]
    name = schema["name"]
    version = schema["latest_version"]["id"]
    replicate_model = f"{author}/{name}:{version}"
    node_name = f"Replicate {author}/{name}"
    return replicate_model, node_name


def resolve_schema(prop_data, openapi_schemas):
    if "$ref" in prop_data:
        ref_path = prop_data["$ref"].split("/")
        current = openapi_schemas
        for path in ref_path[1:]:  # Skip the first '#' element
            current = current[path]
        return current
    return prop_data


def schema_to_comfyui_input_types(schema):
    openapi_schema = schema["latest_version"]["openapi_schema"]
    input_schema = openapi_schema["components"]["schemas"]["Input"]
    input_types = {"required": {}, "optional": {}}

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
                prop_data["type"], prop_data.get("format")
            )
        else:
            input_type = "STRING"

        input_config = {"default": default_value}

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


def get_return_type(schema):
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    video_extensions = (".mp4", ".mkv", ".webm", ".mov", ".mpg", ".mpeg")
    audio_extensions = (".mp3", ".wav")

    openapi_schema = schema["latest_version"]["openapi_schema"]
    output_schema = openapi_schema["components"]["schemas"].get("Output")
    default_example = schema.get("default_example")
    default_example_output = default_example.get("output") if default_example else None

    if is_type(default_example_output, image_extensions):
        return "IMAGE"
    elif is_type(default_example_output, video_extensions):
        return "VIDEO_URI"
    elif is_type(default_example_output, audio_extensions):
        return "AUDIO_URI"

    if output_schema:
        if (
            output_schema.get("type") == "string"
            and output_schema.get("format") == "uri"
        ):
            # Handle single image output
            return "IMAGE"
        elif (
            output_schema.get("type") == "array"
            and output_schema["items"].get("type") == "string"
            and output_schema["items"].get("format") == "uri"
        ):
            # Handle multiple image output
            return "IMAGE"

    return "STRING"
