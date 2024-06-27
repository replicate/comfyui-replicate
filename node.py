import os
import json
import replicate
import requests
from PIL import Image
from io import BytesIO
import io
from torchvision import transforms
import torch
import base64
import time
from .schema_to_node import (
    schema_to_comfyui_input_types,
    get_return_type,
    name_and_version,
    inputs_that_need_arrays,
)


def create_comfyui_node(schema):
    replicate_model, node_name = name_and_version(schema)
    return_type = get_return_type(schema)

    class ReplicateToComfyUI:
        @classmethod
        def IS_CHANGED(cls, **kwargs):
            return time.time() if kwargs["force_rerun"] else ""

        @classmethod
        def INPUT_TYPES(cls):
            return schema_to_comfyui_input_types(schema)

        RETURN_TYPES = (return_type,)
        FUNCTION = "run_replicate_model"
        CATEGORY = "Replicate"

        def convert_input_images_to_base64(self, kwargs):
            for key, value in kwargs.items():
                if value is not None:
                    input_type = (
                        self.INPUT_TYPES()["required"].get(key, (None,))[0]
                        or self.INPUT_TYPES().get("optional", {}).get(key, (None,))[0]
                    )
                    if input_type == "IMAGE":
                        kwargs[key] = self.image_to_base64(value)

        def image_to_base64(self, image):
            if isinstance(image, torch.Tensor):
                image = image.permute(0, 3, 1, 2).squeeze(0)
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(image)
            else:
                pil_image = image

            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        def handle_array_inputs(self, kwargs):
            array_inputs = inputs_that_need_arrays(schema)
            for input_name in array_inputs:
                if input_name in kwargs:
                    if isinstance(kwargs[input_name], str):
                        if kwargs[input_name] == "":
                            kwargs[input_name] = []
                        else:
                            kwargs[input_name] = kwargs[input_name].split("\n")
                    else:
                        kwargs[input_name] = [kwargs[input_name]]

        def log_input(self, kwargs):
            truncated_kwargs = {
                k: v[:20] + "..."
                if isinstance(v, str) and v.startswith("data:image")
                else v
                for k, v in kwargs.items()
            }
            print(f"Running {replicate_model} with {truncated_kwargs}")

        def handle_image_output(self, output):
            # Handle both string and list outputs
            output_list = [output] if isinstance(output, str) else list(output)
            if output_list:
                output_tensors = []
                transform = transforms.ToTensor()
                for image_url in output_list:
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        tensor_image = transform(image)
                        tensor_image = tensor_image.unsqueeze(0)
                        tensor_image = tensor_image.permute(0, 2, 3, 1).cpu().float()
                        output_tensors.append(tensor_image)
                    else:
                        print(
                            f"Failed to download image. Status code: {response.status_code}"
                        )
                # Combine all tensors into a single batch if multiple images
                return (
                    torch.cat(output_tensors, dim=0)
                    if len(output_tensors) > 1
                    else output_tensors[0]
                )
            else:
                print("No output received from the model")
                return None

        def run_replicate_model(self, **kwargs):
            self.handle_array_inputs(kwargs)
            self.convert_input_images_to_base64(kwargs)
            self.log_input(kwargs)
            output = replicate.run(replicate_model, input=kwargs)
            print(f"Output: {output}")

            if return_type == "IMAGE":
                output = self.handle_image_output(output)
            else:
                output = "".join(list(output)).strip()

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
                node_name, node_class = create_comfyui_node(schema)
                nodes[node_name] = node_class
    return nodes


_cached_node_class_mappings = None


def get_node_class_mappings():
    global _cached_node_class_mappings
    if _cached_node_class_mappings is None:
        _cached_node_class_mappings = create_comfyui_nodes_from_schemas("schemas")
    return _cached_node_class_mappings


NODE_CLASS_MAPPINGS = get_node_class_mappings()
