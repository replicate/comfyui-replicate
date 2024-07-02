import os
import json
import requests
from PIL import Image
from io import BytesIO
import io
from torchvision import transforms
import torch
import base64
import time
import torchaudio
import soundfile as sf
from replicate.client import Client
from .schema_to_node import (
    schema_to_comfyui_input_types,
    get_return_type,
    name_and_version,
    inputs_that_need_arrays,
)

replicate = Client(headers={"User-Agent": "comfyui-replicate/1.0.1"})


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

        RETURN_TYPES = (
            tuple(return_type.values())
            if isinstance(return_type, dict)
            else (return_type,)
        )
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
                    elif input_type == "AUDIO":
                        kwargs[key] = self.audio_to_base64(value)

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

        def audio_to_base64(self, audio):
            if (
                isinstance(audio, dict)
                and "waveform" in audio
                and "sample_rate" in audio
            ):
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
            else:
                waveform, sample_rate = audio

            # Ensure waveform is 2D
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() > 2:
                waveform = waveform.squeeze()
                if waveform.dim() > 2:
                    raise ValueError("Waveform must be 1D or 2D")

            buffer = io.BytesIO()
            sf.write(buffer, waveform.numpy().T, sample_rate, format="wav")
            buffer.seek(0)
            audio_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:audio/wav;base64,{audio_str}"

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
                if isinstance(v, str)
                and (v.startswith("data:image") or v.startswith("data:audio"))
                else v
                for k, v in kwargs.items()
            }
            print(f"Running {replicate_model} with {truncated_kwargs}")

        def handle_image_output(self, output):
            if output is None:
                print("No image output received")
                return None

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

        def handle_audio_output(self, output):
            if output is None:
                print("No audio output received from the model")
                return None

            if isinstance(output, str):
                output_list = [output]
            elif isinstance(output, (list, tuple)):
                output_list = list(output)
            else:
                print(f"Unexpected output type: {type(output)}")
                return None

            if output_list:
                audio_data = []
                for audio_url in output_list:
                    if audio_url:
                        response = requests.get(audio_url)
                        if response.status_code == 200:
                            audio_content = BytesIO(response.content)
                            waveform, sample_rate = torchaudio.load(audio_content)
                            audio_data.append(
                                {
                                    "waveform": waveform.unsqueeze(0),
                                    "sample_rate": sample_rate,
                                }
                            )
                        else:
                            print(
                                f"Failed to download audio. Status code: {response.status_code}"
                            )
                    else:
                        print("Empty audio URL received")

                # If there's only one audio file, return it directly
                if len(audio_data) == 1:
                    return audio_data[0]
                # If there are multiple audio files, return them as a list
                return audio_data
            else:
                print("No valid audio URLs in the output")
                return None

        def remove_falsey_optional_inputs(self, kwargs):
            optional_inputs = self.INPUT_TYPES().get("optional", {})
            for key in list(kwargs.keys()):
                if key in optional_inputs and not kwargs[key]:
                    del kwargs[key]

        def run_replicate_model(self, **kwargs):
            self.handle_array_inputs(kwargs)
            self.remove_falsey_optional_inputs(kwargs)
            self.convert_input_images_to_base64(kwargs)
            self.log_input(kwargs)
            kwargs_without_force_rerun = {
                k: v for k, v in kwargs.items() if k != "force_rerun"
            }
            output = replicate.run(replicate_model, input=kwargs_without_force_rerun)
            print(f"Output: {output}")

            processed_outputs = []
            if isinstance(return_type, dict):
                for prop_name, prop_type in return_type.items():
                    if prop_type == "IMAGE":
                        processed_outputs.append(
                            self.handle_image_output(output.get(prop_name))
                        )
                    elif prop_type == "AUDIO":
                        processed_outputs.append(
                            self.handle_audio_output(output.get(prop_name))
                        )
                    elif prop_type == "STRING":
                        processed_outputs.append(
                            "".join(list(output.get(prop_name, ""))).strip()
                        )
            else:
                if return_type == "IMAGE":
                    processed_outputs.append(self.handle_image_output(output))
                elif return_type == "AUDIO":
                    processed_outputs.append(self.handle_audio_output(output))
                else:
                    processed_outputs.append("".join(list(output)).strip())

            return tuple(processed_outputs)

    return node_name, ReplicateToComfyUI


def create_comfyui_nodes_from_schemas(schemas_dir):
    nodes = {}
    current_path = os.path.dirname(os.path.abspath(__file__))
    schemas_dir_path = os.path.join(current_path, schemas_dir)
    for schema_file in os.listdir(schemas_dir_path):
        if schema_file.endswith(".json"):
            with open(
                os.path.join(schemas_dir_path, schema_file), "r", encoding="utf-8"
            ) as f:
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
