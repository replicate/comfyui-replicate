# comfyui-replicate

Custom nodes for running [Replicate models](https://replicate.com/explore) in ComfyUI.

Take a look at the [example workflows](https://github.com/replicate/comfyui-replicate/tree/main/example_workflows) and the [supported Replicate models](https://github.com/replicate/comfyui-replicate/blob/main/supported_models.json) to get started.

![example-screenshot](https://github.com/replicate/comfyui-replicate/assets/319055/0eedb026-de3e-402a-b8fc-0a14c2fd209e)

## Set your Replicate API token before running

Make sure you set your REPLICATE_API_TOKEN in your environment. Get your API tokens here, we recommend creating a new one:

https://replicate.com/account/api-tokens

To pass in your API token when running ComfyUI you could do:

On MacOS or Linux:

```sh
export REPLICATE_API_TOKEN="r8_************"; python main.py
```

On Windows:

```sh
set REPLICATE_API_TOKEN="r8_************"; python main.py
```

## Direct installation

```sh
cd ComfyUI/custom-nodes
git clone https://github.com/replicate/comfyui-replicate
cd comfyui-replicate
pip install -r requirements.txt
```

## Supported Replicate models

View the `supported_models.json` to see which models are packaged by default.

## Update Replicate models

Simply run `./import_schemas.py` to update all model nodes. The latest version of a model is used by default.

## Add more models

Only models that return simple text or image outputs are currently supported. If a model returns audio, video, JSON objects or a combination of outputs, the node will not work as expected.

If you want to add more models, you can:

- add the model to `supported_models.json` (for example, `fofr/consistent-character`)
- run `./import_schemas.py`, this will update all schemas and import your new model
- restart ComfyUI
- use the model in workflow, it’ll have the title ‘Replicate [model author/model name]’

## Roadmap

Things to investigate and add to this custom node package:

- support for more types of Replicate model (audio and video first)
- showing logs, prediction status and progress (via tqdm)

## Contributing

If you add models that others would find useful, feel free to raise PRs.
