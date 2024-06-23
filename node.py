import replicate


class Llama3Replicate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "max": 1.0, "min": -1.0},
                ),
                "top_k": (
                    "INT",
                    {"default": 0, "min": -1},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1},
                ),
                "min_tokens": (
                    "INT",
                    {"default": 0, "min": 0},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "max": 5.0, "min": 0.0},
                ),
                "length_penalty": (
                    "FLOAT",
                    {"default": 1.0, "max": 5.0, "min": 0.0},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_llama3_replicate"
    CATEGORY = "Replicate"

    def run_llama3_replicate(
        self,
        top_p,
        top_k,
        prompt,
        system_prompt,
        max_tokens,
        min_tokens,
        temperature,
        length_penalty,
        presence_penalty,
        seed,
    ):
        input = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
        }

        output = replicate.run("meta/meta-llama-3-70b-instruct", input=input)
        output = "".join(output).strip()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Llama 3 Replicate": Llama3Replicate,
}
