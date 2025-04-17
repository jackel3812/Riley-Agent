"""Global model and inference configuration for RILEY's language model capabilities

This module manages the global configuration object shared between other
modules in the language models package. It implements a dictionary with data validation
on the keys and values.
"""

import re
import os
from collections import namedtuple
from huggingface_hub import hf_hub_download
import json

ConfigItem = namedtuple("ConfigItem", "initfn default")


class ModelFilterException(Exception):
    pass


# Model list
# This list is sorted in priority order, with the best models first
# The best model that fits in the memory bounds and matches the model filter
# will be selected
models = [
    {
        "name": "openchat-3.5-0106",
        "tuning": "instruct",
        "datasets": ["mistral", "openorca", "flan"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "GPT4 Correct User: {instruction}<|end_of_turn|>" "GPT4 Correct Assistant:"
        ),
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "tuning": "instruct",
        "revision": "d02fc85",
        "datasets": ["llama3"],
        "params": 8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3",
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "Meta-Llama-3-8B-Instruct",
        "tuning": "instruct",
        "datasets": ["llama3"],
        "params": 8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3",
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "openchat-3.5-1210",
        "tuning": "instruct",
        "datasets": ["mistral", "openorca", "flan"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "GPT4 Correct User: {instruction}<|end_of_turn|>" "GPT4 Correct Assistant:"
        ),
    },
    {
        "name": "WizardLM-2-7B",
        "tuning": "instruct",
        "datasets": ["mistral", "wizardlm"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": "USER: {instruction} ASSISTANT:",
    },
    {
        "name": "neural-chat-7b-v3-1",
        "tuning": "instruct",
        "datasets": ["mistral", "slimorca"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "### System:\n"
            "Be helpful\n"
            "### User:\n{instruction}\n"
            "### Assistant:\n"
        ),
    },
    {
        "name": "Mistral-7B-Instruct-v0.2",
        "tuning": "instruct",
        "datasets": ["mistral"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": "<s>[INST] {instruction} [/INST]",
    },
    {
        "name": "flan-alpaca-gpt4-xl",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "gpt4-alpaca"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "flan-alpaca-xl",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "alpaca"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "flan-t5-xl",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "Llama-3.2-3B-Instruct",
        "tuning": "instruct",
        "revision": "5da4ba8",
        "datasets": ["llama3"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3.2",
        "repetition_penalty": 1.1,
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "fastchat-t5-3b-v1.0",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "sharegpt"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "LaMini-Flan-T5-783M",
        "tuning": "instruct",
        "revision": "e5e20a1",
        "datasets": ["c4", "flan", "lamini"],
        "params": 783e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-large",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 783e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "Llama-3.2-1B-Instruct",
        "tuning": "instruct",
        "revision": "6e3e3a1",
        "datasets": ["llama3"],
        "params": 1e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3.2",
        "repetition_penalty": 1.1,
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "LaMini-Flan-T5-248M",
        "tuning": "instruct",
        "revision": "96cfe99",
        "datasets": ["c4", "flan", "lamini"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-base",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "flan-alpaca-base",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "alpaca"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "dialogstudio-t5-base-v1.0",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "dialogstudio"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
        "prompt_fmt": ("Instruction: Be helpful. <USER> {instruction}"),
    },
    {
        "name": "LaMini-Flan-T5-77M",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "lamini"],
        "params": 77e6,
        "backend": "ct2",
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-small",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 77e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "Phi-3-mini-4k-instruct-20240701",
        "tuning": "instruct",
        "datasets": ["phi-3"],
        "params": 3.8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "mit",
        "prompt_fmt": "<|user|>\n{instruction}<|end|>\n<|assistant|>",
        "repetition_penalty": 1.1,
    },
    {
        "name": "Phi-3-mini-4k-instruct",
        "tuning": "instruct",
        "datasets": ["phi-3"],
        "params": 3.8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "mit",
        "prompt_fmt": "<|user|>\n{instruction}<|end|>\n<|assistant|>",
        "repetition_penalty": 1.1,
    },
]


class Config(dict):
    """Configuration object with validation

    This is the primary configuration interface for languagemodels. It is
    accessible through the config object and is used to configure the
    package behavior.
    """

    def __init__(self):
        self._callbacks = {}

        # Default configuration
        self.update(
            {
                "instruct_model": "",
                "code_model": "",
                "embedding_model": "gte-tiny",
                "max_ram": 0.5,
                "max_tokens": 512,
                "device": "cpu",
                "model_license": "",
                "echo": False,
            }
        )

        # Auto-selection will set reasonable defaults
        self.select_models()

    def __setitem__(self, key, value):
        """Add key validation to setitems"""
        if key == "max_ram":
            if isinstance(value, str):
                if value.lower().endswith("gb"):
                    value = float(value[:-2])
                else:
                    value = float(value)

        if key == "device":
            if value not in ["cpu", "cuda", "auto"]:
                raise KeyError("Device must be one of: [cpu, cuda, auto]")

        super().__setitem__(key, value)

        if key in self._callbacks:
            for callback in self._callbacks[key]:
                callback(value)

    def register_callback(self, key, callback):
        """Register a callback for when a key is updated"""
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def filter_models(self, key, pred):
        """Gets name of model for key based on model filters"""
        # Params are given in billions
        # We translate to an approximate runtime memory amount
        # Typically, inference requires 2x parameters storage requirement
        
        # Decoder-only models:
        #   2 * RAM for matrix + 2x overhead for inference runtime
        # Encoder-decoder models:
        #   2 * RAM for matrix + decoder runtime

        ram_gb = float(self["max_ram"])

        # Find all models matching our key
        matched = [m for m in models if m.get("tuning", "") == key]

        if not matched:
            raise ModelFilterException(f"No models with tuning: {key}")

        # Filter by ram limit
        param_bits = int(re.search(r"\d+", matched[0]["quantization"]).group(0))
        ram_limit = ram_gb * 1e9 * 8 / param_bits

        # For embedding models, we only load the encoder during embedding
        if key != "embedding":
            # This is a conservative estimate, but helps avoid OOM exceptions
            if any(m["architecture"].startswith("encoder") for m in matched):
                # Encoder-decoder models need less ram during inference
                ram_limit /= 3
            else:
                # Decoder-only models need more
                ram_limit /= 4

        matched = [m for m in matched if m["params"] <= ram_limit]

        if not matched:
            raise ModelFilterException(f"No models smaller than {ram_gb}GB")

        # Filter by license
        if self["model_license"] != "":
            license_filter = re.compile(self["model_license"])
            matched = [
                m for m in matched if "license" in m and license_filter.match(m["license"])
            ]

            if not matched:
                raise ModelFilterException(
                    f'No models found matching license filter "{self["model_license"]}"'
                )

        # Return first model, which is our preferred model
        return matched[0]["name"]

    def select_models(self):
        """Update dict with auto-selected model values

        >>> config=Config()

        >>> config.select_models()

        >>> config['instruct_model'] # doctest: +ELLIPSIS
        '...-ct2-int8'

        >>> config["max_ram"] = "8gb"
        >>> config.select_models()
        >>> config["instruct_model"] # doctest: +ELLIPSIS
        '...-ct2-int8'

        >>> config["max_ram"] = "512mb"
        >>> config.select_models()
        >>> config["instruct_model"] # doctest: +ELLIPSIS
        '...-ct2-int8'
        """
        try:
            model = self.filter_models("instruct", None)
            self["instruct_model"] = f"{model}-{models[0]['backend']}-{models[0]['quantization']}"
        except ModelFilterException as e:
            self["instruct_model"] = "LaMini-Flan-T5-248M-ct2-int8"

        try:
            model = self.filter_models("code", None)
            self["code_model"] = f"{model}-{models[0]['backend']}-{models[0]['quantization']}"
        except ModelFilterException as e:
            self["code_model"] = self["instruct_model"]

    def require_model_license(self, pattern):
        """Filter models to match license

        >>> config = Config()
        >>> config.require_model_license("apache-2.0")
        >>> config.select_models()
        >>> config["instruct_model"] # doctest: +ELLIPSIS
        '...-ct2-int8'
        """
        self["model_license"] = pattern

        # Force model reselection for new license pattern
        self.select_models()


def use_hf_model(repository, model_type="instruct"):
    """Directly use a HuggingFace repository as a model"""
    model_name = repository.split("/")[-1]
    config[f"{model_type}_model"] = f"{model_name}"


# Initialize configuration
config = Config()