"""
RILEY - Language Models
This module manages the loading and caching of language models for RILEY.
"""

import re
from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import Tokenizer
import ctranslate2

from jarvis.languagemodels.config import config, models


modelcache = {}


class ModelException(Exception):
    pass


def get_model_info(model_type="instruct"):
    """Gets info about the current model in use"""
    model_name = config[f"{model_type}_model"]

    m = [m for m in models if m["name"] == model_name][0]

    param_bits = int(re.search(r"\d+", m["quantization"]).group(0))

    m["size_gb"] = m["params"] * param_bits / 8 / 1e9
    if "/" in m["name"]:
        m["path"] = m["name"]
    else:
        m["path"] = f"jncraton/{m['name']}-{m['backend']}-{m['quantization']}"

    return m


def initialize_tokenizer(model_type, model_name):
    model_info = get_model_info(model_type)
    rev = model_info.get("revision", None)

    tok_config = hf_hub_download(
        model_info["path"], "tokenizer.json", revision=rev, local_files_only=True
    )
    tokenizer = Tokenizer.from_file(tok_config)

    if model_type == "embedding":
        tokenizer.no_padding()
        tokenizer.no_truncation()

    return tokenizer


def initialize_model(model_type, model_name, tokenizer_only=False):
    model_info = get_model_info(model_type)

    allowed = ["*.bin", "*.txt", "*.json"]
    rev = model_info.get("revision", None)

    # snapshot_download checks for updates by default
    # This can cause significant lag in offline usecases or high latency networks
    # To avoid this penalty, we try to use the local cache first.
    # If the files are not available, then we attempt a download
    try:
        path = snapshot_download(
            model_info["path"],
            max_workers=1,
            allow_patterns=allowed,
            revision=rev,
            local_files_only=True,
        )
    except FileNotFoundError:
        path = snapshot_download(
            model_info["path"], max_workers=1, allow_patterns=allowed, revision=rev
        )

    if tokenizer_only:
        return None

    if model_info["architecture"] == "encoder-only-transformer":
        return ctranslate2.Encoder(
            path,
            "cpu",
            compute_type="int8",
        )
    elif model_info["architecture"] == "decoder-only-transformer":
        return ctranslate2.Generator(path, config["device"], compute_type="int8")
    else:
        return ctranslate2.Translator(path, config["device"], compute_type="int8")


def get_model(model_type, tokenizer_only=False):
    """Gets a model from the loaded model cache

    If tokenizer_only, the model itself will not be (re)loaded
    """
    model_name = config[f"{model_type}_model"]

    if config["max_ram"] < 4 and not tokenizer_only:
        for model in modelcache:
            if model != model_name:
                try:
                    modelcache[model][1].unload_model()
                except AttributeError:
                    # Encoder-only models can't be unloaded by ctranslate2
                    pass

    if model_name not in modelcache:
        model = initialize_model(model_type, model_name, tokenizer_only)
        tokenizer = initialize_tokenizer(model_type, model_name)
        modelcache[model_name] = (tokenizer, model)
    elif not tokenizer_only:
        # Make sure model is loaded if we've never loaded it
        if not modelcache[model_name][1]:
            modelcache[model_name] = (
                modelcache[model_name][0],
                initialize_model(model_type, model_name),
            )
        # Make sure the model is reloaded if we've unloaded it
        try:
            modelcache[model_name][1].load_model()
        except AttributeError:
            # Encoder-only models can't be unloaded in ctranslate2
            pass

    return modelcache[model_name]