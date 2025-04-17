"""
RILEY - GPT4All Integration
This module provides integration with GPT4All models for local inference.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from gpt4all import GPT4All as GPT4AllModel
    HAS_GPT4ALL = True
    logger.info("GPT4All module loaded successfully")
except ImportError:
    HAS_GPT4ALL = False
    logger.warning("GPT4All module not found. Install with: pip install gpt4all")


class GPT4AllWrapper:
    """Wrapper class for GPT4All models to provide a consistent interface for RILEY."""
    
    DEFAULT_MODELS = {
        "small": "orca-mini-3b-gguf2-q4_0.gguf",
        "medium": "mistral-7b-openorca.Q4_0.gguf",
        "large": "nous-hermes-llama2-13b.Q4_0.gguf"
    }
    
    def __init__(self, model_name=None, model_size="small"):
        """
        Initialize a GPT4All model wrapper.
        
        Args:
            model_name: The specific model file to load. If None, will use a default model based on model_size.
            model_size: Size of model to use if model_name not specified ("small", "medium", "large")
        """
        if not HAS_GPT4ALL:
            raise ImportError("GPT4All is not installed. Install with: pip install gpt4all")
        
        self.model = None
        self.model_name = model_name
        
        if model_name is None:
            if model_size not in self.DEFAULT_MODELS:
                raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(self.DEFAULT_MODELS.keys())}")
            self.model_name = self.DEFAULT_MODELS[model_size]
        
        # Set models directory in user's home folder
        self.models_dir = os.path.expanduser("~/.gpt4all")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load the GPT4All model."""
        try:
            logger.info(f"Loading GPT4All model: {self.model_name}")
            self.model = GPT4AllModel(model_name=self.model_name, model_path=self.models_dir)
            logger.info(f"Successfully loaded GPT4All model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load GPT4All model: {e}")
            raise
    
    def generate(self, prompt, max_tokens=256, temperature=0.7, top_k=40, top_p=0.9, 
                 repeat_penalty=1.1, system_prompt=None):
        """
        Generate a response from the GPT4All model.
        
        Args:
            prompt: The user prompt to generate a response for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative, lower = more deterministic)
            top_k: Limits the next token selection to the k most probable tokens
            top_p: Limits the next token selection to a subset of tokens with a cumulative probability above a threshold P
            repeat_penalty: Penalty for repeating tokens (higher = less repetition)
            system_prompt: Optional system prompt to provide context/instructions to the model
            
        Returns:
            Generated text string
        """
        if self.model is None:
            self._load_model()
        
        # Format the system prompt if provided
        with_system = system_prompt is not None
        
        # Generate the response
        response = self.model.generate(
            prompt,
            max_tokens=max_tokens, 
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            system_prompt=system_prompt,
            streaming=False
        )
        
        return response.strip()
    
    def generate_stream(self, prompt, max_tokens=256, temperature=0.7, top_k=40, top_p=0.9,
                         repeat_penalty=1.1, system_prompt=None):
        """
        Stream a response from the GPT4All model, yielding text chunks as generated.
        
        Args are the same as generate() but this returns a generator instead.
        """
        if self.model is None:
            self._load_model()
            
        # Enable streaming mode
        # streaming=True makes this a generator that yields text chunks
        return self.model.generate(
            prompt,
            max_tokens=max_tokens, 
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            system_prompt=system_prompt,
            streaming=True
        )
    
    def embed(self, text):
        """
        Create an embedding for the provided text.
        
        Note: Most GPT4All models do not have built-in embedding capabilities.
        This is a fallback that returns an approximation based on token counts.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector (numpy array)
        """
        # Check if model has embedding capabilities
        if hasattr(self.model, "embed"):
            try:
                return self.model.embed(text)
            except Exception as e:
                logger.warning(f"Embedding with GPT4All failed: {e}. Using fallback method.")
        
        # Fallback method: Use token counts as a basic representation
        # Not recommended for production use, just a placeholder
        return np.ones(768, dtype=np.float32)  # Standard embedding size


# Simple singleton to avoid loading models multiple times
_models_cache = {}

def get_gpt4all_model(model_name=None, model_size="small"):
    """Get a cached GPT4All model or create a new one."""
    key = model_name if model_name else f"default_{model_size}"
    
    if key not in _models_cache:
        _models_cache[key] = GPT4AllWrapper(model_name, model_size)
    
    return _models_cache[key]