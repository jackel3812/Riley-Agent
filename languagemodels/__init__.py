"""
RILEY - Language Models Integration
This module integrates the languagemodels package capabilities into RILEY.
Provides advanced natural language processing abilities with local model inference.
"""

import os
import requests
import datetime
import json
import re
import logging
import numpy as np
from typing import overload, List, Dict, Any, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

# Local RILEY imports
from jarvis.core.knowledge_base import get_riley_information, get_jarvis_information

# Initialize a basic configuration
class Config(dict):
    """Configuration object for language models"""
    def __init__(self):
        self.update({
            "max_ram": 4.0,  # Use up to 4GB RAM for models
            "device": "cpu",  # Use CPU for inference
            "max_tokens": 512,  # Maximum tokens to generate
            "model_size": "small",  # small, medium, large
            "use_openai": False,  # Whether to use OpenAI API
            "use_anthropic": False,  # Whether to use Anthropic API
        })

# Initialize configuration
config = Config()

class Document:
    """A document used for semantic search"""
    def __init__(self, content, name="", embedding=None):
        self.content = content
        self.name = name
        self.embedding = embedding or np.random.rand(768)  # Placeholder embedding

class RetrievalContext:
    """Provides a context for document retrieval"""
    def __init__(self):
        self.docs = []
        self.chunks = []
        
    def clear(self):
        self.docs = []
        self.chunks = []
    
    def store(self, doc, name=""):
        """Store a document for later retrieval"""
        if not any(d.content == doc for d in self.docs):
            self.docs.append(Document(doc, name))
            self._chunk_and_store(doc, name)
    
    def _chunk_and_store(self, doc, name=""):
        """Split document into chunks and store them"""
        # Simple chunking by paragraphs
        chunks = [p for p in doc.split("\n\n") if p.strip()]
        
        for chunk in chunks:
            if chunk:
                self.chunks.append(Document(chunk, name))
    
    def get_match(self, query):
        """Find the closest matching document for a query"""
        if not self.docs:
            return None
            
        # Simple keyword matching as a fallback when we don't have embeddings
        best_score = 0
        best_doc = None
        
        for doc in self.docs:
            # Count shared words as a simple similarity measure
            query_words = set(query.lower().split())
            doc_words = set(doc.content.lower().split())
            shared = len(query_words.intersection(doc_words))
            
            if shared > best_score:
                best_score = shared
                best_doc = doc.content
                
        return best_doc
    
    def get_context(self, query, max_tokens=128):
        """Get relevant context chunks for a query"""
        if not self.chunks:
            return None
            
        # Simple keyword matching
        chunks = []
        words_count = 0
        max_words = max_tokens // 2  # Rough approximation
        
        for chunk in sorted(self.chunks, 
                         key=lambda d: len(set(query.lower().split()).intersection(set(d.content.lower().split()))), 
                         reverse=True):
            chunk_words = len(chunk.content.split())
            if words_count + chunk_words <= max_words:
                chunks.append(chunk.content)
                words_count += chunk_words
                
        return "\n\n".join(chunks) if chunks else None

# Initialize retrieval context
docs = RetrievalContext()

def complete(prompt: str) -> str:
    """Provide one completion for a given open-ended prompt"""
    # Fallback implementation using our knowledge base
    riley_info = get_riley_information()
    
    # Try to provide a sensible completion based on the RILEY information
    if "who are you" in prompt.lower() or "what is your name" in prompt.lower():
        return f"I am {riley_info['name']}, which stands for {riley_info['full_name']}."
    
    if "capabilities" in prompt.lower() or "what can you do" in prompt.lower():
        capabilities = ", ".join(riley_info['core_capabilities'][:3])
        return f"I can help with {capabilities}, and more."
        
    # Generic completion
    return "I'm here to assist you with that."

def do(prompt, choices=None):
    """Follow a single-turn instructional prompt"""
    prompts = [prompt] if isinstance(prompt, str) else prompt
    
    # Handle classification with choices
    if choices:
        return classify_text(prompts[0], choices)
    
    # Generate a response using fallback method
    responses = [generate_response(p) for p in prompts]
    
    return responses[0] if isinstance(prompt, str) else responses

def classify_text(text, choices):
    """Classify text into one of the given choices"""
    # Simple keyword matching classification as fallback
    text = text.lower()
    scores = {}
    
    for choice in choices:
        # Count occurrences of the choice and related words in the text
        choice_lower = choice.lower()
        score = text.count(choice_lower)
        
        # Add scores for word overlap
        text_words = set(text.split())
        choice_words = set(choice_lower.split())
        score += len(text_words.intersection(choice_words))
        
        scores[choice] = score
    
    # Return the highest scoring choice, or first choice if all scores are 0
    max_score = max(scores.values()) if scores else 0
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    else:
        return choices[0] if choices else ""

def chat(prompt: str) -> str:
    """Get new message from chat-optimized language model"""
    # Extract the last user message as the query
    last_user_msg = ""
    parts = prompt.split("\n\n")
    for part in reversed(parts):
        if part.lower().startswith("user:"):
            last_user_msg = part[5:].strip()
            break
    
    # If we found a user message, respond to it
    if last_user_msg:
        return generate_response(last_user_msg)
    
    return "I'm not sure I understand. Could you please rephrase your question?"

def extract_answer(question: str, context: str) -> str:
    """Extract an answer to a question from a provided context"""
    # Basic implementation to find sentences most relevant to the question
    question_words = set(question.lower().split())
    
    # Split context into sentences
    sentences = re.split(r'(?<=[.!?])\s+', context)
    
    # Score each sentence by word overlap with question
    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(question_words.intersection(sentence_words))
        scored_sentences.append((sentence, overlap))
    
    # Sort by score and return top sentence
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if scored_sentences:
        return scored_sentences[0][0]
    else:
        return "I couldn't find an answer in the provided context."

def code(prompt: str) -> str:
    """Complete a code prompt"""
    # Simple code generation fallback
    # Look for common patterns and provide simple completions
    prompt_lower = prompt.lower()
    
    if "def " in prompt_lower and ":" in prompt:
        # Function definition - add a simple return statement
        return "    return None"
        
    if "for " in prompt_lower and ":" in prompt:
        # For loop - add a simple body
        return "    pass"
        
    if "if " in prompt_lower and ":" in prompt:
        # If statement - add a simple body
        return "    pass"
        
    if "class " in prompt_lower and ":" in prompt:
        # Class definition - add a simple initializer
        return "    def __init__(self):\n        pass"
    
    # Generic code continuation
    return "# Add your code here"

def classify(doc: str, label1: str, label2: str) -> str:
    """Performs binary classification on an input"""
    return classify_text(doc, [label1, label2])

def store_doc(doc: str, name: str = "") -> None:
    """Store document for later retrieval"""
    docs.store(doc, name)

def load_doc(query: str) -> str:
    """Load a matching document"""
    return docs.get_match(query) or "No matching document found."

def get_doc_context(query: str) -> str:
    """Loads context from documents"""
    return docs.get_context(query) or "No relevant context found."

def get_web(url: str) -> str:
    """Return the text of paragraphs from a web page"""
    try:
        res = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; RILEY/1.0)"}
        )
        
        if res.status_code == 200:
            if "text/html" in res.headers.get("Content-Type", ""):
                # Very simple HTML text extraction
                text = res.text
                # Remove tags
                text = re.sub(r'<[^>]*>', ' ', text)
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            else:
                return res.text
        else:
            return f"Failed to retrieve content: HTTP {res.status_code}"
    except Exception as e:
        return f"Error retrieving content: {str(e)}"

def get_wiki(topic: str) -> str:
    """Return Wikipedia summary for a topic"""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        res = requests.get(url)
        
        if res.status_code == 200:
            data = res.json()
            return data.get("extract", f"No Wikipedia information found for '{topic}'")
        else:
            return f"Failed to retrieve Wikipedia information: HTTP {res.status_code}"
    except Exception as e:
        return f"Error retrieving Wikipedia information: {str(e)}"

def get_weather(latitude, longitude):
    """Fetch the current weather for a supplied longitude and latitude"""
    try:
        res = requests.get(f"https://api.weather.gov/points/{latitude},{longitude}")
        if res.status_code != 200:
            return f"Weather data unavailable: HTTP {res.status_code}"
            
        points_data = res.json()
        forecast_url = points_data["properties"]["forecast"]
        
        res = requests.get(forecast_url)
        if res.status_code != 200:
            return f"Forecast data unavailable: HTTP {res.status_code}"
            
        forecast_data = res.json()
        periods = forecast_data["properties"]["periods"]
        
        # Return forecasts for next 3 periods
        forecasts = []
        for p in periods[:3]:
            short = p["shortForecast"]
            temp = p["temperature"]
            unit = p["temperatureUnit"]
            name = p["name"]
            forecasts.append(f"{name}: {short} with a temperature of {temp}°{unit}")
        
        return " ".join(forecasts)
    except Exception as e:
        return f"Error retrieving weather information: {str(e)}"

def get_date():
    """Return the current date and time in a human-readable format"""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M%p")

def generate_response(prompt: str) -> str:
    """Generate a response to a prompt using our knowledge base"""
    riley_info = get_riley_information()
    jarvis_info = get_jarvis_information()
    
    # Check if prompt is about RILEY or JARVIS
    prompt_lower = prompt.lower()
    
    # Check if we have context from stored documents
    context = get_doc_context(prompt)
    if context:
        # Try to answer based on retrieved context
        return extract_answer(prompt, context)
    
    # Check for specific topic questions
    if "jarvis" in prompt_lower:
        if "who is jarvis" in prompt_lower or "what is jarvis" in prompt_lower:
            return f"J.A.R.V.I.S. stands for {jarvis_info['full_name']}. It was {jarvis_info['description']} and was voiced by {jarvis_info['voice_actor']}."
        
        features = ", ".join(jarvis_info['key_features'][:3])
        return f"JARVIS is Tony Stark's AI assistant from the Marvel universe. Some of its key features include {features}, and more."
    
    if "who are you" in prompt_lower or "what is your name" in prompt_lower:
        return f"I'm {riley_info['name']}, which stands for {riley_info['full_name']}. I'm an advanced AI assistant designed to help with a wide range of tasks!"
    
    if "what can you do" in prompt_lower or "capabilities" in prompt_lower:
        capabilities = ", ".join(riley_info['core_capabilities'][:3])
        return f"I can assist with various tasks including {capabilities}, and much more. How can I help you today?"
    
    if "weather" in prompt_lower:
        return "I can provide weather information when given specific coordinates. For example, try asking about the weather at latitude 41.8 and longitude -87.6 (Chicago)."
    
    if "math" in prompt_lower or "calculate" in prompt_lower:
        domains = ", ".join(riley_info['domains']['mathematics'][:5])
        return f"I can help with various math topics including {domains}. What specific calculation do you need help with?"
    
    if "physics" in prompt_lower:
        domains = ", ".join(riley_info['domains']['physics'][:5])
        return f"I can assist with physics topics including {domains}. What specific physics question do you have?"
    
    if "science" in prompt_lower:
        domains = ", ".join(riley_info['domains']['science'][:5])
        return f"I can help with scientific topics including {domains}. What specific science question are you interested in?"
    
    # Generic fallback response
    return "I'm RILEY, your AI assistant. I can help with mathematics, physics, scientific data analysis, creative content, and much more. How can I assist you today?"