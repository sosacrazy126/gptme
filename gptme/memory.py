"""Memory management module for gptme using memoripy."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from memoripy import MemoryManager, JSONStorage, InMemoryStorage

from .config import Config
from .dirs import get_data_dir
from .message import Message

class GPTMEMemoryManager:
    """Manages memory and context for gptme conversations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_dir = Path(get_data_dir()) / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage based on configuration
        if config.memory_storage_type == "json":
            storage = JSONStorage(str(self.memory_dir / "interaction_history.json"))
        else:  # "in_memory"
            storage = InMemoryStorage()
        
        # Initialize memory manager with appropriate models
        self.manager = MemoryManager(
            api_key=config.openai_api_key,
            chat_model="openai",  # Using OpenAI for consistency
            chat_model_name=config.model,  # Use configured model
            embedding_model="openai",
            embedding_model_name="text-embedding-3-small",  # Latest embedding model
            storage=storage,
            decay_rate=config.memory_decay_rate
        )
    
    def add_interaction(self, message: Message, response: str) -> None:
        """Add a new interaction to memory."""
        # Combine message and response for embedding
        combined_text = f"{message.content}\n{response}"
        
        # Generate embedding and extract concepts
        embedding = self.manager.get_embedding(combined_text)
        concepts = self.manager.extract_concepts(combined_text)
        
        # Add to memory store
        self.manager.add_interaction(
            prompt=message.content,
            output=response,
            embedding=embedding,
            concepts=concepts,
            timestamp=time.time()  # Add timestamp for decay calculations
        )
    
    def get_relevant_context(self, message: Message, max_context: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant past interactions for the given message."""
        # Get recent interactions
        short_term, _ = self.manager.load_history()
        last_interactions = short_term[-max_context:] if len(short_term) >= max_context else short_term
        
        # Get relevant past interactions
        relevant = self.manager.retrieve_relevant_interactions(
            message.content,
            similarity_threshold=self.config.memory_similarity_threshold,
            exclude_last_n=len(last_interactions)
        )
        
        # Combine and format context
        context = []
        
        # Add recent interactions
        for interaction in last_interactions:
            context.append({
                'role': 'user',
                'content': interaction['prompt']
            })
            context.append({
                'role': 'assistant',
                'content': interaction['output']
            })
        
        # Add relevant past interactions
        for interaction in relevant:
            context.append({
                'role': 'user',
                'content': interaction['prompt']
            })
            context.append({
                'role': 'assistant',
                'content': interaction['output']
            })
        
        return context
    
    def format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Format context messages into a string for the prompt."""
        formatted = []
        for msg in context:
            role = msg['role'].capitalize()
            content = msg['content']
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
