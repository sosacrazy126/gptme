"""LLM interaction module for gptme."""

import logging
import shutil
import sys
from collections.abc import Iterator
from functools import lru_cache
from typing import cast, Optional, List

from rich import print

from .config import get_config, Config
from .constants import PROMPT_ASSISTANT
from .llm_anthropic import chat as chat_anthropic
from .llm_anthropic import get_client as get_anthropic_client
from .llm_anthropic import init as init_anthropic
from .llm_anthropic import stream as stream_anthropic
from .llm_openai import chat as chat_openai
from .llm_openai import get_client as get_openai_client
from .llm_openai import init as init_openai
from .llm_openai import stream as stream_openai
from .message import Message, format_msgs, len_tokens
from .models import (
    MODELS,
    PROVIDERS_OPENAI,
    Provider,
    get_summary_model,
)
from .tools import ToolUse
from .util import console
from .memory import GPTMEMemoryManager

logger = logging.getLogger(__name__)

class LLM:
    """Manages interactions with language models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.provider = self._init_provider()
        self.memory_manager = GPTMEMemoryManager(config) if config.memory_enabled else None
    
    def _init_provider(self) -> Provider:
        """Initialize the LLM provider based on configuration."""
        config = self.config
        
        if not config.model:
            raise ValueError("No model specified in configuration")
        
        # Determine provider from model name
        if config.model.startswith("gpt-"):
            provider = cast(Provider, "openai")
            init_openai(provider, config)
            assert get_openai_client()
        elif config.model.startswith("claude-"):
            provider = cast(Provider, "anthropic")
            init_anthropic(config)
            assert get_anthropic_client()
        else:
            raise ValueError(f"Unknown model type: {config.model}")
        
        return provider
    
    def complete(self, prompt: str, stream: bool = False) -> str:
        """Complete a prompt using the configured LLM."""
        messages = [Message("user", prompt)]
        if stream:
            return self._reply_stream(messages)
        else:
            print(f"{PROMPT_ASSISTANT}: Thinking...", end="\r")
            return self._chat_complete(messages)
    
    def _prepare_messages_with_context(self, messages: List[Message]) -> List[Message]:
        """Prepare messages with relevant context from memory."""
        if not self.memory_manager:
            return messages
            
        # Get relevant context for the last user message
        context = self.memory_manager.get_relevant_context(
            messages[-1], 
            max_context=self.config.memory_max_context_window
        )
        if not context:
            return messages
            
        # Format context as a system message
        context_str = self.memory_manager.format_context_for_prompt(context)
        context_msg = Message(
            "system",
            "Here is some relevant context from previous conversations:\n\n" + context_str
        )
        
        # Insert context before user messages
        return [context_msg] + messages
    
    def _chat_complete(self, messages: List[Message]) -> str:
        """Complete a chat conversation with memory context."""
        response = ""
        if self.provider == "openai":
            response = chat_openai(messages, self.config)
        elif self.provider == "anthropic":
            response = chat_anthropic(messages, self.config)

        # Store interaction in memory if enabled
        if self.memory_manager and len(messages) > 0:
            self.memory_manager.add_interaction(messages[-1], response)

        return response
    
    def _stream(self, messages: List[Message]) -> Iterator[str]:
        """Stream a chat conversation."""
        # Add context from memory if enabled
        messages_with_context = self._prepare_messages_with_context(messages)
        
        if self.provider in PROVIDERS_OPENAI:
            return stream_openai(messages_with_context, self.config.model)
        elif self.provider == "anthropic":
            return stream_anthropic(messages_with_context, self.config.model)
        else:
            raise ValueError("LLM not initialized")
    
    def _reply_stream(self, messages: List[Message]) -> Iterator[str]:
        """Stream a chat response with memory context."""
        response_stream = None
        if self.provider == "openai":
            response_stream = stream_openai(messages, self.config)
        elif self.provider == "anthropic":
            response_stream = stream_anthropic(messages, self.config)

        # Collect full response for memory storage
        full_response = []
        for chunk in response_stream:
            full_response.append(chunk)
            yield chunk

        # Store interaction in memory if enabled
        if self.memory_manager and len(messages) > 0:
            self.memory_manager.add_interaction(messages[-1], "".join(full_response))
    
    def summarize(self, content: str) -> str:
        """Summarize content using a cheaper model."""
        messages = [
            Message(
                "system",
                "You are a helpful assistant that helps summarize messages with an AI assistant through a tool called gptme.",
            ),
            Message("user", f"Summarize this:\n{content}"),
        ]
        
        model = get_summary_model(self.provider)
        context_limit = MODELS[self.provider][model]["context"]
        if len_tokens(messages) > context_limit:
            raise ValueError(
                f"Cannot summarize more than {context_limit} tokens, got {len_tokens(messages)}"
            )
        
        summary = self._chat_complete(messages)
        assert summary
        logger.debug(
            f"Summarized long output ({len_tokens(content)} -> {len_tokens(summary)} tokens): "
            + summary
        )
        return summary
    
    def generate_name(self, msgs: List[Message]) -> str:
        """Generate a name for a conversation."""
        # filter out system messages
        msgs = [m for m in msgs if m.role != "system"]
        
        msgs = (
            [
                Message(
                    "system",
                    """
The following is a conversation between a user and an assistant.
You should generate a descriptive name for it.

The name should be 3-6 words describing the conversation, separated by dashes. Examples:
 - install-llama
 - implement-game-of-life
 - capitalize-words-in-python

Focus on the main and/or initial topic of the conversation. Avoid using names that are too generic or too specific.

IMPORTANT: output only the name, no preamble or postamble.
""",
                )
            ]
            + msgs
            + [
                Message(
                    "user",
                    "That was the context of the conversation. Now, answer with a descriptive name for this conversation according to system instructions.",
                )
            ]
        )
        name = self._chat_complete(msgs).strip()
        return name

# Legacy functions for backward compatibility
def init_llm(llm: str):
    config = get_config()
    LLM(config)._init_provider()

def reply(messages: List[Message], model: str, stream: bool = False) -> Message:
    config = get_config()
    config.model = model
    llm = LLM(config)
    
    if stream:
        return llm._reply_stream(messages)
    else:
        response = llm._chat_complete(messages)
        return Message("assistant", response)

def generate_name(msgs: List[Message]) -> str:
    config = get_config()
    llm = LLM(config)
    return llm.generate_name(msgs)

def summarize(msg: str | Message | List[Message]) -> Message:
    """Uses a cheap LLM to summarize long outputs."""
    config = get_config()
    llm = LLM(config)
    
    # construct plaintext from message(s)
    if isinstance(msg, str):
        content = msg
    elif isinstance(msg, Message):
        content = msg.content
    else:
        content = "\n".join(format_msgs(msg))
    
    logger.info(f"{content[:200]=}")
    summary = _summarize_helper(llm, content)
    logger.info(f"{summary[:200]=}")
    
    # construct message from summary
    content = f"Here's a summary of the conversation:\n{summary}"
    return Message(role="system", content=content)

@lru_cache(maxsize=128)
def _summarize_helper(llm: LLM, s: str, tok_max_start=400, tok_max_end=400) -> str:
    """Helper function for summarizing long outputs."""
    if len_tokens(s) > tok_max_start + tok_max_end:
        beginning = " ".join(s.split()[:tok_max_start])
        end = " ".join(s.split()[-tok_max_end:])
        summary = llm.summarize(beginning + "\n...\n" + end)
    else:
        summary = llm.summarize(s)
    return summary
