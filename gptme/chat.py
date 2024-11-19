"""Chat interaction module for gptme."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from .config import Config
from .llm import LLM
from .memory import GPTMEMemoryManager
from .message import Message

class Chat:
    """Manages chat interactions with the LLM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLM(config)
        self.memory_manager = GPTMEMemoryManager(config)
    
    def get_response(self, message: Message) -> str:
        """Get a response from the LLM with context from memory."""
        # Get relevant context from memory
        context = self.memory_manager.get_relevant_context(message)
        
        # Format context for the prompt
        context_str = self.memory_manager.format_context_for_prompt(context)
        
        # Create the full prompt with context
        full_prompt = f"{context_str}\n\nCurrent message: {message.content}"
        
        # Get response from LLM
        response = self.llm.complete(full_prompt)
        
        # Store interaction in memory
        self.memory_manager.add_interaction(message, response)
        
        return response
    
    def process_message(self, message: Message) -> str:
        """Process a message and return the response.
        
        This is the main entry point for chat interactions. It handles:
        1. Getting relevant context from memory
        2. Generating a response using the LLM
        3. Storing the interaction in memory
        """
        try:
            response = self.get_response(message)
            return response
        except Exception as e:
            # Log error and return error message
            error_msg = f"Error processing message: {str(e)}"
            return error_msg
    
    def clear_memory(self) -> None:
        """Clear the memory store and start fresh."""
        self.memory_manager = GPTMEMemoryManager(self.config)

def chat(
    prompt_msgs: List[Message],
    initial_msgs: List[Message],
    logdir: Path,
    model: Optional[str],
    stream: bool,
    no_confirm: bool,
    interactive: bool,
    show_hidden: bool,
    workspace_path: Optional[Path],
    tool_allowlist: Optional[List[str]],
) -> None:
    """Main chat function that processes messages and manages the conversation.
    
    Args:
        prompt_msgs: List of initial prompt messages to process
        initial_msgs: List of system messages to initialize the conversation
        logdir: Directory to store conversation logs
        model: Name of the model to use
        stream: Whether to stream responses
        no_confirm: Skip confirmation prompts
        interactive: Whether to run in interactive mode
        show_hidden: Show hidden system messages
        workspace_path: Path to workspace directory
        tool_allowlist: List of allowed tools
    """
    config = Config(
        model=model,
        stream=stream,
        no_confirm=no_confirm,
        interactive=interactive,
        show_hidden=show_hidden,
        workspace_path=workspace_path,
        tool_allowlist=tool_allowlist,
    )
    
    chat_instance = Chat(config)
    
    # Process initial system messages
    for msg in initial_msgs:
        chat_instance.process_message(msg)
    
    # Process prompt messages
    for msg in prompt_msgs:
        chat_instance.process_message(msg)
    
    # If interactive, start chat loop
    if interactive:
        while True:
            try:
                # Get user input
                user_input = input("> ")
                if not user_input:
                    continue
                
                # Process user message
                msg = Message("user", user_input)
                response = chat_instance.process_message(msg)
                print(response)
                
            except (KeyboardInterrupt, EOFError):
                break
