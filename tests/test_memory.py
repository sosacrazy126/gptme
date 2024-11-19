"""Tests for the memory management functionality."""

import pytest
import time
from pathlib import Path
import shutil
import tempfile

from gptme.config import Config
from gptme.memory import GPTMEMemoryManager
from gptme.message import Message

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def base_config(temp_dir):
    """Create a base test configuration."""
    return Config(
        openai_api_key="test-key",
        model="gpt-4",
        data_dir=temp_dir
    )

@pytest.fixture
def json_config(base_config):
    """Create a config with JSON storage."""
    base_config.memory = {
        "enabled": True,
        "storage_type": "json",
        "similarity_threshold": 40,
        "max_context_window": 5,
        "decay_rate": 0.0001,
    }
    return base_config

@pytest.fixture
def in_memory_config(base_config):
    """Create a config with in-memory storage."""
    base_config.memory = {
        "enabled": True,
        "storage_type": "in_memory",
        "similarity_threshold": 40,
        "max_context_window": 5,
        "decay_rate": 0.0001,
    }
    return base_config

@pytest.fixture
def json_memory_manager(json_config):
    """Create a test memory manager with JSON storage."""
    return GPTMEMemoryManager(json_config)

@pytest.fixture
def in_memory_manager(in_memory_config):
    """Create a test memory manager with in-memory storage."""
    return GPTMEMemoryManager(in_memory_config)

def test_memory_initialization_json(json_memory_manager):
    """Test that memory manager initializes correctly with JSON storage."""
    assert json_memory_manager is not None
    assert json_memory_manager.memory_dir.exists()
    assert json_memory_manager.config.memory_storage_type == "json"

def test_memory_initialization_in_memory(in_memory_manager):
    """Test that memory manager initializes correctly with in-memory storage."""
    assert in_memory_manager is not None
    assert in_memory_manager.config.memory_storage_type == "in_memory"

def test_add_interaction_json(json_memory_manager):
    """Test adding an interaction to JSON storage."""
    message = Message("user", "What is Python?")
    response = "Python is a high-level programming language."
    
    json_memory_manager.add_interaction(message, response)
    
    # Verify interaction was stored
    short_term, _ = json_memory_manager.manager.load_history()
    assert len(short_term) > 0
    assert short_term[-1]['prompt'] == message.content
    assert short_term[-1]['output'] == response

def test_add_interaction_in_memory(in_memory_manager):
    """Test adding an interaction to in-memory storage."""
    message = Message("user", "What is Python?")
    response = "Python is a high-level programming language."
    
    in_memory_manager.add_interaction(message, response)
    
    # Verify interaction was stored
    short_term, _ = in_memory_manager.manager.load_history()
    assert len(short_term) > 0
    assert short_term[-1]['prompt'] == message.content
    assert short_term[-1]['output'] == response

def test_context_window_limit(json_memory_manager):
    """Test that context window size is respected."""
    # Add more interactions than the context window
    max_window = json_memory_manager.config.memory_max_context_window
    for i in range(max_window + 3):
        message = Message("user", f"Test message {i}")
        response = f"Test response {i}"
        json_memory_manager.add_interaction(message, response)
    
    # Get context for a new message
    context = json_memory_manager.get_relevant_context(
        Message("user", "Test query"),
        max_context=max_window
    )
    
    # Verify context size is limited
    assert len(context) <= max_window * 2  # *2 because each interaction has user+assistant messages

def test_memory_decay(json_memory_manager):
    """Test memory decay functionality."""
    # Add an old interaction
    old_message = Message("user", "What is Python?")
    old_response = "Python is a programming language."
    json_memory_manager.add_interaction(old_message, old_response)
    
    # Wait a bit and add a new interaction
    time.sleep(0.1)  # Small delay to test decay
    new_message = Message("user", "How do I use Python lists?")
    new_response = "Lists are created using square brackets."
    json_memory_manager.add_interaction(new_message, new_response)
    
    # Get context for a Python-related query
    context = json_memory_manager.get_relevant_context(
        Message("user", "Tell me about Python")
    )
    
    # Newer interaction should appear before older one due to decay
    newer_found = False
    for msg in context:
        if "lists" in msg['content'].lower():
            newer_found = True
            break
        if "programming language" in msg['content'].lower():
            # Older message found before newer one
            assert False, "Decay not working - older message found before newer one"
    assert newer_found, "Newer message not found in context"

def test_similarity_threshold(json_memory_manager):
    """Test similarity threshold for context retrieval."""
    # Add unrelated interactions
    json_memory_manager.add_interaction(
        Message("user", "What's the weather like?"),
        "I cannot check the current weather."
    )
    json_memory_manager.add_interaction(
        Message("user", "What time is it?"),
        "I cannot tell the current time."
    )
    
    # Add Python-related interaction
    json_memory_manager.add_interaction(
        Message("user", "What is Python?"),
        "Python is a programming language."
    )
    
    # Get context for a Python-related query
    context = json_memory_manager.get_relevant_context(
        Message("user", "How do I write Python code?")
    )
    
    # Should only include relevant (Python-related) context
    assert any("Python" in msg['content'] for msg in context)
    assert not any("weather" in msg['content'].lower() for msg in context)
    assert not any("time" in msg['content'].lower() for msg in context)

def test_format_context(json_memory_manager):
    """Test context formatting."""
    context = [
        {'role': 'user', 'content': 'What is Python?'},
        {'role': 'assistant', 'content': 'Python is a programming language.'},
    ]
    
    formatted = json_memory_manager.format_context_for_prompt(context)
    assert "User: What is Python?" in formatted
    assert "Assistant: Python is a programming language." in formatted

def test_memory_persistence(temp_dir):
    """Test that JSON storage persists between sessions."""
    # Create first memory manager and add interaction
    config1 = Config(
        openai_api_key="test-key",
        model="gpt-4",
        data_dir=temp_dir,
        memory_config={"storage_type": "json"}
    )
    manager1 = GPTMEMemoryManager(config1)
    manager1.add_interaction(
        Message("user", "What is Python?"),
        "Python is a programming language."
    )
    
    # Create new memory manager with same config
    config2 = Config(
        openai_api_key="test-key",
        model="gpt-4",
        data_dir=temp_dir,
        memory_config={"storage_type": "json"}
    )
    manager2 = GPTMEMemoryManager(config2)
    
    # Verify interaction was loaded
    short_term, _ = manager2.manager.load_history()
    assert len(short_term) > 0
    assert "Python" in short_term[0]['prompt']
