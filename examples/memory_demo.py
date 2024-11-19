"""Demonstration of GPTME's memory management capabilities."""

import os
import time
from pathlib import Path
import sys

# Add the parent directory to the Python path to import gptme
sys.path.insert(0, str(Path(__file__).parent.parent))

from gptme.config import Config
from gptme.chat import Chat
from gptme.message import Message

def demonstrate_basic_memory(chat):
    """Demonstrate basic memory capabilities with context retention."""
    print("\n1. Basic Memory and Context Retention")
    print("=" * 80)
    print("This section demonstrates how the system maintains context across conversations.")
    
    conversations = [
        "What is Python?",
        "How do I create a list in Python?",
        "What's the weather like in Paris?",  # Unrelated question
        "How do I append items to a list in Python?",  # Should recall previous Python context
        "Tell me more about Python lists.",  # Should have full context of previous list discussions
    ]
    
    for i, prompt in enumerate(conversations, 1):
        print(f"\nInteraction {i}:")
        print(f"User: {prompt}")
        
        message = Message(content=prompt)
        response = chat.process_message(message)
        
        print(f"Assistant: {response}")
        print("-" * 80)
        
        if i == 3:
            print("\nNote: The above question was unrelated to Python. Watch how the system")
            print("maintains Python context in the next programming-related question.")
        elif i == 4:
            print("\nNote: Notice how the response incorporates knowledge from previous")
            print("Python-related discussions, showing effective context maintenance.")

def demonstrate_memory_decay(chat):
    """Demonstrate how memory decay affects context retrieval."""
    print("\n2. Memory Decay")
    print("=" * 80)
    print("This section demonstrates how older memories become less prominent over time.")
    
    print("\nFirst interaction (older memory):")
    message1 = Message(content="What is a Python dictionary?")
    response1 = chat.process_message(message1)
    print(f"User: {message1.content}")
    print(f"Assistant: {response1}")
    
    # Wait to demonstrate decay
    print("\nWaiting a moment to demonstrate decay effect...")
    time.sleep(2)
    
    print("\nSecond interaction (newer memory):")
    message2 = Message(content="How do I create a Python list?")
    response2 = chat.process_message(message2)
    print(f"User: {message2.content}")
    print(f"Assistant: {response2}")
    
    print("\nFinal query to test memory prominence:")
    message3 = Message(content="Tell me about Python data structures")
    response3 = chat.process_message(message3)
    print(f"User: {message3.content}")
    print(f"Assistant: {response3}")
    print("\nNote: Notice how the more recent list discussion might be more prominent in the response.")

def demonstrate_context_window(chat):
    """Demonstrate context window limits."""
    print("\n3. Context Window Management")
    print("=" * 80)
    print("This section demonstrates how the system manages the context window size.")
    
    # Generate several quick interactions
    topics = [
        "What is Python?",
        "What are variables?",
        "How do loops work?",
        "What are functions?",
        "What are classes?",
        "What is inheritance?",
    ]
    
    print("\nGenerating multiple interactions...")
    for topic in topics:
        message = Message(content=topic)
        response = chat.process_message(message)
        print(f"Topic: {topic}")
    
    print("\nNow asking a question that should focus on recent context:")
    final_message = Message(content="Can you summarize what we've discussed about Python?")
    final_response = chat.process_message(final_message)
    print(f"User: {final_message.content}")
    print(f"Assistant: {final_response}")
    print("\nNote: The response should focus on the most recent and relevant discussions.")

def main():
    # Load test configuration
    config_path = Path(__file__).parent / "test_config.toml"
    if not config_path.exists():
        print(f"Error: Test configuration file not found at {config_path}")
        sys.exit(1)
    
    # Initialize with test configuration
    config = Config()
    
    # Ensure we have an API key
    if not config.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Configure memory settings
    config.memory = {
        "enabled": True,
        "storage_type": "json",  # Can be changed to "in_memory" for comparison
        "similarity_threshold": 40,
        "max_context_window": 5,
        "decay_rate": 0.0001
    }
    
    # Initialize chat
    chat = Chat(config)
    
    print("\nGPTME Memory Management Demonstration")
    print("=" * 80)
    print("\nThis demo showcases various aspects of GPTME's memory system:")
    print("1. Basic memory and context retention")
    print("2. Memory decay over time")
    print("3. Context window management")
    print("4. Semantic relevance and context maintenance")
    print("\nCurrent Configuration:")
    print(f"- Storage Type: {config.memory_storage_type}")
    print(f"- Similarity Threshold: {config.memory_similarity_threshold}")
    print(f"- Max Context Window: {config.memory_max_context_window}")
    print(f"- Decay Rate: {config.memory_decay_rate}")
    
    # Run demonstrations
    demonstrate_basic_memory(chat)
    demonstrate_memory_decay(chat)
    demonstrate_context_window(chat)
    
    print("\nDemo completed!")
    print("=" * 80)
    print("\nKey points demonstrated:")
    print("1. Short-term memory retention")
    print("2. Context-aware responses")
    print("3. Memory decay over time")
    print("4. Context window management")
    print("5. Semantic relevance")
    print("\nTry modifying the configuration settings to see how they affect the system's behavior:")
    print("- Change storage_type to 'in_memory' for temporary storage")
    print("- Adjust similarity_threshold to control context relevance")
    print("- Modify max_context_window to change the amount of context maintained")
    print("- Adjust decay_rate to change how quickly older memories fade")

if __name__ == "__main__":
    main()
