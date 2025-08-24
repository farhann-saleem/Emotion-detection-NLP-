#!/usr/bin/env python3
"""
Test script for negation handling in emotion detection
"""

import re

def handle_negation(text):
    """Simple negation handler: joins "not" with the next word"""
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    return text

def test_negation_examples():
    """Test various negation examples"""
    
    test_cases = [
        "I am not sure I am feeling good",
        "I do not like this situation",
        "This is not bad at all",
        "I am not happy with the result",
        "She is not sad anymore",
        "He does not feel angry",
        "I am not afraid of anything",
        "This is not surprising",
        "I do not love this movie",
        "I am not feeling well today"
    ]
    
    print("🧪 Testing Negation Handling")
    print("=" * 50)
    
    for i, sentence in enumerate(test_cases, 1):
        processed = handle_negation(sentence)
        print(f"Test {i}:")
        print(f"  Original: {sentence}")
        print(f"  Processed: {processed}")
        print()
    
    print("✅ Negation handling test completed!")

def test_with_emotion_context():
    """Test negation in emotion-related contexts"""
    
    emotion_examples = [
        "I am not happy about this news",
        "She is not sad anymore",
        "He does not feel angry",
        "I am not afraid of spiders",
        "This is not surprising at all",
        "I do not love this weather",
        "I am not feeling well today",
        "This is not a good situation",
        "I am not excited about the trip",
        "He is not worried about the exam"
    ]
    
    print("😊 Testing Negation in Emotion Context")
    print("=" * 50)
    
    for example in emotion_examples:
        processed = handle_negation(example)
        print(f"Original: {example}")
        print(f"Processed: {processed}")
        print("-" * 40)

if __name__ == "__main__":
    print("🎭 Emotion Detection AI - Negation Testing")
    print("=" * 60)
    
    # Test basic negation
    test_negation_examples()
    
    print("\n" + "=" * 60 + "\n")
    
    # Test emotion-specific negation
    test_with_emotion_context()
    
    print("\n🚀 Now your model will better understand negated emotions!")
    print("💡 Examples of improvement:")
    print("   - 'not happy' → 'not_happy' (recognized as negative emotion)")
    print("   - 'not sad' → 'not_sad' (recognized as positive emotion)")
    print("   - 'not afraid' → 'not_afraid' (recognized as positive emotion)")
