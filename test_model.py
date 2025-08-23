#!/usr/bin/env python3
"""
Test script for Emotion Detection AI
This script tests the model functionality without starting the server
"""

import os
import sys
from pathlib import Path

def test_model():
    """Test the emotion detection model"""
    print("🧪 Testing Emotion Detection Model")
    print("=" * 40)
    
    try:
        # Import the model
        from model import EmotionDetectionModel
        
        # Initialize model
        model = EmotionDetectionModel()
        print("✅ Model class imported successfully")
        
        # Check if model files exist
        model_files = ["emotion_model.pkl", "vectorizer.pkl", "emotion_mapping.pkl"]
        model_exists = all(Path(f).exists() for f in model_files)
        
        if model_exists:
            print("✅ Pre-trained model files found")
            
            # Load the model
            model.load_model()
            print("✅ Model loaded successfully")
            
            # Test predictions
            test_texts = [
                "I am feeling so happy today!",
                "This makes me really angry",
                "I'm scared about what might happen",
                "I love this beautiful day",
                "I feel surprised by the news"
            ]
            
            print("\n📝 Testing predictions:")
            print("-" * 40)
            
            for text in test_texts:
                try:
                    result = model.predict_emotion(text)
                    print(f"Text: '{text}'")
                    print(f"  → Emotion: {result['emotion']}")
                    print(f"  → Confidence: {result['confidence']:.3f}")
                    print(f"  → Processed: '{result['processed_text']}'")
                    print()
                except Exception as e:
                    print(f"❌ Failed to predict for '{text}': {e}")
            
            print("✅ All tests passed!")
            
        else:
            print("📝 No pre-trained model found")
            print("Training a new model...")
            
            # Train the model
            accuracy = model.train_model()
            print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
            
            # Save the model
            model.save_model()
            print("✅ Model saved to disk")
            
            # Test a simple prediction
            result = model.predict_emotion("I am happy!")
            print(f"✅ Test prediction: {result['emotion']} (confidence: {result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the API endpoints (requires server to be running)"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 40)
    
    try:
        import requests
        
        base_url = "http://localhost:5000"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check: {data['status']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Health check: Server not running (expected)")
        
        # Test model info endpoint
        try:
            response = requests.get(f"{base_url}/model-info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info: {data['status']}")
            else:
                print(f"❌ Model info failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Model info: Server not running (expected)")
        
        print("ℹ️  To test API endpoints, start the server with 'python app.py'")
        
    except ImportError:
        print("⚠️  Requests library not installed, skipping API tests")
        print("Install with: pip install requests")

def main():
    """Main test function"""
    print("🎭 Emotion Detection AI - Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("train.txt").exists():
        print("❌ Error: train.txt not found!")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Test the model
    if test_model():
        print("\n🎉 Model tests completed successfully!")
    else:
        print("\n❌ Model tests failed!")
        sys.exit(1)
    
    # Test API endpoints
    test_api_endpoints()
    
    print("\n✨ All tests completed!")
    print("🚀 To start the server, run: python app.py")
    print("📱 Then open index.html in your browser")

if __name__ == "__main__":
    main()
