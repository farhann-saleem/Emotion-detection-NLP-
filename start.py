#!/usr/bin/env python3
"""
Startup script for Emotion Detection AI
This script will train the model and start the Flask server
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import sklearn
        import pandas
        import numpy
        import nltk
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def train_model():
    """Train the emotion detection model"""
    try:
        print("🤖 Training emotion detection model...")
        print("This may take 2-5 minutes depending on your hardware...")
        
        # Import and train the model
        from model import EmotionDetectionModel
        
        model = EmotionDetectionModel()
        accuracy = model.train_model()
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Save the model
        model.save_model()
        print("💾 Model saved to disk")
        
        return True
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        return False

def start_server():
    """Start the Flask server"""
    try:
        print("🚀 Starting Flask server...")
        print("🌐 Server will be available at: http://localhost:5000")
        print("📱 Open index.html in your browser to use the web interface")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main startup function"""
    print("🎭 Emotion Detection AI - Startup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("train.txt").exists():
        print("❌ Error: train.txt not found!")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  Warning: NLTK data download failed, continuing anyway...")
    
    # Check if model already exists
    model_files = ["emotion_model.pkl", "vectorizer.pkl", "emotion_mapping.pkl"]
    model_exists = all(Path(f).exists() for f in model_files)
    
    if model_exists:
        print("✅ Pre-trained model found!")
        choice = input("Do you want to retrain the model? (y/N): ").lower().strip()
        if choice == 'y':
            if not train_model():
                sys.exit(1)
    else:
        print("📝 No pre-trained model found")
        choice = input("Do you want to train a new model? (Y/n): ").lower().strip()
        if choice != 'n':
            if not train_model():
                sys.exit(1)
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
