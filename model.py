import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

class EmotionDetectionModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.emotion_mapping = {}
        self.reverse_emotion_mapping = {}
        
    def preprocess_text(self, text):
        """Preprocess text similar to training data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = ''.join([i for i in text if not i.isdigit()])
        
        # Remove non-ASCII characters (emojis, etc.)
        text = ''.join([i for i in text if i.isascii()])
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = text.split()
            cleaned = [i for i in words if i.lower() not in stop_words]
            text = ' '.join(cleaned)
        except:
            # If NLTK data is not available, skip stopword removal
            pass
            
        return text
    
    def train_model(self, data_path='train.txt'):
        """Train the emotion detection model"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(data_path, sep=';', header=None, names=['text', 'emotions'])
        
        # Create emotion mapping
        unique_emotions = df['emotions'].unique()
        for i, emo in enumerate(unique_emotions):
            self.emotion_mapping[emo] = i
            self.reverse_emotion_mapping[i] = emo
        
        # Convert emotions to numbers
        df['emotions'] = df['emotions'].map(self.emotion_mapping)
        
        # Preprocess text
        df['text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['emotions'], random_state=42, test_size=0.20
        )
        
        # Vectorize text using TF-IDF
        print("Vectorizing text...")
        self.vectorizer = TfidfVectorizer()
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train logistic regression model
        print("Training logistic regression model...")
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_vectorized, y_train)
        
        # Evaluate model
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = self.model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(self.emotion_mapping.keys())))
        
        return accuracy
    
    def predict_emotion(self, text):
        """Predict emotion for given text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        probability = np.max(self.model.predict_proba(text_vectorized))
        
        # Get emotion label
        emotion = self.reverse_emotion_mapping[prediction]
        
        return {
            'emotion': emotion,
            'confidence': float(probability),
            'processed_text': processed_text
        }
    
    def save_model(self, model_path='emotion_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save the trained model and vectorizer"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save emotion mapping
        mapping_path = 'emotion_mapping.pkl'
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'emotion_mapping': self.emotion_mapping,
                'reverse_emotion_mapping': self.reverse_emotion_mapping
            }, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Emotion mapping saved to {mapping_path}")
    
    def load_model(self, model_path='emotion_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load a trained model and vectorizer"""
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model files not found. Please train the model first.")
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load emotion mapping
        mapping_path = 'emotion_mapping.pkl'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                mapping_data = pickle.load(f)
                self.emotion_mapping = mapping_data['emotion_mapping']
                self.reverse_emotion_mapping = mapping_data['reverse_emotion_mapping']
        
        print("Model loaded successfully!")

if __name__ == "__main__":
    # Initialize and train the model
    model = EmotionDetectionModel()
    
    # Train the model (this will take some time)
    accuracy = model.train_model()
    
    # Save the trained model
    model.save_model()
    
    # Test the model
    test_texts = [
        "I am feeling so happy today!",
        "This makes me really angry",
        "I'm scared about what might happen",
        "I love this beautiful day"
    ]
    
    print("\nTesting the model:")
    for text in test_texts:
        result = model.predict_emotion(text)
        print(f"Text: '{text}'")
        print(f"Predicted emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
        print(f"Processed text: '{result['processed_text']}'")
        print("-" * 50) 