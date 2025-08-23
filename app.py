from flask import Flask, request, jsonify
from flask_cors import CORS
from model import EmotionDetectionModel
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the model
model = EmotionDetectionModel()

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Emotion Detection API',
        'status': 'running',
        'endpoints': {
            'predict': '/predict',
            'health': '/health',
            'model_info': '/model-info'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        if model.model is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Model not loaded',
                'error': 'Model needs to be trained or loaded'
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'message': 'Model is loaded and ready',
            'model_loaded': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'message': 'Health check failed',
            'error': str(e)
        }), 500

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    try:
        if model.model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 400
        
        # Get emotion classes
        emotion_classes = list(model.emotion_mapping.keys()) if model.emotion_mapping else []
        
        return jsonify({
            'status': 'success',
            'model_type': 'Logistic Regression',
            'vectorizer_type': 'TF-IDF',
            'emotion_classes': emotion_classes,
            'num_classes': len(emotion_classes),
            'model_loaded': True
        })
    except Exception as e:
        logger.error(f"Model info failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get model info',
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion for given text"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Text field is required'
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                'status': 'error',
                'message': 'Text cannot be empty'
            }), 400
        
        # Check if model is loaded
        if model.model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please train or load the model first.'
            }), 503
        
        # Make prediction
        result = model.predict_emotion(text)
        
        return jsonify({
            'status': 'success',
            'input_text': text,
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Prediction failed',
            'error': str(e)
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with the dataset"""
    try:
        # Check if training data exists
        if not os.path.exists('train.txt'):
            return jsonify({
                'status': 'error',
                'message': 'Training data (train.txt) not found'
            }), 404
        
        # Train the model
        logger.info("Starting model training...")
        accuracy = model.train_model()
        
        # Save the trained model
        model.save_model()
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'accuracy': accuracy,
            'model_saved': True
        })
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Training failed',
            'error': str(e)
        }), 500

@app.route('/load-model', methods=['POST'])
def load_model():
    """Load a pre-trained model"""
    try:
        # Check if model files exist
        required_files = ['emotion_model.pkl', 'vectorizer.pkl', 'emotion_mapping.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': 'Model files not found',
                'missing_files': missing_files
            }), 404
        
        # Load the model
        model.load_model()
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_loaded': True
        })
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to load model',
            'error': str(e)
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict emotions for multiple texts"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Texts array is required'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Texts must be a non-empty array'
            }), 400
        
        # Check if model is loaded
        if model.model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please train or load the model first.'
            }), 503
        
        # Make predictions for all texts
        results = []
        for text in texts:
            try:
                result = model.predict_emotion(text)
                results.append({
                    'input_text': text,
                    'prediction': result
                })
            except Exception as e:
                results.append({
                    'input_text': text,
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'predictions': results,
            'total_texts': len(texts)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Batch prediction failed',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Try to load existing model if available
    try:
        if os.path.exists('emotion_model.pkl'):
            logger.info("Loading existing model...")
            model.load_model()
            logger.info("Model loaded successfully!")
        else:
            logger.info("No existing model found. Please train the model first.")
    except Exception as e:
        logger.warning(f"Could not load existing model: {str(e)}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 