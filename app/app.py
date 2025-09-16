from flask import Flask, render_template, request, jsonify
import joblib
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import TextPreprocessor

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
preprocessor = None

def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    global model, vectorizer, preprocessor
    
    try:
        # Try to load the best model
        model_path = os.path.join('..', 'models', 'best_model_logistic_regression.pkl')
        vectorizer_path = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            # Look for any model files
            models_dir = os.path.join('..', 'models')
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
                if model_files:
                    model_path = os.path.join(models_dir, model_files[0])
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        preprocessor = TextPreprocessor()
        
        print(f"Model loaded successfully from {model_path}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run the training pipeline first!")
        return False

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for given text"""
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Preprocess text
        processed_text = preprocessor.preprocess_text(text)
        
        if not processed_text:
            return jsonify({'error': 'Text could not be processed'}), 400
        
        # Transform text
        X = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Prepare response
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = float(max(probabilities))
        positive_prob = float(probabilities[1])
        negative_prob = float(probabilities[0])
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_probability': positive_prob,
            'negative_probability': negative_prob,
            'processed_text': processed_text
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple texts"""
    try:
        # Get texts from request
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'No texts provided or invalid format'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        results = []
        
        for text in texts:
            if not text.strip():
                continue
                
            # Preprocess text
            processed_text = preprocessor.preprocess_text(text)
            
            if not processed_text:
                results.append({
                    'original_text': text[:100] + "..." if len(text) > 100 else text,
                    'error': 'Could not process text'
                })
                continue
            
            # Transform text
            X = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Prepare result
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = float(max(probabilities))
            
            results.append({
                'original_text': text[:100] + "..." if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_probability': float(probabilities[1]),
                'negative_probability': float(probabilities[0])
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    try:
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Try to load metadata
        metadata_path = os.path.join('..', 'models', 'model_metadata.json')
        metadata = {}
        
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Basic model info
        model_info = {
            'model_type': type(model).__name__,
            'vectorizer_type': type(vectorizer).__name__,
            'feature_count': len(vectorizer.get_feature_names_out()),
            'model_loaded': True
        }
        
        # Add metadata if available
        if metadata:
            model_info.update(metadata)
        
        return jsonify(model_info)
        
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and vectorizer is not None
    })

if __name__ == '__main__':
    # Load model on startup
    model_loaded = load_model_and_vectorizer()
    
    if not model_loaded:
        print("Warning: Model could not be loaded. Some endpoints will not work.")
        print("Please run 'python main.py' to train the model first.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
