"""
1. Data preprocessing
2. Feature extraction
3. Model training
4. Model evaluation
5. Visualization
6. Model saving
"""

import os
import sys
import joblib
import pandas as pd
from datetime import datetime

sys.path.append('src')

from data_preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from model_training import SentimentModelTrainer
from model_evaluation import ModelEvaluator
from visualization import SentimentVisualizer, create_model_comparison_dashboard

def create_directories():
    """Create necessary directories for the project"""
    directories = ['data/raw', 'data/processed', 'models', 'results', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Project directories created successfully!")

def run_complete_pipeline(data_path=r"data\raw\IMDB Dataset.csv", save_results=True):
    """
    Run the complete sentiment analysis pipeline
    
    Args:
        data_path: Path to the dataset (optional)
        save_results: Whether to save results and models
    """
    print("="*60)
    print("SENTIMENT ANALYSIS PROJECT PIPELINE")
    print("="*60)
    
    create_directories()
    
    # Step 1: Data Preprocessing
    print("\n1. DATA PREPROCESSING")
    print("-" * 30)
    
    preprocessor = TextPreprocessor()
    df = preprocessor.load_and_preprocess_data(data_path)
    
    if save_results:
        df.to_csv('data/processed/processed_reviews.csv', index=False)
        print("Processed data saved to data/processed/processed_reviews.csv")
    
    # Step 2: Feature Extraction
    print("\n2. FEATURE EXTRACTION")
    print("-" * 30)
    
    extractor = FeatureExtractor(method='tfidf', max_features=5000, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = extractor.extract_features(
        df['processed_review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    if save_results:
        extractor.save_vectorizer('models/tfidf_vectorizer.pkl')
    
    # Step 3: Model Training
    print("\n3. MODEL TRAINING")
    print("-" * 30)
    
    trainer = SentimentModelTrainer()
    
    print("\nQuick Model Comparison:")
    from model_training import quick_model_comparison
    quick_results = quick_model_comparison(X_train, X_test, y_train, y_test)
    
    print("\nDetailed Training with Hyperparameter Tuning:")
    trained_models = trainer.hyperparameter_tuning(X_train, y_train, cv=5)
    
    cv_results = trainer.cross_validate_models(X_train, y_train, cv=5)
    
    # Select best model
    best_model, best_model_name = trainer.select_best_model(cv_results, metric='f1')
    
    # Step 4: Model Evaluation
    print("\n4. MODEL EVALUATION")
    print("-" * 30)
    
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate_multiple_models(trained_models, X_test, y_test)
    
    # Generate comparison
    comparison_df = evaluator.compare_models(evaluation_results)
    
    # Generate detailed report
    report = evaluator.generate_detailed_report(
        evaluation_results, 
        'results/evaluation_report.txt' if save_results else None
    )
    
    # Step 5: Visualization
    print("\n5. VISUALIZATION")
    print("-" * 30)
    
    visualizer = SentimentVisualizer()
    
    visualizer.create_comprehensive_visualization(
        df['processed_review'], df['sentiment'],
        extractor.vectorizer, best_model, best_model_name
    )
    
    create_model_comparison_dashboard(evaluation_results, y_test)
    
    # Step 6: Save Best Model
    print("\n6. SAVING BEST MODEL")
    print("-" * 30)
    
    if save_results:
        model_filename = f'models/best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
        trainer.save_model(best_model, model_filename, best_model_name)
        
        metadata = {
            'model_name': best_model_name,
            'model_file': model_filename,
            'vectorizer_file': 'models/tfidf_vectorizer.pkl',
            'training_date': datetime.now().isoformat(),
            'performance': {
                'accuracy': evaluation_results[best_model_name]['accuracy'],
                'precision': evaluation_results[best_model_name]['precision'],
                'recall': evaluation_results[best_model_name]['recall'],
                'f1_score': evaluation_results[best_model_name]['f1_score']
            },
            'features': {
                'method': 'tfidf',
                'max_features': 5000,
                'ngram_range': [1, 2]
            }
        }
        
        import json
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Model metadata saved to models/model_metadata.json")
        
        # Save comparison results
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        print("Model comparison saved to results/model_comparison.csv")
    
    # Step 7: Final Summary
    print("\n7. PIPELINE SUMMARY")
    print("-" * 30)
    
    print(f"✓ Dataset processed: {df.shape[0]} samples")
    print(f"✓ Features extracted: {X_train.shape[1]} features")
    print(f"✓ Models trained: {len(trained_models)}")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Best F1-Score: {evaluation_results[best_model_name]['f1_score']:.4f}")
    print(f"✓ Test Accuracy: {evaluation_results[best_model_name]['accuracy']:.4f}")
    
    return {
        'data': df,
        'features': {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        },
        'vectorizer': extractor.vectorizer,
        'models': trained_models,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'evaluation_results': evaluation_results,
        'comparison_df': comparison_df
    }

def load_and_predict(text_input, model_path='models/best_model_logistic_regression.pkl',
                    vectorizer_path='models/tfidf_vectorizer.pkl'):
    """
    Load saved model and make predictions on new text
    
    Args:
        text_input: Text to analyze (string or list of strings)
        model_path: Path to saved model
        vectorizer_path: Path to saved vectorizer
        
    Returns:
        Predictions and probabilities
    """
    try:
        # Load model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        
        if isinstance(text_input, str):
            text_input = [text_input]
        
        processed_texts = [preprocessor.preprocess_text(text) for text in text_input]
        
        # Transform text
        X = vectorizer.transform(processed_texts)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        results = []
        for i, (text, pred, prob) in enumerate(zip(text_input, predictions, probabilities)):
            sentiment = "Positive" if pred == 1 else "Negative"
            confidence = max(prob)
            
            results.append({
                'text': text[:100] + "..." if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_prob': prob[1],
                'negative_prob': prob[0]
            })
        
        return results
    
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run the training pipeline first!")
        return None
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def demo_predictions():
    """Demonstrate predictions on sample texts"""
    print("\n" + "="*50)
    print("DEMO: PREDICTIONS ON SAMPLE TEXTS")
    print("="*50)
    
    sample_texts = [
        "This movie is absolutely fantastic! Great acting and amazing plot.",
        "I loved every minute of this film. Brilliant cinematography and direction.",
        "This movie is terrible. Poor acting and boring plot.",
        "Waste of time and money. One of the worst films I've ever seen.",
        "The movie was okay, nothing special but not bad either.",
        "Amazing special effects but the story was confusing."
    ]
    
    results = load_and_predict(sample_texts)
    
    if results:
        print("\nPrediction Results:")
        print("-" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. Text: {result['text']}")
            print(f"   Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
            print(f"   Positive: {result['positive_prob']:.3f}, Negative: {result['negative_prob']:.3f}")
            print()
    else:
        print("Could not load model for predictions. Train the model first!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--data', type=str, help='Path to dataset CSV file')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results and models')
    parser.add_argument('--demo', action='store_true', help='Run demo predictions only')
    parser.add_argument('--quick', action='store_true', help='Run quick pipeline without hyperparameter tuning')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_predictions()
    else:
        # Run the complete pipeline
        results = run_complete_pipeline(
            data_path=r"data\raw\IMDB Dataset.csv",
            save_results=not args.no_save
        )
        
        # Run demo predictions if models were saved
        if not args.no_save:
            demo_predictions()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")

        print("="*60)
