"""
Feature Extraction Module for Sentiment Analysis
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

class FeatureExtractor:
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize feature extractor
        
        Args:
            method: 'tfidf' or 'count'
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
    def create_vectorizer(self):
        """Create and return appropriate vectorizer"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                lowercase=True,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                lowercase=True,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError("Method must be 'tfidf' or 'count'")
        
        return self.vectorizer
    
    def extract_features(self, texts, labels=None, test_size=0.2, random_state=42):
        """
        Extract features from text data
        
        Args:
            texts: List or Series of text data
            labels: List or Series of labels (optional)
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test (if labels provided)
            X_vectorized (if labels not provided)
        """
        # Create vectorizer if not exists
        if self.vectorizer is None:
            self.create_vectorizer()
        
        # Split data if labels provided
        if labels is not None:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
            
            # Fit vectorizer on training data and transform both sets
            X_train = self.vectorizer.fit_transform(X_train_text)
            X_test = self.vectorizer.transform(X_test_text)
            
            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Number of features: {len(self.vectorizer.get_feature_names_out())}")
            
            return X_train, X_test, y_train, y_test
        else:
            # Just vectorize the data
            X_vectorized = self.vectorizer.fit_transform(texts)
            print(f"Vectorized data shape: {X_vectorized.shape}")
            return X_vectorized
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        if self.vectorizer is not None:
            return self.vectorizer.get_feature_names_out()
        return None
    
    def save_vectorizer(self, filepath):
        """Save the trained vectorizer"""
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, filepath)
            print(f"Vectorizer saved to {filepath}")
        else:
            print("No vectorizer to save. Train the vectorizer first.")
    
    def load_vectorizer(self, filepath):
        """Load a trained vectorizer"""
        self.vectorizer = joblib.load(filepath)
        print(f"Vectorizer loaded from {filepath}")
    
    def transform_text(self, texts):
        """Transform new text using trained vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not trained. Train or load a vectorizer first.")
        return self.vectorizer.transform(texts)
    
    def get_top_features(self, feature_names, coefficients, top_n=20):
        """
        Get top features based on model coefficients
        
        Args:
            feature_names: Array of feature names
            coefficients: Model coefficients
            top_n: Number of top features to return
            
        Returns:
            Dictionary with positive and negative features
        """
        # Get indices of top positive and negative features
        top_pos_indices = np.argsort(coefficients)[-top_n:]
        top_neg_indices = np.argsort(coefficients)[:top_n]
        
        top_positive = [(feature_names[i], coefficients[i]) for i in reversed(top_pos_indices)]
        top_negative = [(feature_names[i], coefficients[i]) for i in top_neg_indices]
        
        return {
            'positive': top_positive,
            'negative': top_negative
        }

def analyze_features(vectorizer, texts, labels):
    """Analyze feature distribution and statistics"""
    print("\n=== Feature Analysis ===")
    
    # Transform texts
    X = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Feature statistics
    print(f"Total features: {len(feature_names)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Matrix density: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")
    
    # Most frequent features
    feature_sums = np.array(X.sum(axis=0))[0]
    top_indices = np.argsort(feature_sums)[-20:]
    
    print("\nTop 20 most frequent features:")
    for i in reversed(top_indices):
        print(f"{feature_names[i]}: {feature_sums[i]:.2f}")
    
    return X, feature_names

if __name__ == "__main__":
    # Test feature extraction
    from data_preprocessing import TextPreprocessor
    
    # Load and preprocess data
    preprocessor = TextPreprocessor()
    df = preprocessor.load_and_preprocess_data()
    
    # Extract features
    extractor = FeatureExtractor(method='tfidf', max_features=3000)
    X_train, X_test, y_train, y_test = extractor.extract_features(
        df['processed_review'], df['sentiment']
    )
    
    # Analyze features
    analyze_features(extractor.vectorizer, df['processed_review'], df['sentiment'])