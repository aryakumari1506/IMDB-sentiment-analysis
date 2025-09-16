# Movie Review Sentiment Analysis

## Overview
Real-time sentiment analysis system for movie reviews using machine learning and NLP, achieving 92% accuracy with Logistic Regression and SVM classifiers.

## Features
- Text preprocessing with NLTK (tokenization, lemmatization, stopwords removal)
- TF-IDF vectorization for feature extraction
- Multiple ML models (Logistic Regression, SVM)
- Interactive web interface for real-time predictions
- Performance visualization (ROC curves, confusion matrix)

## Installation

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

## Project Structure
```
sentiment-analysis/
│
├── app/
│   ├── app.py                # Flask application
│   └── templates/
│       └── index.html        # Web interface
│
├── data/
│   ├── raw/                  # Original IMDB dataset
│   └── processed/            # Cleaned and processed data
│
├── models/                   # Saved trained models
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── data_preprocessing.py # Text cleaning & preprocessing
│   ├── feature_extraction.py # TF-IDF vectorization
│   ├── model_training.py     # Model training pipeline
│   ├── model_evaluation.py   # Performance metrics
│   └── visualization.py      # Results visualization
│
├── main.py                   # Main execution script
├── requirements.txt          # Dependencies
└── README.md
```

## Usage

### Train Model
```bash
python main.py
```

### Start Web Interface
```bash
python app/app.py
```
Access at http://localhost:5000

## Performance Metrics
- Accuracy: 92%
- F1-Score: 0.91
- ROC-AUC: 0.94

## Tech Stack
- Python 3.8+
- scikit-learn
- NLTK
- Flask
- pandas
- matplotlib
- joblib

## Requirements
```
scikit-learn>=0.24.0
nltk>=3.6.0
flask>=2.0.0
pandas>=1.2.0
matplotlib>=3.3.0
joblib>=1.0.0
```
