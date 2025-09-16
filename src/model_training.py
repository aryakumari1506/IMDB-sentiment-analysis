import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time

class SentimentModelTrainer:
    def __init__(self):
        """Initialize model trainer with faster algorithms"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, n_jobs=-1
            ),
            'Multinomial Naive Bayes': MultinomialNB(),
            # switched from SVC to LinearSVC (much faster on large data)
            'Linear SVM': LinearSVC(random_state=42, max_iter=3000),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        }

        self.param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'Multinomial Naive Bayes': {
                'alpha': [0.1, 1.0, 10.0]
            },
            'Linear SVM': {
                'C': [0.1, 1, 10]
            },
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 20]
            }
        }

        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None

    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return model

    def train_all_models(self, X_train, y_train):
        """Train all models"""
        print("=== Training All Models ===")

        for name, model in self.models.items():
            trained_model = self.train_model(model, X_train, y_train, name)
            self.trained_models[name] = trained_model

        print("\nAll models trained successfully!")
        return self.trained_models

    def hyperparameter_tuning(self, X_train, y_train, cv=3):
        """Perform hyperparameter tuning for all models (faster cv=3)"""
        print("\n=== Hyperparameter Tuning ===")

        tuned_models = {}

        for name, model in self.models.items():
            if name in self.param_grids:
                print(f"\nTuning {name}...")

                grid_search = GridSearchCV(
                    model,
                    self.param_grids[name],
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)

                tuned_models[name] = grid_search.best_estimator_
                print(f"Best parameters for {name}: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")
            else:
                tuned_models[name] = self.train_model(model, X_train, y_train, name)

        self.trained_models = tuned_models
        return tuned_models

    def cross_validate_models(self, X_train, y_train, cv=3):
        """Perform cross-validation for all trained models"""
        print("\n=== Cross-Validation Results ===")

        cv_results = {}

        for name, model in self.trained_models.items():
            print(f"\nCross-validating {name}...")

            scores = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                vals = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)
                scores[metric] = {'mean': vals.mean(), 'std': vals.std()}
                print(f"{metric.capitalize()}: {vals.mean():.4f} (+/- {vals.std() * 2:.4f})")

            cv_results[name] = scores

        return cv_results

    def select_best_model(self, cv_results, metric='f1'):
        """Select the best model based on cross-validation results"""
        print(f"\n=== Selecting Best Model (based on {metric}) ===")

        best_score = -1
        best_model_name = None

        for name, results in cv_results.items():
            score = results[metric]['mean']
            if score > best_score:
                best_score = score
                best_model_name = name

        self.best_model = self.trained_models[best_model_name]
        self.best_model_name = best_model_name

        print(f"Best model: {best_model_name}")
        print(f"Best {metric} score: {best_score:.4f}")

        return self.best_model, best_model_name

    def save_model(self, model, filepath, model_name="model"):
        """Save trained model"""
        joblib.dump(model, filepath)
        print(f"{model_name} saved to {filepath}")

    def load_model(self, filepath):
        """Load saved model"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

    def get_model_summary(self):
        """Get summary of all trained models"""
        summary = {
            'total_models': len(self.trained_models),
            'model_names': list(self.trained_models.keys()),
            'best_model': self.best_model_name
        }
        return summary


def quick_model_comparison(X_train, X_test, y_train, y_test):
    """Quick comparison of models without hyperparameter tuning"""
    print("=== Quick Model Comparison ===")

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'Multinomial NB': MultinomialNB(),
        'Linear SVM': LinearSVC(random_state=42, max_iter=3000),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time
        })

        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")

    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison Summary ===")
    print(results_df.round(4))

    return results_df


if __name__ == "__main__":
    from data_preprocessing import TextPreprocessor
    from feature_extraction import FeatureExtractor

    preprocessor = TextPreprocessor()
    df = preprocessor.load_and_preprocess_data("data/raw/IMDB Dataset.csv")

    extractor = FeatureExtractor(method='tfidf', max_features=3000)
    X_train, X_test, y_train, y_test = extractor.extract_features(
        df['processed_review'], df['sentiment']
    )

    results = quick_model_comparison(X_train, X_test, y_train, y_test)

    trainer = SentimentModelTrainer()
    trained_models = trainer.hyperparameter_tuning(X_train, y_train, cv=3)
    cv_results = trainer.cross_validate_models(X_train, y_train, cv=3)
    best_model, best_name = trainer.select_best_model(cv_results)
