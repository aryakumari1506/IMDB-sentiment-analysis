"""
Model Evaluation Module for Sentiment Analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        """Initialize model evaluator"""
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive evaluation of a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            model_name: Name for the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n=== Evaluating {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # AUC-ROC if probabilities available
        auc_roc = None
        if y_pred_proba is not None:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if auc_roc:
            print(f"AUC-ROC:   {auc_roc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.evaluation_results[model_name] = results
        return results
    
    def evaluate_multiple_models(self, models_dict, X_test, y_test):
        """
        Evaluate multiple models
        
        Args:
            models_dict: Dictionary of {model_name: trained_model}
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with all evaluation results
        """
        print("=== Evaluating Multiple Models ===")
        
        all_results = {}
        
        for model_name, model in models_dict.items():
            results = self.evaluate_model(model, X_test, y_test, model_name)
            all_results[model_name] = results
        
        return all_results
    
    def plot_confusion_matrix(self, cm, model_name, labels=['Negative', 'Positive']):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            labels: Class labels
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name):
        """
        Plot ROC curve
        
        Args:
            y_test: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
        """
        if y_pred_proba is None:
            print(f"Cannot plot ROC curve for {model_name}: No probabilities available")
            return
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, results_dict=None):
        """
        Compare multiple models and create comparison plots
        
        Args:
            results_dict: Dictionary of evaluation results
        """
        if results_dict is None:
            results_dict = self.evaluation_results
        
        if not results_dict:
            print("No evaluation results available")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results.get('auc_roc', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n=== Model Comparison Summary ===")
        print(comparison_df.round(4))
        
        # Plot comparison
        self.plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """
        Plot model comparison charts
        
        Args:
            comparison_df: DataFrame with model comparison results
        """
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            row = i // 2
            col = i % 2
            
            ax = axes[row, col]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=color, alpha=0.7)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_confusion_matrices(self, results_dict=None):
        """
        Plot confusion matrices for multiple models
        
        Args:
            results_dict: Dictionary of evaluation results
        """
        if results_dict is None:
            results_dict = self.evaluation_results
        
        if not results_dict:
            print("No evaluation results available")
            return
        
        n_models = len(results_dict)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            if i < len(axes):
                cm = results['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=['Negative', 'Positive'], 
                          yticklabels=['Negative', 'Positive'],
                          ax=axes[i])
                axes[i].set_title(f'{model_name}')
                axes[i].set_xlabel('Predicted Label')
                axes[i].set_ylabel('True Label')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_roc_curves(self, y_test, results_dict=None):
        """
        Plot ROC curves for multiple models on the same plot
        
        Args:
            y_test: True labels
            results_dict: Dictionary of evaluation results
        """
        if results_dict is None:
            results_dict = self.evaluation_results
        
        if not results_dict:
            print("No evaluation results available")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            y_pred_proba = results.get('y_pred_proba')
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = results.get('auc_roc', roc_auc_score(y_test, y_pred_proba))
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_detailed_report(self, results_dict=None, save_path=None):
        """
        Generate a detailed evaluation report
        
        Args:
            results_dict: Dictionary of evaluation results
            save_path: Path to save the report (optional)
        """
        if results_dict is None:
            results_dict = self.evaluation_results
        
        if not results_dict:
            print("No evaluation results available")
            return
        
        report = "SENTIMENT ANALYSIS MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall summary
        comparison_data = []
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results.get('auc_roc', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        report += "SUMMARY OF ALL MODELS:\n"
        report += "-" * 25 + "\n"
        report += comparison_df.to_string(index=False) + "\n\n"
        
        # Best performing model
        best_f1_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.iloc[best_f1_idx]['Model']
        report += f"BEST PERFORMING MODEL (by F1-Score): {best_model}\n\n"
        
        # Detailed results for each model
        for model_name, results in results_dict.items():
            report += f"DETAILED RESULTS FOR {model_name.upper()}:\n"
            report += "-" * (len(model_name) + 20) + "\n"
            report += f"Accuracy:  {results['accuracy']:.4f}\n"
            report += f"Precision: {results['precision']:.4f}\n"
            report += f"Recall:    {results['recall']:.4f}\n"
            report += f"F1-Score:  {results['f1_score']:.4f}\n"
            if results.get('auc_roc'):
                report += f"AUC-ROC:   {results['auc_roc']:.4f}\n"
            
            report += "\nConfusion Matrix:\n"
            cm = results['confusion_matrix']
            report += f"                Predicted\n"
            report += f"              Neg    Pos\n"
            report += f"Actual  Neg   {cm[0,0]:3d}    {cm[0,1]:3d}\n"
            report += f"        Pos   {cm[1,0]:3d}    {cm[1,1]:3d}\n\n"
            
            report += "Classification Report:\n"
            report += results['classification_report'] + "\n"
            report += "-" * 50 + "\n\n"
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report

def calculate_additional_metrics(y_test, y_pred, y_pred_proba=None):
    """
    Calculate additional evaluation metrics
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary with additional metrics
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    
    additional_metrics = {
        'specificity': specificity,
        'sensitivity': sensitivity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'balanced_accuracy': balanced_accuracy,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return additional_metrics

if __name__ == "__main__":
    # Test model evaluation
    from data_preprocessing import TextPreprocessor
    from feature_extraction import FeatureExtractor
    from model_training import SentimentModelTrainer
    
    # Load and preprocess data
    preprocessor = TextPreprocessor()
    df = preprocessor.load_and_preprocess_data()
    
    # Extract features
    extractor = FeatureExtractor(method='tfidf', max_features=3000)
    X_train, X_test, y_train, y_test = extractor.extract_features(
        df['processed_review'], df['sentiment']
    )
    
    # Train models
    trainer = SentimentModelTrainer()
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_multiple_models(trained_models, X_test, y_test)
    
    # Generate comparison
    comparison_df = evaluator.compare_models()
    
    # Generate detailed report
    evaluator.generate_detailed_report()