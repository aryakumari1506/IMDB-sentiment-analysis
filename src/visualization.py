import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SentimentVisualizer:
    def __init__(self, figsize=(12, 8)):
        """Initialize visualizer with default figure size"""
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_sentiment_distribution(self, labels, title="Sentiment Distribution"):
        """
        Plot sentiment distribution
        
        Args:
            labels: Array of sentiment labels
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        
        # Count sentiments
        sentiment_counts = pd.Series(labels).value_counts().sort_index()
        labels_text = ['Negative', 'Positive']
        colors = ['lightcoral', 'lightblue']
        
        # Create bar plot
        bars = plt.bar(labels_text, sentiment_counts.values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, sentiment_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{count}\n({count/len(labels)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_text_length_distribution(self, texts, labels, title="Text Length Distribution"):
        """
        Plot distribution of text lengths by sentiment
        
        Args:
            texts: Array of text data
            labels: Array of sentiment labels
            title: Plot title
        """
        # Calculate text lengths
        text_lengths = [len(text.split()) for text in texts]
        df = pd.DataFrame({'length': text_lengths, 'sentiment': labels})
        
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram by sentiment
        for sentiment in [0, 1]:
            sentiment_lengths = df[df['sentiment'] == sentiment]['length']
            label = 'Negative' if sentiment == 0 else 'Positive'
            ax1.hist(sentiment_lengths, bins=30, alpha=0.7, label=label)
        
        ax1.set_xlabel('Text Length (words)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Text Length Distribution by Sentiment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 1: 'Positive'})
        sns.boxplot(data=df, x='sentiment_label', y='length', ax=ax2)
        ax2.set_title('Text Length Box Plot by Sentiment')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("Text Length Statistics:")
        print(df.groupby('sentiment')['length'].describe())
    
    def create_word_clouds(self, texts, labels, max_words=100):
        """
        Create word clouds for positive and negative sentiments
        
        Args:
            texts: Array of text data
            labels: Array of sentiment labels
            max_words: Maximum number of words in word cloud
        """
        # Separate texts by sentiment
        positive_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
        negative_texts = [texts[i] for i in range(len(texts)) if labels[i] == 0]
        
        # Combine texts
        positive_text = ' '.join(positive_texts)
        negative_text = ' '.join(negative_texts)
        
        # Create word clouds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Positive word cloud
        if positive_text:
            wordcloud_pos = WordCloud(width=800, height=400, 
                                     max_words=max_words, 
                                     background_color='white',
                                     colormap='Blues').generate(positive_text)
            ax1.imshow(wordcloud_pos, interpolation='bilinear')
            ax1.set_title('Positive Sentiment Word Cloud', fontsize=16, fontweight='bold')
            ax1.axis('off')
        
        # Negative word cloud
        if negative_text:
            wordcloud_neg = WordCloud(width=800, height=400, 
                                     max_words=max_words, 
                                     background_color='white',
                                     colormap='Reds').generate(negative_text)
            ax2.imshow(wordcloud_neg, interpolation='bilinear')
            ax2.set_title('Negative Sentiment Word Cloud', fontsize=16, fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_words(self, texts, labels, top_n=20):
        """
        Plot top words for each sentiment
        
        Args:
            texts: Array of text data
            labels: Array of sentiment labels
            top_n: Number of top words to display
        """
        # Separate texts by sentiment
        positive_texts = ' '.join([texts[i] for i in range(len(texts)) if labels[i] == 1])
        negative_texts = ' '.join([texts[i] for i in range(len(texts)) if labels[i] == 0])
        
        # Count words
        positive_words = Counter(positive_texts.split())
        negative_words = Counter(negative_texts.split())
        
        # Get top words
        top_positive = positive_words.most_common(top_n)
        top_negative = negative_words.most_common(top_n)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot positive words
        if top_positive:
            words, counts = zip(*top_positive)
            ax1.barh(range(len(words)), counts, color='lightblue', alpha=0.7)
            ax1.set_yticks(range(len(words)))
            ax1.set_yticklabels(words)
            ax1.set_xlabel('Frequency')
            ax1.set_title(f'Top {top_n} Words in Positive Reviews', fontweight='bold')
            ax1.invert_yaxis()
            
            # Add count labels
            for i, count in enumerate(counts):
                ax1.text(count + max(counts)*0.01, i, str(count), 
                        va='center', fontweight='bold')
        
        # Plot negative words
        if top_negative:
            words, counts = zip(*top_negative)
            ax2.barh(range(len(words)), counts, color='lightcoral', alpha=0.7)
            ax2.set_yticks(range(len(words)))
            ax2.set_yticklabels(words)
            ax2.set_xlabel('Frequency')
            ax2.set_title(f'Top {top_n} Words in Negative Reviews', fontweight='bold')
            ax2.invert_yaxis()
            
            # Add count labels
            for i, count in enumerate(counts):
                ax2.text(count + max(counts)*0.01, i, str(count), 
                        va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names, coefficients, model_name, top_n=20):
        """
        Plot feature importance from model coefficients
        
        Args:
            feature_names: Array of feature names
            coefficients: Model coefficients
            model_name: Name of the model
            top_n: Number of top features to display
        """
        # Get top positive and negative features
        top_pos_indices = np.argsort(coefficients)[-top_n:]
        top_neg_indices = np.argsort(coefficients)[:top_n]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Positive features
        pos_features = [feature_names[i] for i in reversed(top_pos_indices)]
        pos_coeffs = [coefficients[i] for i in reversed(top_pos_indices)]
        
        ax1.barh(range(len(pos_features)), pos_coeffs, color='lightblue', alpha=0.7)
        ax1.set_yticks(range(len(pos_features)))
        ax1.set_yticklabels(pos_features)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title(f'Top {top_n} Positive Features - {model_name}', fontweight='bold')
        ax1.invert_yaxis()
        
        # Add coefficient labels
        for i, coeff in enumerate(pos_coeffs):
            ax1.text(coeff + max(pos_coeffs)*0.01, i, f'{coeff:.3f}', 
                    va='center', fontweight='bold')
        
        # Negative features
        neg_features = [feature_names[i] for i in top_neg_indices]
        neg_coeffs = [coefficients[i] for i in top_neg_indices]
        
        ax2.barh(range(len(neg_features)), neg_coeffs, color='lightcoral', alpha=0.7)
        ax2.set_yticks(range(len(neg_features)))
        ax2.set_yticklabels(neg_features)
        ax2.set_xlabel('Coefficient Value')
        ax2.set_title(f'Top {top_n} Negative Features - {model_name}', fontweight='bold')
        ax2.invert_yaxis()
        
        # Add coefficient labels
        for i, coeff in enumerate(neg_coeffs):
            ax2.text(coeff + min(neg_coeffs)*0.01, i, f'{coeff:.3f}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_distribution(self, y_pred_proba, y_test, model_name):
        """
        Plot distribution of prediction probabilities
        
        Args:
            y_pred_proba: Prediction probabilities
            y_test: True labels
            model_name: Name of the model
        """
        if y_pred_proba is None:
            print(f"No prediction probabilities available for {model_name}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of prediction probabilities
        ax1.hist(y_pred_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Prediction Probabilities - {model_name}')
        ax1.grid(True, alpha=0.3)
        
        # Probability distribution by true label
        correct_probs = y_pred_proba[y_test == 1]
        incorrect_probs = y_pred_proba[y_test == 0]
        
        ax2.hist(correct_probs, bins=30, alpha=0.7, label='Positive (True)', color='lightgreen')
        ax2.hist(incorrect_probs, bins=30, alpha=0.7, label='Negative (True)', color='lightcoral')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Prediction Probabilities by True Label - {model_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_comprehensive_visualization(self, texts, labels, vectorizer=None, 
                                         model=None, model_name="Model"):
        """
        Create a comprehensive visualization dashboard
        
        Args:
            texts: Array of text data
            labels: Array of sentiment labels
            vectorizer: Trained vectorizer (optional)
            model: Trained model (optional)
            model_name: Name of the model
        """
        print("Creating comprehensive visualization dashboard...")
        
        # 1. Sentiment distribution
        self.plot_sentiment_distribution(labels, "Dataset Sentiment Distribution")
        
        # 2. Text length distribution
        self.plot_text_length_distribution(texts, labels, "Text Length Analysis")
        
        # 3. Word clouds
        print("Generating word clouds...")
        self.create_word_clouds(texts, labels)
        
        # 4. Top words
        print("Analyzing top words...")
        self.plot_top_words(texts, labels)
        
        # 5. Feature importance (if model and vectorizer available)
        if model is not None and vectorizer is not None and hasattr(model, 'coef_'):
            print(f"Plotting feature importance for {model_name}...")
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            self.plot_feature_importance(feature_names, coefficients, model_name)
        
        print("Visualization dashboard complete!")

def create_model_comparison_dashboard(results_dict, y_test):
    """
    Create a comprehensive model comparison dashboard
    
    Args:
        results_dict: Dictionary of evaluation results
        y_test: True test labels
    """
    print("Creating model comparison dashboard...")
    
    visualizer = SentimentVisualizer()
    
    # 1. Performance comparison
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plot performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison Dashboard', fontsize=16, fontweight='bold')
    
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
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Confusion matrices
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
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {model_name}')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
