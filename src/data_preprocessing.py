import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        # Ensure NLTK resources are available
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize
        """
        text = text.lower()
        text = ''.join([c for c in text if c.isalnum() or c.isspace()])
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    def load_and_preprocess_data(self, data_path):
        """
        Load dataset, clean reviews, and standardize column names
        Returns a DataFrame with columns: ['processed_review', 'sentiment']
        """
        # Load dataset
        df = pd.read_csv(data_path)

        # Drop duplicates (based on review text if available)
        if 'review' in df.columns:
            df = df.drop_duplicates(subset=['review'])

        # Clean reviews and store in standardized column
        if 'review' not in df.columns:
            raise KeyError("Dataset must contain a 'review' column with text data.")
        df['processed_review'] = df['review'].apply(self.clean_text)

        # Standardize sentiment/label column
        if 'sentiment' not in df.columns:
            for alt in ['label', 'target', 'class']:
                if alt in df.columns:
                    df.rename(columns={alt: 'sentiment'}, inplace=True)
                    break
            else:
                raise KeyError("Dataset must contain a 'sentiment' (or label/target/class) column.")

        # 🔹 Convert sentiment to numeric (negative=0, positive=1)
        df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

        # Save processed dataset
        df.to_csv('data/processed/processed_reviews.csv', index=False)
        return df


# Usage example
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    df = preprocessor.load_and_preprocess_data('data/raw/IMDB Dataset.csv')
    print(f"Processed {len(df)} reviews.")
    print(df.head())
