import pandas as pd
import numpy as np
import re
import nltk
import logging
import joblib
import argparse

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Scarica risorse NLTK la prima volta
nltk.download('stopwords')
nltk.download('wordnet')

class PhishingDetectorAdvanced:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.pipeline = None
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\S+@\S+', 'emailaddr', text)  # maschera email
        text = re.sub(r'http\S+', 'httpaddr', text)   # maschera URL
        text = re.sub(r'\d+', 'number', text)         # maschera numeri
        text = re.sub(r'\W', ' ', text)                # rimuove punteggiatura
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens if tok not in self.stop_words]
        return ' '.join(tokens)
    
    def preprocess_series(self, texts):
        return texts.apply(self.preprocess_text)
    
    def train(self, X, y):
        logging.info("Preprocessing text data...")
        X_clean = self.preprocess_series(X)
        
        # Pipeline con TF-IDF, PCA e RandomForest
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=7000, ngram_range=(1,2))),
            ('pca', PCA(n_components=100)),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        
        # Parametri per GridSearch
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5]
        }
        
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
        
        logging.info("Starting GridSearchCV training...")
        grid.fit(X_clean, y)
        
        logging.info(f"Best parameters: {grid.best_params_}")
        logging.info(f"Best cross-validation F1 score: {grid.best_score_:.4f}")
        
        self.pipeline = grid.best_estimator_
        return self.pipeline
    
    def evaluate(self, X, y):
        if self.pipeline is None:
            raise Exception("Model not trained or loaded!")
        X_clean = self.preprocess_series(X)
        y_pred = self.pipeline.predict(X_clean)
        y_prob = self.pipeline.predict_proba(X_clean)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        report = classification_report(y, y_pred, digits=4)
        cm = confusion_matrix(y, y_pred)
        
        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"ROC AUC: {roc_auc:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Classification Report:\n{report}")
        
        return acc, roc_auc, cm, report
    
    def predict(self, X):
        if self.pipeline is None:
            raise Exception("Model not trained or loaded!")
        X_clean = self.preprocess_series(X)
        preds = self.pipeline.predict(X_clean)
        probs = self.pipeline.predict_proba(X_clean)[:, 1]
        return preds, probs
    
    def save(self, filepath):
        if self.pipeline is None:
            raise Exception("No model to save!")
        joblib.dump(self.pipeline, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        self.pipeline = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV deve contenere colonne 'text' e 'label'")
    return df

def main():
    parser = argparse.ArgumentParser(description="Advanced Automated Phishing Email Detector")
    parser.add_argument('action', choices=['train', 'evaluate', 'predict'], help="Azione da eseguire")
    parser.add_argument('--data', help="CSV file con email e label per train/evaluate")
    parser.add_argument('--input', help="CSV file con email (colonna 'text') per predict")
    parser.add_argument('--model', default='phishing_advanced_model.pkl', help="File modello")
    parser.add_argument('--output', help="File CSV per salvare predizioni (solo predict)")
    args = parser.parse_args()
    
    detector = PhishingDetectorAdvanced()
    
    if args.action == 'train':
        if not args.data:
            logging.error("Specifica il file CSV con --data per il training")
            return
        df = load_dataset(args.data)
        detector.train(df['text'], df['label'])
        detector.save(args.model)
    
    elif args.action == 'evaluate':
        if not args.data:
            logging.error("Specifica il file CSV con --data per la valutazione")
            return
        detector.load(args.model)
        df = load_dataset(args.data)
        detector.evaluate(df['text'], df['label'])
    
    elif args.action == 'predict':
        if not args.input:
            logging.error("Specifica il file CSV con --input per predire")
            return
        detector.load(args.model)
        df = pd.read_csv(args.input)
        if 'text' not in df.columns:
            logging.error("CSV di input deve contenere la colonna 'text'")
            return
        preds, probs = detector.predict(df['text'])
        df['prediction'] = preds
        df['probability'] = probs
        if args.output:
            df.to_csv(args.output, index=False)
            logging.info(f"Predizioni salvate in {args.output}")
        else:
            print(df[['text', 'prediction', 'probability']])

if __name__ == "__main__":
    main()
