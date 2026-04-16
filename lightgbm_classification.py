import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score
from tqdm import tqdm
import time
import re
import lightgbm as lgb
import argparse
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

def clean_and_tokenize(text):
    """Clean text and preserve important crypto symbols"""
    # Preserve crypto symbols and dollar signs, clean other punctuation
    text = re.sub(r"[^a-zA-Z0-9$]", " ", text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def preprocess_texts(texts):
    """Clean and preprocess all texts"""
    processed_texts = []
    
    for text in tqdm(texts, desc="Preprocessing texts", leave=False):
        # Clean and tokenize
        tokens = clean_and_tokenize(text)
        # Join back to text
        processed_text = ' '.join(tokens)
        processed_texts.append(processed_text)
    
    return processed_texts

def load_and_process_data(file_path, split_name):
    """Load and process data for a specific split"""
    print(f"Step 1: Loading {split_name} data...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} {split_name} windows")
    
    print(f"Step 2: Processing text data for {split_name}...")
    # Convert window_messages from string representation to actual list
    df["window_list"] = df["window_messages"].apply(lambda s: eval(s))
    texts = ["\n".join(w) for w in tqdm(df["window_list"], desc=f"Processing {split_name} texts", leave=False)]
    labels = df["nearby_pump"].tolist()
    print(f"Processed {len(texts)} {split_name} text samples")
    
    return texts, labels

def evaluate_model(model, X_val, y_val, X_test, y_test, model_name):
    """Evaluate a model and return detailed results"""
    print(f"\n{model_name} Results:")
    
    # Validation set
    print("Evaluating on validation set...")
    start_time = time.time()
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_time = time.time() - start_time
    
    print("\nValidation Set Results:")
    print(classification_report(y_val, y_val_pred, target_names=["NO","YES"]))
    balanced_acc_val = balanced_accuracy_score(y_val, y_val_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred)
    print(f"Balanced Accuracy (Validation): {balanced_acc_val:.3f}")
    print(f"Accuracy (Validation): {accuracy_val:.3f}")
    print(f"F1-Score (Validation): {f1_val:.3f}")
    print(f"Validation set evaluation time: {val_time:.2f} seconds")
    print(f"Average time per row (Validation): {val_time/len(y_val):.4f} seconds")
    
    # Test set
    print("Evaluating on test set...")
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_time = time.time() - start_time
    
    print("\nTest Set Results:")
    print(classification_report(y_test, y_test_pred, target_names=["NO","YES"]))
    balanced_acc_test = balanced_accuracy_score(y_test, y_test_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    print(f"Balanced Accuracy (Test): {balanced_acc_test:.3f}")
    print(f"Accuracy (Test): {accuracy_test:.3f}")
    print(f"F1-Score (Test): {f1_test:.3f}")
    print(f"Test set evaluation time: {test_time:.2f} seconds")
    print(f"Average time per row (Test): {test_time/len(y_test):.4f} seconds")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print("Predicted:")
    print("         NO  YES")
    print(f"Actual NO  {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"     YES {cm[1,0]:4d} {cm[1,1]:4d}")
    
    return {
        'model_name': model_name,
        'balanced_acc_val': balanced_acc_val,
        'balanced_acc_test': balanced_acc_test,
        'accuracy_val': accuracy_val,
        'accuracy_test': accuracy_test,
        'f1_val': f1_val,
        'f1_test': f1_test,
        'val_time': val_time,
        'test_time': test_time,
        'avg_time_per_row_val': val_time/len(y_val),
        'avg_time_per_row_test': test_time/len(y_test)
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple LightGBM classification for pump detection')
    parser.add_argument('--features', type=int, default=15000, 
                       help='Number of TF-IDF features to use (default: 15000)')
    args = parser.parse_args()
    
    print("Starting Simple LightGBM Classification")
    print("="*60)
    
    print("\nLoading train, validation, and test data...")
    X_train, y_train = load_and_process_data("train_windows.csv", "train")
    X_val, y_val = load_and_process_data("val_windows.csv", "validation")
    X_test, y_test = load_and_process_data("test_windows.csv", "test")
    
    print(f"\nData sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nStep 3: Preprocessing texts (cleaning tokens)...")
    # Preprocess all datasets with progress bars
    print("Processing training data...")
    X_train_processed = preprocess_texts(X_train)
    print("Processing validation data...")
    X_val_processed = preprocess_texts(X_val)
    print("Processing test data...")
    X_test_processed = preprocess_texts(X_test)
    
    print("\nStep 4: Creating TF-IDF features...")
    # Create TF-IDF features
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=args.features)
    print("Fitting TF-IDF vectorizer on training data...")
    X_train_tfidf = tfidf.fit_transform(X_train_processed)
    print("Transforming validation data...")
    X_val_tfidf = tfidf.transform(X_val_processed)
    print("Transforming test data...")
    X_test_tfidf = tfidf.transform(X_test_processed)
    print(f"Created {X_train_tfidf.shape[1]} TF-IDF features")
    
    print("\nTraining LightGBM with default parameters...")
    
    # Create LightGBM model with default parameters
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=0,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=1
    )
    
    print("Model parameters:")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  learning_rate: {model.learning_rate}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  num_leaves: {model.num_leaves}")
    print(f"  class_weight: {model.class_weight}")
    
    # Train the model
    print("\nTraining LightGBM model...")
    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    results = evaluate_model(
        model, 
        X_val_tfidf, y_val, 
        X_test_tfidf, y_test, 
        "LightGBM (Default Parameters)"
    )
    
    print("\n" + "="*60)
    print(" FINAL RESULTS")
    print("="*60)
    
    print(f"\n Model: {results['model_name']}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Balanced Accuracy (Test): {results['balanced_acc_test']:.3f}")
    print(f"Accuracy (Test): {results['accuracy_test']:.3f}")
    print(f"F1-Score (Test): {results['f1_test']:.3f}")
    print(f"Validation time: {results['val_time']:.2f} seconds")
    print(f"Test time: {results['test_time']:.2f} seconds")
    print(f"Average time per row (Validation): {results['avg_time_per_row_val']:.4f} seconds")
    print(f"Average time per row (Test): {results['avg_time_per_row_test']:.4f} seconds")
    
    # Save predictions for McNemar's test
    print("\nSaving predictions for McNemar's test...")
    test_predictions = model.predict(X_test_tfidf)
    test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]
    
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'lightgbm_prediction': test_predictions,
        'lightgbm_probability': test_probabilities
    })
    
    predictions_df.to_csv('lightgbm_predictions.csv', index=False)
    print("LightGBM predictions saved to 'lightgbm_predictions.csv'")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 