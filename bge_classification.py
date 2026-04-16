import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score
from tqdm import tqdm
import time
import re
import argparse
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable tokenizer parallelism to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CryptoDataset(Dataset):
    """Dataset for crypto pump detection with chunking for long texts"""
    def __init__(self, texts, labels, tokenizer, max_length=512, chunk_long_texts=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_long_texts = chunk_long_texts
        
        if chunk_long_texts:
            self.chunk_info = self._prepare_chunk_info()
        else:
            self.chunk_info = None
    
    def _prepare_chunk_info(self):
        """Prepare chunk information without tokenizing"""
        chunk_info = []
        
        for i, text in enumerate(tqdm(self.texts, desc="Preparing chunk info", leave=False)):
            text = str(text)
            label = self.labels[i]
            
            # Check if text needs chunking (rough estimate)
            if len(text.split()) > self.max_length * 0.7:  # Rough estimate
                # Will be chunked - store text and label
                chunk_info.append({
                    'text': text,
                    'label': label,
                    'original_idx': i,
                    'needs_chunking': True
                })
            else:
                # Won't be chunked - store text and label
                chunk_info.append({
                    'text': text,
                    'label': label,
                    'original_idx': i,
                    'needs_chunking': False
                })
        
        return chunk_info
    
    def _split_text_into_chunks(self, text):
        """Split text into chunks of approximately max_length tokens"""
        # Tokenize the full text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Calculate chunk size (leave some room for special tokens)
        chunk_size = self.max_length - 2  # -2 for [CLS] and [SEP]
        
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            # Decode chunk back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
        
        return chunks
    
    def __len__(self):
        if self.chunk_long_texts:
            # Count total chunks
            total_chunks = 0
            for item in self.chunk_info:
                if item['needs_chunking']:
                    # Estimate number of chunks
                    text = item['text']
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    chunk_size = self.max_length - 2
                    num_chunks = max(1, (len(tokens) + chunk_size - 1) // chunk_size)
                    total_chunks += num_chunks
                else:
                    total_chunks += 1
            return total_chunks
        else:
            return len(self.texts)
    
    def __getitem__(self, idx):
        if self.chunk_long_texts:
            # This should not be called for chunked datasets - use ChunkedCryptoDataset instead
            raise NotImplementedError("Use ChunkedCryptoDataset for chunked evaluation")
        else:
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long),
                'original_idx': idx,
                'chunk_idx': 0,
                'total_chunks': 1
            }

class ChunkedCryptoDataset(CryptoDataset):
    """Dataset for evaluation that handles chunked predictions - OPTIMIZED VERSION"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        super().__init__(texts, labels, tokenizer, max_length, chunk_long_texts=True)
        
        # Pre-compute all chunks and build efficient indexing
        self.all_chunks = []
        self.chunk_to_original = {}
        self.chunk_groups = {}
        
        print(f"Pre-computing chunks for {len(self.chunk_info)} texts...")
        chunk_idx = 0
        for item in tqdm(self.chunk_info, desc="Pre-computing chunks", leave=False):
            orig_idx = item['original_idx']
            text = item['text']
            label = item['label']
            
            if orig_idx not in self.chunk_groups:
                self.chunk_groups[orig_idx] = []
            
            if item['needs_chunking']:
                # Pre-compute all chunks for this text
                chunks = self._split_text_into_chunks(text)
                for i, chunk in enumerate(chunks):
                    # Pre-tokenize the chunk
                    encoding = self.tokenizer(
                        chunk,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    self.all_chunks.append({
                        'input_ids': encoding['input_ids'].flatten(),
                        'attention_mask': encoding['attention_mask'].flatten(),
                        'labels': torch.tensor(label, dtype=torch.long),
                        'original_idx': orig_idx,
                        'chunk_idx': i,
                        'total_chunks': len(chunks)
                    })
                    
                    self.chunk_groups[orig_idx].append({
                        'chunk_idx': i,
                        'total_chunks': len(chunks)
                    })
                    self.chunk_to_original[chunk_idx] = orig_idx
                    chunk_idx += 1
            else:
                # Single chunk - pre-tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                self.all_chunks.append({
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long),
                    'original_idx': orig_idx,
                    'chunk_idx': 0,
                    'total_chunks': 1
                })
                
                self.chunk_groups[orig_idx].append({
                    'chunk_idx': 0,
                    'total_chunks': 1
                })
                self.chunk_to_original[chunk_idx] = orig_idx
                chunk_idx += 1
    
    def __len__(self):
        return len(self.all_chunks)
    
    def __getitem__(self, idx):
        # Direct indexing - O(1) access
        return self.all_chunks[idx]

def clean_text(text):
    """Clean text while preserving crypto symbols"""
    # Preserve crypto symbols and dollar signs, clean other punctuation
    text = re.sub(r"[^a-zA-Z0-9$]", " ", text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_texts(texts):
    """Clean and preprocess all texts"""
    processed_texts = []
    
    for text in tqdm(texts, desc="Preprocessing texts", leave=False):
        # Clean text
        processed_text = clean_text(text)
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

def train_epoch(model, dataloader, optimizer, scheduler, device, args):
    """Train for one epoch with optimizations"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    # Setup mixed precision if enabled
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    optimizer.zero_grad()
    accumulation_steps = args.gradient_accumulation_steps
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision training
        if args.mixed_precision and device.type == 'cuda':
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            if args.mixed_precision and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'Loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, chunked_dataset=None):
    """Evaluate the model with support for chunked predictions"""
    model.eval()
    total_loss = 0
    chunk_predictions = []
    chunk_true_labels = []
    chunk_original_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            chunk_predictions.extend(preds.cpu().numpy())
            chunk_true_labels.extend(labels.cpu().numpy())
            
            # Store original indices for chunked datasets
            if 'original_idx' in batch:
                chunk_original_indices.extend(batch['original_idx'].cpu().numpy())
    
    # If we have a chunked dataset, aggregate predictions using voting
    if chunked_dataset and chunked_dataset.chunk_groups:
        return aggregate_chunked_predictions(
            chunk_predictions, 
            chunk_true_labels, 
            chunk_original_indices, 
            chunked_dataset
        )
    else:
        return chunk_predictions, chunk_true_labels, total_loss / len(dataloader)

def aggregate_chunked_predictions(chunk_predictions, chunk_true_labels, chunk_original_indices, chunked_dataset):
    """Aggregate chunk predictions using voting mechanism"""
    # Group predictions by original text index
    grouped_predictions = {}
    grouped_labels = {}
    
    for i, (pred, label, orig_idx) in enumerate(zip(chunk_predictions, chunk_true_labels, chunk_original_indices)):
        if orig_idx not in grouped_predictions:
            grouped_predictions[orig_idx] = []
            grouped_labels[orig_idx] = label  # All chunks have the same label
        
        grouped_predictions[orig_idx].append(pred)
    
    # Apply voting mechanism: if any chunk predicts 1, final prediction is 1
    final_predictions = []
    final_labels = []
    
    for orig_idx in sorted(grouped_predictions.keys()):
        chunk_preds = grouped_predictions[orig_idx]
        label = grouped_labels[orig_idx]
        
        # Voting: if any chunk predicts 1, final prediction is 1
        final_pred = 1 if any(pred == 1 for pred in chunk_preds) else 0
        
        final_predictions.append(final_pred)
        final_labels.append(label)
    
    return final_predictions, final_labels, 0  # Loss not meaningful for aggregated predictions

def print_chunking_statistics(dataset, dataset_name):
    """Print statistics about text chunking"""
    if hasattr(dataset, 'chunk_groups'):
        total_original = len(dataset.chunk_groups)
        total_chunks = len(dataset)
        
        # Count texts that were chunked
        chunked_texts = sum(1 for chunks in dataset.chunk_groups.values() if len(chunks) > 1)
        single_chunk_texts = total_original - chunked_texts
        
        print(f"\nðŸ“Š {dataset_name} Chunking Statistics:")
        print(f"   Original texts: {total_original}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Texts that fit in single chunk: {single_chunk_texts}")
        print(f"   Texts that were chunked: {chunked_texts}")
        print(f"   Average chunks per text: {total_chunks/total_original:.2f}")
        
        if chunked_texts > 0:
            chunk_counts = [len(chunks) for chunks in dataset.chunk_groups.values() if len(chunks) > 1]
            print(f"   Max chunks per text: {max(chunk_counts)}")
            print(f"   Min chunks per text: {min(chunk_counts)}")
    else:
        print(f"\nðŸ“Š {dataset_name}: No chunking (regular dataset)")

def collate_fn(batch):
    """Custom collate function to handle additional fields"""
    # Separate the additional fields
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Handle additional fields if they exist
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    # Add additional fields if they exist in the batch
    if 'original_idx' in batch[0]:
        original_idx = torch.tensor([item['original_idx'] for item in batch])
        result['original_idx'] = original_idx
    
    if 'chunk_idx' in batch[0]:
        chunk_idx = torch.tensor([item['chunk_idx'] for item in batch])
        result['chunk_idx'] = chunk_idx
    
    if 'total_chunks' in batch[0]:
        total_chunks = torch.tensor([item['total_chunks'] for item in batch])
        result['total_chunks'] = total_chunks
    
    return result

def print_metrics(y_true, y_pred, split_name):
    """Print detailed metrics"""
    print(f"\n{split_name} Results:")
    print(classification_report(y_true, y_pred, target_names=["NO","YES"]))
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted:")
    print("         NO  YES")
    print(f"Actual NO  {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"     YES {cm[1,0]:4d} {cm[1,1]:4d}")
    
    return {
        'balanced_accuracy': balanced_acc,
        'accuracy': accuracy,
        'f1_score': f1
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BGE classification for pump detection')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-m3',
                       help='Model to use (default: BAAI/bge-m3)')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate (default: 3e-5)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (faster)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--use_chunking', action='store_true', default=False,
                       help='Use chunking for long texts (default: False with BGE-M3)')
    parser.add_argument('--no_chunking', action='store_true',
                       help='Disable chunking for long texts')
    args = parser.parse_args()
    
    # Handle chunking logic - default to False for BGE-M3 since it supports long sequences
    use_chunking = args.use_chunking and not args.no_chunking
    
    print("Starting BGE Classification for Pump Detection")
    print("="*60)
    if use_chunking:
        print("ðŸ“ Using chunking strategy for long texts")
        print("ðŸ“ Voting mechanism: if any chunk predicts 1, final prediction is 1")
    else:
        print("ðŸ“ Using BGE-M3 with 8192 token support (no chunking needed)")
    print("="*60)
    
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    print("\nðŸ“Š Loading train, validation, and test data...")
    X_train, y_train = load_and_process_data("train_windows.csv", "train")
    X_val, y_val = load_and_process_data("val_windows.csv", "validation")
    X_test, y_test = load_and_process_data("test_windows.csv", "test")
    
    print(f"\nðŸ“ˆ Data sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nðŸ”§ Step 3: Preprocessing texts...")
    # Preprocess all datasets
    print("Processing training data...")
    X_train_processed = preprocess_texts(X_train)
    print("Processing validation data...")
    X_val_processed = preprocess_texts(X_val)
    print("Processing test data...")
    X_test_processed = preprocess_texts(X_test)
    
    print("\nðŸ”¤ Step 4: Loading BGE tokenizer and model...")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2,
        problem_type="single_label_classification"
    )
    model.to(device)
    
    # Check model's maximum sequence length
    model_max_length = tokenizer.model_max_length
    if args.max_length > model_max_length:
        print(f"âš ï¸  Warning: max_length ({args.max_length}) exceeds model's maximum ({model_max_length})")
        print(f"   Reducing max_length to {model_max_length}")
        args.max_length = model_max_length
    
    print(f"Model maximum sequence length: {model_max_length}")
    print(f"Using max_length: {args.max_length}")
    print(f"Model loaded on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nðŸ“¦ Step 5: Creating datasets and dataloaders...")
    # Create datasets - use regular dataset for training, chunked for evaluation
    train_dataset = CryptoDataset(X_train_processed, y_train, tokenizer, args.max_length, chunk_long_texts=False)
    if use_chunking:
        val_dataset = ChunkedCryptoDataset(X_val_processed, y_val, tokenizer, args.max_length)
        test_dataset = ChunkedCryptoDataset(X_test_processed, y_test, tokenizer, args.max_length)
    else:
        val_dataset = CryptoDataset(X_val_processed, y_val, tokenizer, args.max_length, chunk_long_texts=False)
        test_dataset = CryptoDataset(X_test_processed, y_test, tokenizer, args.max_length, chunk_long_texts=False)
    
    # Create dataloaders with optimizations
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    # Print chunking statistics
    print_chunking_statistics(train_dataset, "Training")
    if use_chunking:
        print_chunking_statistics(val_dataset, "Validation")
        print_chunking_statistics(test_dataset, "Test")
    else:
        print("ðŸ“Š Validation/Test: No chunking (regular datasets)")
    
    print("\nðŸŽ¯ Step 6: Setting up training...")
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    print("\nðŸ‹ï¸ Step 7: Training BGE model...")
    best_val_f1 = 0
    best_model = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, args)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        val_preds, val_true, val_loss = evaluate_model(model, val_dataloader, device, val_dataset if use_chunking else None)
        val_metrics = print_metrics(val_true, val_preds, "Validation")
        print(f"Validation loss: {val_loss:.4f}")
        
        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model = model.state_dict().copy()
            print(f"New best model! F1: {best_val_f1:.3f}")
    
    print("\nðŸ† Step 8: Evaluating best model on test set...")
    # Load best model
    model.load_state_dict(best_model)
    
    # Evaluate on test set with explicit inference timing
    test_eval_start = time.time()
    test_preds, test_true, test_loss = evaluate_model(model, test_dataloader, device, test_dataset if use_chunking else None)
    test_eval_time = time.time() - test_eval_start
    test_metrics = print_metrics(test_true, test_preds, "Test")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test inference time: {test_eval_time:.2f} seconds")
    print(f"Average time per sample (Test): {test_eval_time/len(test_true):.4f} seconds")
    
    print("\n" + "="*60)
    print("ðŸ† FINAL RESULTS")
    print("="*60)
    
    print(f"\nðŸ† Model: BGE ({args.model_name})")
    print(f"Best Validation F1: {best_val_f1:.3f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.3f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.3f}")
    
    # Save predictions for McNemar's test
    print("\nSaving predictions for McNemar's test...")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'true_label': test_true,
        'bge_prediction': test_preds,
        'bge_probability': [0.5] * len(test_preds)  # Placeholder probabilities
    })
    
    predictions_df.to_csv('bge_predictions.csv', index=False)
    print("BGE predictions saved to 'bge_predictions.csv'")
    
    print("\nâœ… Done!")

if __name__ == "__main__":
    main() 