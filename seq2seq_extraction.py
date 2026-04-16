import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import logging
import warnings
import gc
import time

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class PumpDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=16384):
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.targets = targets.tolist() if isinstance(targets, pd.Series) else targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = str(self.targets[idx])

        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer.encode_plus(
            target,
            max_length=32,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def predict(model, tokenizer, text, device, max_length=32):
    inputs = tokenizer.encode_plus(
        text,
        max_length=16384,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

def evaluate_model(model, tokenizer, test_df, device):
    print("\nGenerating predictions for test set...")
    start_time = time.time()
    
    predictions = []
    for text in tqdm(test_df['window_messages'], desc="Generating predictions"):
        pred = predict(model, tokenizer, text, device)
        predictions.append(pred)
    
    end_time = time.time()
    test_time = end_time - start_time
    
    results_df = pd.DataFrame({
        'window_messages': test_df['window_messages'],
        'actual_coin': test_df['coin_cleaned'],
        'actual_exchange': test_df['exchange_cleaned'],
        'predicted': predictions
    })
    
    results_df['predicted_coin'] = results_df['predicted'].apply(
        lambda x: x.split('|')[0] if '|' in x else x
    )
    results_df['predicted_exchange'] = results_df['predicted'].apply(
        lambda x: x.split('|')[1] if '|' in x else ''
    )
    
    coin_accuracy = (results_df['actual_coin'] == results_df['predicted_coin']).mean()
    exchange_accuracy = (results_df['actual_exchange'] == results_df['predicted_exchange']).mean()
    exact_match = (results_df['predicted'] == results_df['actual_coin'] + '|' + results_df['actual_exchange']).mean()
    
    print("\nTest Results:")
    print(f"Coin accuracy: {coin_accuracy:.4f}")
    print(f"Exchange accuracy: {exchange_accuracy:.4f}")
    print(f"Exact match accuracy: {exact_match:.4f}")
    print(f"\nTest set evaluation time: {test_time:.2f} seconds")
    print(f"Average time per sample: {test_time/len(test_df):.4f} seconds")
    
    print("\nDiagnostic - Random Sample Results:")
    import random
    sample_size = min(10, len(results_df))
    random_indices = random.sample(range(len(results_df)), sample_size)
    
    for i, idx in enumerate(random_indices):
        result = results_df.iloc[idx]
        print(f"\nSample {i+1}:")
        print(f"  Raw Prediction: '{result['predicted']}'")
        print(f"  Predicted Coin: {result['predicted_coin']}")
        print(f"  Actual Coin:     {result['actual_coin']}")
        print(f"  Coin Match:      {result['predicted_coin'] == result['actual_coin']}")
        print(f"  Predicted Exchange: {result['predicted_exchange']}")
        print(f"  Actual Exchange:    {result['actual_exchange']}")
        print(f"  Exchange Match:     {result['predicted_exchange'] == result['actual_exchange']}")
    
    print(f"\nShowing {sample_size} random samples out of {len(results_df)} total predictions")
    
    results_df.to_csv('test_predictions.csv', index=False)
    print("\nResults saved to 'test_predictions.csv'")

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, patience=3, gradient_accumulation_steps=4):
    best_val_loss = float('inf')
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            total_loss += loss.item() * gradient_accumulation_steps

        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                with autocast():
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
                    val_loss += outputs.loss.item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved! (val_loss: {avg_val_loss:.4f})')
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_df = pd.read_csv('train_extraction.csv')
    val_df = pd.read_csv('val_extraction.csv')
    test_df = pd.read_csv('test_extraction.csv')
    
    train_texts = train_df['window_messages'].apply(eval).apply(lambda x: ' '.join(x))
    train_targets = train_df.apply(lambda x: f"{x['coin_cleaned']}|{x['exchange_cleaned']}", axis=1)
    
    val_texts = val_df['window_messages'].apply(eval).apply(lambda x: ' '.join(x))
    val_targets = val_df.apply(lambda x: f"{x['coin_cleaned']}|{x['exchange_cleaned']}", axis=1)
    
    test_texts = test_df['window_messages'].apply(eval).apply(lambda x: ' '.join(x))
    test_targets = test_df.apply(lambda x: f"{x['coin_cleaned']}|{x['exchange_cleaned']}", axis=1)

    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    print(f"Test set size: {len(test_texts)}")

    model_name = "allenai/led-base-16384"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()
    model.to(device)

    train_dataset = PumpDataset(train_texts, train_targets, tokenizer)
    val_dataset = PumpDataset(val_texts, val_targets, tokenizer)
    test_dataset = PumpDataset(test_texts, test_targets, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("\nTraining model...")
    train_model(model, train_loader, val_loader, optimizer, device)

    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_model.pth'))

    print("\nEvaluating on test set...")
    evaluate_model(model, tokenizer, test_df, device)

if __name__ == "__main__":
    main()
