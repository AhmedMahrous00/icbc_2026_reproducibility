import json
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import time
import argparse
import os

print("Step 1: Loading coin data...")
with open("cryptocurrencies.json", encoding='utf-8') as f:
    coin_map = json.load(f)
ticker_set = {ticker.upper() for ticker in coin_map.keys()}
print(f"Loaded {len(ticker_set)} unique coin tickers")

# 2. Simple tokenizer: split on non‐alphanumeric, uppercase tokens
def tokenize(text):
    tokens = re.split(r"[^A-Za-z0-9]+", text)
    return [t.upper() for t in tokens if t]

# 3. Extraction function: find the first token that matches a known ticker
def extract_ticker_from_window(window_list, ticker_set):
    joined = "\n".join(window_list)
    for tok in tokenize(joined):
        if tok in ticker_set:
            return tok
    return None

print("\nStep 2: Loading exchange list...")
known_exchanges = [
    "Binance",
    "Coinbase Pro",
    "Kraken",
    "Bitstamp",
    "Gemini",
    "Bitfinex",
    "Bittrex",
    "Huobi",
    "OKX",
    "KuCoin",
    "YoBit",
    "Hotbit",
    "BitForex",
    "LBank",
    "Coinsbit",
    "Bilaxy",
    "Crex24",
    "CoinTiger",
    "BigONE",
    "IDAX",
    "CoinExchange",
    "Hotcoin Global",
    "Graviex",
    "TradeOgre",
    "Tokocrypto",
    "WazirX",
    "CoinBene",
    "Allcoin",
    "BitMart",
    "HBTC",
    "FCoin",
    "DigiFinex",
    "CoinEx",
    "ZBG",
    "Uniswap",
    "SushiSwap",
    "PancakeSwap",
    "1inch",
    "QuickSwap",
    "Raydium",
    "Trader Joe",
    "SpookySwap",
    "Camelot"
]
print(f"Loaded {len(known_exchanges)} known exchanges")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract coin and exchange information using baseline method.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which data split to evaluate on. Default is test.")
    parser.add_argument("--test_run", type=int, default=None, 
                        help="Number of samples to process for a test run. Processes all if not specified.")
    args = parser.parse_args()

    output_filename = f"baseline_extraction_{args.split}.csv"
    if args.test_run is not None and args.test_run > 0:
        output_filename = f"baseline_extraction_{args.split}_test.csv"
        print(f"\n--- TEST RUN --- Output will be saved to {output_filename}")

    print(f"\nStep 3: Loading {args.split} data...")
    csv_path = f"{args.split}_extraction.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} {args.split} extraction windows")

    if args.test_run is not None and args.test_run > 0:
        print(f"--- TEST RUN --- Processing only the first {args.test_run} samples.")
        df = df.head(args.test_run)

    pump_idxs = df.index.tolist()
    print(f"Found {len(pump_idxs)} pump events in {args.split} set to process")

    print("\nStep 4: Processing pump events...")
    start_time = time.time()

    predictions = []
    for p in tqdm(pump_idxs, desc="Processing pump events"):
        window = eval(df.at[p, "window_messages"])
        joined = "\n".join(window)

        pred_coin = extract_ticker_from_window(window, ticker_set)
        pred_exch = None
        for ex in known_exchanges:
            if re.search(rf"\b{ex}\b", joined, re.IGNORECASE):
                pred_exch = ex
                break

        true_coin = df.at[p, "coin_cleaned"] if pd.notna(df.at[p, "coin_cleaned"]) else None
        true_exch = df.at[p, "exchange_cleaned"] if pd.notna(df.at[p, "exchange_cleaned"]) else None

        coin_pred_correct = bool(true_coin and pred_coin and true_coin.upper() == pred_coin)
        exch_pred_correct = bool(true_exch and pred_exch and true_exch.lower() == pred_exch.lower())

        predictions.append({
            'pred_coin': pred_coin,
            'pred_exch': pred_exch,
            'coin_pred_correct': coin_pred_correct,
            'exch_pred_correct': exch_pred_correct,
            'true_coin': true_coin,
            'true_exch': true_exch
        })

    end_time = time.time()
    test_time = end_time - start_time

    print("\nStep 5: Creating predictions DataFrame...")
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_filename, index=False)
    print(f"Saved predictions to {output_filename}")

    print("\nStep 6: Results:")
    correct_coin = predictions_df['coin_pred_correct'].sum()
    correct_exch = predictions_df['exch_pred_correct'].sum()
    correct_both = (predictions_df['coin_pred_correct'] & predictions_df['exch_pred_correct']).sum()
    total = len(predictions_df)

    print(f"Coin accuracy:     {correct_coin}/{total} = {correct_coin/total:.2f}")
    print(f"Exchange accuracy: {correct_exch}/{total} = {correct_exch/total:.2f}")
    print(f"Joint accuracy:    {correct_both}/{total} = {correct_both/total:.2f}")
    print(f"\nTest set evaluation time: {test_time:.2f} seconds")
    print(f"Average time per sample: {test_time/total:.4f} seconds")

    print("\nStep 7: Diagnostic - Random Sample Results:")
    import random
    sample_size = min(10, len(predictions_df))
    random_indices = random.sample(range(len(predictions_df)), sample_size)
    
    for i, idx in enumerate(random_indices):
        pred = predictions_df.iloc[idx]
        print(f"\nSample {i+1}:")
        print(f"  Predicted Coin: {pred['pred_coin']}")
        print(f"  Actual Coin:     {pred['true_coin']}")
        print(f"  Coin Correct:    {pred['coin_pred_correct']}")
        print(f"  Predicted Exchange: {pred['pred_exch']}")
        print(f"  Actual Exchange:    {pred['true_exch']}")
        print(f"  Exchange Correct:   {pred['exch_pred_correct']}")
    
    print(f"\nShowing {sample_size} random samples out of {total} total predictions")
