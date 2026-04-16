import argparse
import ast
import os
import time
import sys

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

API_URL = "https://api.deepseek.com/v1/chat/completions"

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)


def get_deepseek_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
    return api_key


def call_deepseek(prompt, model="deepseek-chat", temperature=0.0, max_retries=3):
    api_key = get_deepseek_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1000,
    }

    for attempt in range(max_retries):
        try:
            response = session.post(API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Error calling DeepSeek API after {max_retries} attempts: {str(e)}")
                if hasattr(e, "response") and e.response is not None:
                    try:
                        print(f"Response content: {e.response.text}")
                    except Exception:
                        print("Could not get response content")
                return None
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(2 ** attempt)
    return None


def create_prompt(window_list, max_messages_to_use=None):
    prompt = (
        "You are given a set of Telegram messages. "
        "Extract the coin being pumped and the exchange where it is being traded. Only one coin and one exchange is being pumped in all the list of messages.\n\n"
        "Messages:\n"
    )

    messages_for_prompt = window_list
    if max_messages_to_use is not None and max_messages_to_use > 0:
        messages_for_prompt = window_list[:max_messages_to_use]

    for i, msg in enumerate(messages_for_prompt, 1):
        prompt += f"{i}. {msg}\n"
    prompt += "\nOutput format:\nCoin: <ticker_symbol>\nExchange: <exchange>"
    return prompt


def extract_from_output(out):
    coin, exch = None, None
    for line in out.split("\n"):
        line = line.strip()
        if line.startswith("Coin:"):
            coin = line.split("Coin:")[1].strip() or None
        if line.startswith("Exchange:"):
            exch = line.split("Exchange:")[1].strip() or None
    return {"coin": coin, "exchange": exch}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract coin and exchange information from pump messages using DeepSeek API."
    )
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model to use.")
    parser.add_argument(
        "--test_run_pumps",
        type=int,
        default=None,
        help="Number of pump events to process for a test run. Processes all if not specified.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=11,
        help="Number of messages from window_messages to include in the prompt. Default is 11.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Default is 0.0.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="test_extraction.csv",
        help="Input extraction CSV file. Default is test_extraction.csv.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Optional output filename. If not set, defaults are used.",
    )
    args = parser.parse_args()

    try:
        _ = get_deepseek_key()
    except Exception as e:
        print(f"DeepSeek initialization error: {e}")
        sys.exit(1)

    output_filename = args.output_filename or "deepseek_extraction.csv"
    if args.test_run_pumps is not None and args.test_run_pumps > 0 and args.output_filename is None:
        output_filename = "deepseek_extraction_test.csv"
        print(f"\n--- TEST RUN --- Output will be saved to {output_filename}")

    print("\nStep 1: Loading and preparing data...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} total rows")

    pump_idxs = df[df["nearby_pump"] == 1].index.tolist()

    if args.test_run_pumps is not None and args.test_run_pumps > 0:
        print(f"--- TEST RUN --- Processing only the first {args.test_run_pumps} pump events.")
        pump_idxs = pump_idxs[: args.test_run_pumps]

    print(f"Found {len(pump_idxs)} pump events in test set to process")
    if args.window_size > 0:
        print(f"Using a window size of up to {args.window_size} messages for prompts.")
    else:
        print("Using all available messages from window_messages for prompts.")

    print("\nStep 2: Processing pump events...")
    df["pred_coin"] = None
    df["pred_exch"] = None
    df["coin_pred_correct"] = False
    df["exch_pred_correct"] = False

    start_time = time.time()

    for p_idx in tqdm(pump_idxs, desc="Processing pump events"):
        window = ast.literal_eval(df.at[p_idx, "window_messages"])
        prompt = create_prompt(window, args.window_size)

        if args.test_run_pumps is not None:
            print(f"\nDEBUG Prompt:\n{prompt}\n---")

        response = call_deepseek(prompt, model=args.model, temperature=args.temperature)

        if response:
            if args.test_run_pumps is not None:
                print(f"DEBUG Raw API Response: '{response}'")

            res = extract_from_output(response)
            if args.test_run_pumps is not None:
                print(f"DEBUG Extracted Info: {res}")

            df.at[p_idx, "pred_coin"] = res["coin"]
            df.at[p_idx, "pred_exch"] = res["exchange"]

            true_coin = df.at[p_idx, "pump_coin"] if pd.notna(df.at[p_idx, "pump_coin"]) else None
            true_exch = df.at[p_idx, "pump_exchange"] if pd.notna(df.at[p_idx, "pump_exchange"]) else None

            if isinstance(true_coin, str) and isinstance(res["coin"], str):
                true_coin_base = true_coin.lower().split('/')[0]
                pred_coin_base = res["coin"].lower()
                if true_coin_base == pred_coin_base:
                    df.at[p_idx, "coin_pred_correct"] = True

            if isinstance(true_exch, str) and isinstance(res["exchange"], str) and true_exch.lower() == res["exchange"].lower():
                df.at[p_idx, "exch_pred_correct"] = True

    end_time = time.time()
    test_time = end_time - start_time

    print("\nStep 3: Saving results...")
    df.to_csv(output_filename, index=False)
    print(f"Saved predictions to {output_filename}")

    print("\nStep 4: Calculating accuracy metrics...")
    total = len(pump_idxs)
    correct_coin = df.loc[pump_idxs, "coin_pred_correct"].sum()
    correct_exch = df.loc[pump_idxs, "exch_pred_correct"].sum()
    correct_both = (
        (df.loc[pump_idxs, "coin_pred_correct"]) & (df.loc[pump_idxs, "exch_pred_correct"])
    ).sum()

    print("\nResults:")
    print(f"Total pump rows:   {total}")
    print(f"Coin accuracy:     {correct_coin}/{total} = {correct_coin/total:.2f}")
    print(f"Exchange accuracy: {correct_exch}/{total} = {correct_exch/total:.2f}")
    print(f"Joint accuracy:    {correct_both}/{total} = {correct_both/total:.2f}")
    print(f"\nTest set evaluation time: {test_time:.2f} seconds")
    print(f"Average time per sample: {test_time/total:.4f} seconds")
