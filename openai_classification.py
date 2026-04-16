import argparse
import ast
import os
import random
import time
import sys

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

CLIENT = None


def get_openai_client():
    global CLIENT
    if CLIENT is not None:
        return CLIENT
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    CLIENT = OpenAI(api_key=api_key)
    return CLIENT


def call_openai(prompt, model="gpt-5.4", temperature=0.0, max_retries=3):
    client = get_openai_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error calling OpenAI API after {max_retries} attempts: {str(e)}")
                return None
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    return None


def create_prompt(window_list, max_messages_to_use=None):
    prompt = (
        "You are given a sequence of Telegram messages from a cryptocurrency pump group.\n"
        "Your task is to classify whether a pump start occurs within this set of messages.\n\n"
        "A pump start typically follows a pattern where there is a countdown (e.g., '5 minutes left', 'Get ready') followed by the release of the coin to buy (e.g., 'Buy #XYZ now').\n"
        "This may include messages identified as:\n"
        "- Pump Countdown\n"
        "- Target Coin\n"
        "- Pump Canceled after countdown\n"
        "You should ignore messages that only fall into these categories:\n"
        "- Formal Announcement (planning only)\n"
        "- Post Analysis (after the fact)\n"
        "- Rest (unrelated content)\n\n"
        "Respond with:\n"
        "1 if the window includes an actual pump start (e.g., countdown followed by a coin announcement).\n"
        "0 if not.\n\n"
        "Only return the number 1 or 0, nothing else.\n\n"
        "Messages:\n"
    )
    messages_for_prompt = window_list
    if max_messages_to_use is not None and max_messages_to_use > 0:
        messages_for_prompt = window_list[:max_messages_to_use]

    for i, msg in enumerate(messages_for_prompt, 1):
        prompt += f"{i}. {msg}\n"

    return prompt


def extract_prediction(response):
    if not response:
        return None

    response = response.strip()
    if response in ["0", "1"]:
        return int(response)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify pump start announcements using OpenAI API.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        choices=[
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.1-chat-latest",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-nano",
            "ft:gpt-4.1-nano-2025-04-14:research:pump-detector:Bv3KL4Hi",
        ],
        help="OpenAI model to use. Default is gpt-5.4.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Number of messages from window_messages to include in the prompt. Uses all if not specified.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default is 42.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Default is 0.0.")
    parser.add_argument(
        "--neg_samples",
        type=int,
        default=100,
        help="Number of class-0 samples for sampled evaluation. Default is 100.",
    )
    parser.add_argument(
        "--pos_samples",
        type=int,
        default=5,
        help="Number of class-1 samples for sampled evaluation. Default is 5.",
    )
    parser.add_argument(
        "--use_all",
        action="store_true",
        help="Use all test rows instead of class-balanced sampling.",
    )
    parser.add_argument("--input_csv", type=str, default="test_windows.csv", help="Input test windows CSV.")
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Optional output filename. If not set, defaults are used.",
    )
    args = parser.parse_args()

    try:
        get_openai_client()
    except Exception as e:
        print(f"OpenAI initialization error: {e}")
        sys.exit(1)

    np.random.seed(args.seed)

    model_name_safe = args.model.replace("-", "_").replace(":", "_").replace("/", "_")
    if args.output_filename:
        output_filename = args.output_filename
    elif args.use_all:
        output_filename = f"openai_classification_{model_name_safe}_all_data_seed_{args.seed}.csv"
    else:
        output_filename = f"openai_classification_{model_name_safe}_{args.neg_samples}_{args.pos_samples}_seed_{args.seed}.csv"

    print(f"\nOutput will be saved to {output_filename}")
    print(f"Using model: {args.model}")

    print(f"\nStep 1: Loading {args.input_csv} data...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} total test samples")

    if not args.use_all:
        print(
            f"Sampling {args.neg_samples} samples from class 0 and {args.pos_samples} samples from class 1 with seed {args.seed}..."
        )

        neg_df = df[df["nearby_pump"] == 0]
        pos_df = df[df["nearby_pump"] == 1]

        neg_take = min(args.neg_samples, len(neg_df))
        pos_take = min(args.pos_samples, len(pos_df))

        df_0 = neg_df.sample(n=neg_take, random_state=args.seed).reset_index(drop=True)
        df_1 = pos_df.sample(n=pos_take, random_state=args.seed).reset_index(drop=True)

        df = pd.concat([df_0, df_1], ignore_index=True)
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        print(f"Selected {len(df)} samples ({neg_take} from class 0, {pos_take} from class 1)")
        print(f"Class distribution: {df['nearby_pump'].value_counts().to_dict()}")
    else:
        print("Using all rows without sampling.")
        print(f"Class distribution: {df['nearby_pump'].value_counts().to_dict()}")

    print(f"Processing {len(df)} samples")
    if args.window_size is not None and args.window_size > 0:
        print(f"Using a window size of up to {args.window_size} messages for prompts.")
    else:
        print("Using all available messages from window_messages for prompts.")

    print("\nStep 2: Processing samples...")
    df["prediction"] = None
    df["prediction_correct"] = False

    start_time = time.time()

    for idx in tqdm(df.index, desc="Processing samples"):
        window = ast.literal_eval(df.at[idx, "window_messages"])
        prompt = create_prompt(window, args.window_size)

        if idx == 0 or (idx + 1) % 50 == 0:
            print(f"\n--- DEBUG: Row {idx + 1} ---")
            print(f"Prompt:\n{prompt}")
            print("---")

        response = call_openai(prompt, model=args.model, temperature=args.temperature)

        if response:
            if idx == 0 or (idx + 1) % 50 == 0:
                print(f"Response: '{response}'")
                print("---")

            pred = extract_prediction(response)
            if idx == 0 or (idx + 1) % 50 == 0:
                print(f"Extracted Prediction: {pred}")
                print("---")

            df.at[idx, "prediction"] = pred

            true_label = df.at[idx, "nearby_pump"]
            if pred is not None and pred == true_label:
                df.at[idx, "prediction_correct"] = True

    end_time = time.time()
    test_time = end_time - start_time

    print("\nStep 3: Saving results...")
    df.to_csv(output_filename, index=False)
    print(f"Saved predictions to {output_filename}")

    print("\nStep 4: Calculating accuracy metrics...")
    y_true = df["nearby_pump"].tolist()
    y_pred = [pred if pred is not None else 0 for pred in df["prediction"].tolist()]

    print(f"\nResults for test set (n={len(y_true)}):")
    print(f"Total samples: {len(y_true)}")
    print(f"Valid predictions: {sum(1 for pred in df['prediction'] if pred is not None)}")
    print(f"Invalid predictions: {sum(1 for pred in df['prediction'] if pred is None)}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["NO", "YES"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print("Predicted:")
    print("         NO  YES")
    print(f"Actual NO  {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"     YES {cm[1,0]:4d} {cm[1,1]:4d}")

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"\nEvaluation time: {test_time:.2f} seconds")
    print(f"Average time per sample: {test_time/len(y_true):.4f} seconds")
