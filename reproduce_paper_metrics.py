#!/usr/bin/env python3
"""
Reviewer-focused reproduction orchestrator.

This script builds window datasets from chat_data/*/result11.csv and orchestrates
existing training/inference scripts to produce metrics-only reports for:
- main detection/extraction results
- time-series CV robustness
- centered vs causal and context-length ablations
- LLM variability and temperature sensitivity
- onset timing / detection delay (centered + causal)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import ssl
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------- logging ----------------------------------

def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


# ----------------------------- utils ------------------------------------

def to_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    return s if s else None


def normalize_exchange(value: object) -> Optional[str]:
    s = to_optional_str(value)
    return s.lower() if s else None


def normalize_coin(value: object) -> Optional[str]:
    s = to_optional_str(value)
    if not s:
        return None
    return s.lower().split("/")[0].strip() or None


def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    return float(statistics.stdev(values))


def parse_time_from_output(output: str, patterns: Sequence[str]) -> Optional[float]:
    for pattern in patterns:
        m = re.search(pattern, output)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


def sanitize_label(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s).strip("_")


def has_any_env_var(names: Sequence[str]) -> Tuple[bool, Optional[str]]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return True, name
    return False, None


def post_json(url: str, payload: Dict[str, object], headers: Optional[Dict[str, str]] = None, timeout: int = 20) -> Tuple[bool, str]:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url=url, data=body, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
            _ = resp.read()
            return True, f"http_{resp.status}"
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="ignore")[:300]
        except Exception:
            detail = ""
        return False, f"http_{e.code}:{detail}"
    except Exception as e:
        return False, str(e)


def run_preflight_checks(selected_providers: Optional[set] = None) -> Dict[str, object]:
    if selected_providers is None:
        selected_providers = {"openai", "deepseek", "gemini"}

    checks: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "api": {},
        "packages": {},
        "gpu": {},
        "selected_providers": sorted(selected_providers),
    }

    # SDK/package checks (used by pipeline scripts at runtime)
    package_specs = {
        "openai": ("openai", "openai"),
        "deepseek": ("requests", "requests"),
        "gemini": ("google.genai", "google-genai"),
    }
    for provider, (module_name, pip_name) in package_specs.items():
        selected = provider in selected_providers
        if not selected:
            checks["packages"][provider] = {
                "selected": False,
                "module": module_name,
                "pip_package": pip_name,
                "available": False,
                "detail": "not selected",
            }
            continue
        try:
            importlib.import_module(module_name)
            checks["packages"][provider] = {
                "selected": True,
                "module": module_name,
                "pip_package": pip_name,
                "available": True,
                "detail": "import_ok",
            }
        except Exception as e:
            checks["packages"][provider] = {
                "selected": True,
                "module": module_name,
                "pip_package": pip_name,
                "available": False,
                "detail": str(e),
            }

    # API checks
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    gemini_key = (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()

    if "openai" not in selected_providers:
        checks["api"]["openai"] = {"selected": False, "key_present": False, "working": False, "detail": "not selected"}
    elif openai_key:
        ok, msg = post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Reply exactly with OK."}],
                "max_completion_tokens": 3,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {openai_key}"},
        )
        checks["api"]["openai"] = {"selected": True, "key_present": True, "working": bool(ok), "detail": msg}
    else:
        checks["api"]["openai"] = {"selected": True, "key_present": False, "working": False, "detail": "missing OPENAI_API_KEY"}

    if "deepseek" not in selected_providers:
        checks["api"]["deepseek"] = {"selected": False, "key_present": False, "working": False, "detail": "not selected"}
    elif deepseek_key:
        ok, msg = post_json(
            "https://api.deepseek.com/v1/chat/completions",
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "Reply exactly with OK."}],
                "max_tokens": 3,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {deepseek_key}"},
        )
        checks["api"]["deepseek"] = {"selected": True, "key_present": True, "working": bool(ok), "detail": msg}
    else:
        checks["api"]["deepseek"] = {"selected": True, "key_present": False, "working": False, "detail": "missing DEEPSEEK_API_KEY"}

    if "gemini" not in selected_providers:
        checks["api"]["gemini"] = {"selected": False, "key_present": False, "working": False, "detail": "not selected"}
    elif gemini_key:
        ok, msg = post_json(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent"
            f"?key={gemini_key}",
            {
                "contents": [{"parts": [{"text": "Reply exactly with OK."}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 3},
            },
        )
        checks["api"]["gemini"] = {"selected": True, "key_present": True, "working": bool(ok), "detail": msg}
    else:
        checks["api"]["gemini"] = {"selected": True, "key_present": False, "working": False, "detail": "missing GEMINI_API_KEY/GOOGLE_API_KEY"}

    # GPU checks
    gpu_info: Dict[str, object] = {"torch_imported": False, "cuda_available": False, "cuda_execution_ok": False}
    try:
        import torch  # type: ignore

        gpu_info["torch_imported"] = True
        gpu_info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            t0 = time.time()
            x = torch.randn((1024, 1024), device=dev)
            y = torch.randn((1024, 1024), device=dev)
            _ = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_info["cuda_execution_ok"] = True
            gpu_info["cuda_test_runtime_sec"] = round(time.time() - t0, 6)
            gpu_info["device_count"] = int(torch.cuda.device_count())
    except Exception as e:
        gpu_info["torch_error"] = str(e)

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        if smi.returncode == 0:
            lines = [ln.strip() for ln in smi.stdout.splitlines() if ln.strip()]
            gpu_info["nvidia_smi_ok"] = True
            gpu_info["nvidia_smi_gpus"] = lines
        else:
            gpu_info["nvidia_smi_ok"] = False
            gpu_info["nvidia_smi_error"] = smi.stderr.strip() or "nvidia-smi failed"
    except Exception as e:
        gpu_info["nvidia_smi_ok"] = False
        gpu_info["nvidia_smi_error"] = str(e)

    checks["gpu"] = gpu_info
    selected_api = [checks["api"][name] for name in ["openai", "deepseek", "gemini"] if checks["api"][name].get("selected")]
    selected_packages = [
        checks["packages"][name]
        for name in ["openai", "deepseek", "gemini"]
        if checks["packages"][name].get("selected")
    ]
    checks["summary"] = {
        "all_llm_keys_working": all(item["working"] for item in selected_api) if selected_api else False,
        "all_selected_sdk_packages_available": all(item["available"] for item in selected_packages) if selected_packages else False,
        "selected_llm_count": len(selected_api),
        "gpu_ready": bool(gpu_info.get("cuda_available")) and bool(gpu_info.get("cuda_execution_ok")),
    }
    return checks


# ----------------------------- command runner ---------------------------

@dataclass
class CommandResult:
    cmd: List[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float


def run_command(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> CommandResult:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=env,
    )
    return CommandResult(
        cmd=cmd,
        cwd=cwd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_sec=time.time() - start,
    )


# ----------------------------- corpus + windows -------------------------

REQUIRED_RESULT_COLS = ["date", "text", "Pump", "Exchange", "Coin", "Notes"]


def load_message_corpus(chat_data_root: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows: List[pd.DataFrame] = []
    warnings: List[str] = []
    group_dirs = sorted([d for d in chat_data_root.iterdir() if d.is_dir()])

    for group_dir in tqdm(
        group_dirs,
        desc="Loading result11.csv files",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        result_path = group_dir / "result11.csv"
        if not result_path.exists():
            warnings.append(f"missing result11.csv in {group_dir.name}")
            continue
        try:
            df = pd.read_csv(result_path, low_memory=False)
        except Exception as exc:
            warnings.append(f"failed to read {result_path}: {exc}")
            continue

        for col in REQUIRED_RESULT_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        if "photo" not in df.columns:
            df["photo"] = pd.NA

        df = df[["date", "text", "Pump", "Exchange", "Coin", "Notes", "photo"]].copy()
        df["group"] = group_dir.name
        df["_source_row"] = np.arange(len(df), dtype=int)
        df["Pump"] = pd.to_numeric(df["Pump"], errors="coerce").fillna(0).astype(int)
        df["date_ts"] = pd.to_datetime(df["date"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["date_ts"]).copy()
        dropped = before - len(df)
        if dropped > 0:
            warnings.append(f"{group_dir.name}: dropped {dropped} rows with invalid date")
        if df.empty:
            continue

        df = df.sort_values(["date_ts", "_source_row"]).reset_index(drop=True)
        df["group_message_idx"] = np.arange(len(df), dtype=int)
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No readable result11.csv files found under {chat_data_root}")

    corpus = pd.concat(rows, ignore_index=True)
    stats = {
        "groups_found": len(group_dirs),
        "groups_with_data": int(corpus["group"].nunique()),
        "total_messages": int(len(corpus)),
        "total_pump_messages": int(corpus["Pump"].sum()),
        "warnings": warnings,
    }
    return corpus, stats


def build_windows(corpus_df: pd.DataFrame, window_size: int, window_type: str) -> pd.DataFrame:
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be odd and >= 3")
    if window_type not in {"centered", "causal"}:
        raise ValueError("window_type must be one of: centered, causal")

    half = window_size // 2
    out_rows: List[dict] = []

    grouped = list(corpus_df.groupby("group", sort=True))
    total_messages = int(sum(len(gdf) for _, gdf in grouped))
    pbar = tqdm(
        total=total_messages,
        desc=f"Building {window_type}-w{window_size} windows",
        leave=False,
        disable=not sys.stdout.isatty(),
    )

    for group, gdf in grouped:
        gdf = gdf.sort_values(["date_ts", "_source_row"]).reset_index(drop=True)
        n = len(gdf)

        for i in range(n):
            pbar.update(1)
            if window_type == "centered":
                start_idx = i - half
                end_idx = i + half
                if start_idx < 0 or end_idx >= n:
                    continue
            else:
                start_idx = i - (window_size - 1)
                end_idx = i
                if start_idx < 0:
                    continue

            w = gdf.iloc[start_idx : end_idx + 1].copy()
            center = gdf.iloc[i]

            msg_list = []
            for j, text in enumerate(w["text"].tolist(), start=1):
                text_str = "nan" if (isinstance(text, float) and math.isnan(text)) else str(text)
                msg_list.append(f"message {j}: {text_str}")

            nearby_pump = int((w["Pump"] == 1).any())
            pump_coin = None
            pump_exchange = None
            if nearby_pump:
                first_pump = w[w["Pump"] == 1].iloc[0]
                pump_coin = to_optional_str(first_pump["Coin"])
                pump_exchange = to_optional_str(first_pump["Exchange"])

            center_coin = to_optional_str(center["Coin"])
            center_exchange = to_optional_str(center["Exchange"])

            out_rows.append(
                {
                    "group": group,
                    "window_start_date": w.iloc[0]["date_ts"].strftime("%Y-%m-%d %H:%M:%S"),
                    "window_end_date": w.iloc[-1]["date_ts"].strftime("%Y-%m-%d %H:%M:%S"),
                    "nearby_pump": nearby_pump,
                    "pump_coin": pump_coin,
                    "pump_exchange": pump_exchange,
                    "window_messages": msg_list,
                    "message_count": len(msg_list),
                    "text": to_optional_str(center["text"]),
                    "Pump": int(center["Pump"]),
                    "date": center["date_ts"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Exchange": center_exchange,
                    "Coin": center_coin,
                    "Notes": to_optional_str(center["Notes"]),
                    "photo": to_optional_str(center["photo"]),
                    "exchange_cleaned": normalize_exchange(center_exchange),
                    "pump_exchange_cleaned": normalize_exchange(pump_exchange),
                    "pump_coin_cleaned": normalize_coin(pump_coin),
                    "coin_cleaned": normalize_coin(center_coin),
                    "center_idx": int(i),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "window_type": window_type,
                    "window_size": int(window_size),
                    "date_ts": center["date_ts"],
                }
            )

    pbar.close()

    if not out_rows:
        raise RuntimeError(f"No windows produced for type={window_type}, size={window_size}")

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["date_ts", "group", "center_idx"]).reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=int)
    return out


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to < 1")

    ordered = df.sort_values(["date_ts", "group", "center_idx"]).reset_index(drop=True)
    n = len(ordered)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = ordered.iloc[:train_end].copy().reset_index(drop=True)
    val_df = ordered.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = ordered.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def write_split_csvs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp["window_messages"] = tmp["window_messages"].apply(lambda x: str(x))
        tmp = tmp.drop(columns=["date_ts"], errors="ignore")
        return tmp

    paths = {
        "train_windows": out_dir / "train_windows.csv",
        "val_windows": out_dir / "val_windows.csv",
        "test_windows": out_dir / "test_windows.csv",
    }
    _prep(train_df).to_csv(paths["train_windows"], index=False)
    _prep(val_df).to_csv(paths["val_windows"], index=False)
    _prep(test_df).to_csv(paths["test_windows"], index=False)
    return paths


def create_extraction_fallback(workdir: Path) -> CommandResult:
    start = time.time()
    train = pd.read_csv(workdir / "train_windows.csv")
    val = pd.read_csv(workdir / "val_windows.csv")
    test = pd.read_csv(workdir / "test_windows.csv")
    all_df = pd.concat([train, val, test], ignore_index=True)

    # Keep only center-pump windows; aligns with existing extraction scripts.
    pump_df = all_df[all_df["Pump"] == 1].copy()
    pump_df["date"] = pd.to_datetime(pump_df["date"], errors="coerce")
    pump_df = pump_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    n = len(pump_df)
    tr = int(0.6 * n)
    va = int(0.2 * n)

    train_ex = pump_df.iloc[:tr].copy()
    val_ex = pump_df.iloc[tr : tr + va].copy()
    test_ex = pump_df.iloc[tr + va :].copy()

    train_ex.to_csv(workdir / "train_extraction.csv", index=False)
    val_ex.to_csv(workdir / "val_extraction.csv", index=False)
    test_ex.to_csv(workdir / "test_extraction.csv", index=False)

    return CommandResult(
        cmd=["fallback:create_extraction"],
        cwd=workdir,
        returncode=0,
        stdout=(
            "Fallback extraction split completed. "
            f"train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}"
        ),
        stderr="",
        duration_sec=time.time() - start,
    )


# ----------------------------- metrics ----------------------------------

def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    yt = np.array(list(y_true), dtype=int)
    yp = np.array(list(y_pred), dtype=int)

    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = (recall + tnr) / 2.0
    accuracy = (tp + tn) / len(yt) if len(yt) else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "accuracy": accuracy,
    }


def extraction_metrics_from_bool_df(df: pd.DataFrame, total_mask: Optional[pd.Series] = None) -> Dict[str, float]:
    if total_mask is None:
        total_mask = pd.Series([True] * len(df), index=df.index)

    subset = df[total_mask].copy()
    if subset.empty:
        return {
            "total": 0,
            "coin_accuracy": float("nan"),
            "exchange_accuracy": float("nan"),
            "joint_accuracy": float("nan"),
        }

    coin = subset["coin_pred_correct"].fillna(False).astype(bool)
    exch = subset["exch_pred_correct"].fillna(False).astype(bool)
    joint = coin & exch

    return {
        "total": int(len(subset)),
        "coin_accuracy": float(coin.mean()),
        "exchange_accuracy": float(exch.mean()),
        "joint_accuracy": float(joint.mean()),
    }


def extraction_metrics_from_seq2seq(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "total": 0,
            "coin_accuracy": float("nan"),
            "exchange_accuracy": float("nan"),
            "joint_accuracy": float("nan"),
        }

    ac = df["actual_coin"].fillna("").astype(str).str.lower().str.split("/").str[0]
    ae = df["actual_exchange"].fillna("").astype(str).str.lower()

    coin_col = "predicted_coin" if "predicted_coin" in df.columns else "pred_coin"
    exch_col = "predicted_exchange" if "predicted_exchange" in df.columns else "pred_exchange"

    pc = df[coin_col].fillna("").astype(str).str.lower().str.split("/").str[0]
    pe = df[exch_col].fillna("").astype(str).str.lower()

    coin_ok = ac == pc
    exch_ok = ae == pe
    joint_ok = coin_ok & exch_ok

    return {
        "total": int(len(df)),
        "coin_accuracy": float(coin_ok.mean()),
        "exchange_accuracy": float(exch_ok.mean()),
        "joint_accuracy": float(joint_ok.mean()),
    }


# ----------------------------- model runners ----------------------------

@dataclass
class ModelRun:
    method: str
    status: str
    run_name: str
    window_type: Optional[str] = None
    window_size: Optional[int] = None
    fold_id: Optional[int] = None
    temperature: Optional[float] = None
    time_per_sample: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    predictions: Optional[List[int]] = None
    command_result: Optional[CommandResult] = None
    output_path: Optional[Path] = None
    note: Optional[str] = None


def ensure_support_files(base_dir: Path, workdir: Path) -> None:
    crypto_src = base_dir / "cryptocurrencies.json"
    if crypto_src.exists():
        crypto_dst = workdir / "cryptocurrencies.json"
        if not crypto_dst.exists():
            shutil.copy2(crypto_src, crypto_dst)


def run_lightgbm(base_dir: Path, workdir: Path, features: int) -> ModelRun:
    script = base_dir / "lightgbm_classification.py"
    cmd = [sys.executable, str(script), "--features", str(features)]
    result = run_command(cmd, cwd=workdir)

    if result.returncode != 0:
        return ModelRun(method="LightGBM", status="failed", run_name="main", command_result=result, note="lightgbm script failed")

    pred_path = workdir / "lightgbm_predictions.csv"
    if not pred_path.exists():
        return ModelRun(method="LightGBM", status="failed", run_name="main", command_result=result, note="missing lightgbm_predictions.csv")

    pred_df = pd.read_csv(pred_path)
    met = classification_metrics(pred_df["true_label"].tolist(), pred_df["lightgbm_prediction"].tolist())
    tps = parse_time_from_output(
        result.stdout,
        [
            r"Average time per row \(Test\):\s*([0-9.eE+-]+)",
            r"Average time per sample \(Test\):\s*([0-9.eE+-]+)",
        ],
    )
    return ModelRun(
        method="LightGBM",
        status="ok",
        run_name="main",
        time_per_sample=tps,
        metrics=met,
        predictions=[int(x) for x in pred_df["lightgbm_prediction"].tolist()],
        command_result=result,
        output_path=pred_path,
    )


def run_bge(base_dir: Path, workdir: Path, epochs: int, batch_size: int, max_length: int, learning_rate: float) -> ModelRun:
    script = base_dir / "bge_classification.py"
    cmd = [
        sys.executable,
        str(script),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--max_length",
        str(max_length),
        "--learning_rate",
        str(learning_rate),
    ]
    result = run_command(cmd, cwd=workdir)

    if result.returncode != 0:
        return ModelRun(method="BGE-M3", status="failed", run_name="main", command_result=result, note="bge script failed")

    pred_path = workdir / "bge_predictions.csv"
    if not pred_path.exists():
        return ModelRun(method="BGE-M3", status="failed", run_name="main", command_result=result, note="missing bge_predictions.csv")

    pred_df = pd.read_csv(pred_path)
    met = classification_metrics(pred_df["true_label"].tolist(), pred_df["bge_prediction"].tolist())
    tps = parse_time_from_output(
        result.stdout,
        [
            r"Average time per sample \(Test\):\s*([0-9.eE+-]+)",
            r"Average time per row \(Test\):\s*([0-9.eE+-]+)",
        ],
    )

    return ModelRun(
        method="BGE-M3",
        status="ok",
        run_name="main",
        time_per_sample=tps,
        metrics=met,
        predictions=[int(x) for x in pred_df["bge_prediction"].tolist()],
        command_result=result,
        output_path=pred_path,
    )


def run_openai_detection_sample(
    base_dir: Path,
    workdir: Path,
    model_name: str,
    seed: int,
    temperature: float,
    neg_samples: int,
    pos_samples: int,
    run_label: str,
) -> ModelRun:
    script = base_dir / "openai_classification.py"
    out_name = f"openai_detection_{sanitize_label(run_label)}.csv"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model_name,
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--neg_samples",
        str(neg_samples),
        "--pos_samples",
        str(pos_samples),
        "--input_csv",
        "test_windows.csv",
        "--output_filename",
        out_name,
    ]
    result = run_command(cmd, cwd=workdir)

    if result.returncode != 0:
        return ModelRun(method="LLM-GPT-Detection", status="failed", run_name=run_label, temperature=temperature, command_result=result)

    pred_path = workdir / out_name
    if not pred_path.exists():
        return ModelRun(
            method="LLM-GPT-Detection",
            status="failed",
            run_name=run_label,
            temperature=temperature,
            command_result=result,
            note="missing detection output csv",
        )

    pred_df = pd.read_csv(pred_path)
    y_true = pd.to_numeric(pred_df["nearby_pump"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(pred_df["prediction"], errors="coerce").fillna(0).astype(int)
    met = classification_metrics(y_true.tolist(), y_pred.tolist())
    tps = parse_time_from_output(result.stdout, [r"Average time per sample:\s*([0-9.eE+-]+)"])

    return ModelRun(
        method="LLM-GPT-Detection",
        status="ok",
        run_name=run_label,
        temperature=temperature,
        time_per_sample=tps,
        metrics=met,
        predictions=y_pred.tolist(),
        command_result=result,
        output_path=pred_path,
    )


def run_create_extraction_dataset(base_dir: Path, workdir: Path) -> ModelRun:
    script = base_dir / "create_extraction_datasets.py"
    cmd = [sys.executable, str(script)]
    result = run_command(cmd, cwd=workdir)
    if result.returncode == 0:
        return ModelRun(method="ExtractionSplit", status="ok", run_name="main", command_result=result)

    # fallback for environments missing optional dependencies imported by the script
    fb = create_extraction_fallback(workdir)
    return ModelRun(method="ExtractionSplit", status="ok", run_name="main", command_result=fb, note="used fallback split")


def run_baseline_extraction(base_dir: Path, workdir: Path) -> ModelRun:
    ensure_support_files(base_dir, workdir)
    script = base_dir / "baseline_extraction.py"
    cmd = [sys.executable, str(script), "--split", "test"]
    result = run_command(cmd, cwd=workdir)

    if result.returncode != 0:
        return ModelRun(method="Rule-based", status="failed", run_name="main", command_result=result)

    out_path = workdir / "baseline_extraction_test.csv"
    if not out_path.exists():
        out_path = workdir / "baseline_extraction_test.csv"
    if not out_path.exists():
        return ModelRun(method="Rule-based", status="failed", run_name="main", command_result=result, note="missing baseline output")

    out_df = pd.read_csv(out_path)
    met = extraction_metrics_from_bool_df(out_df)
    tps = parse_time_from_output(result.stdout, [r"Average time per sample:\s*([0-9.eE+-]+)"])

    return ModelRun(
        method="Rule-based",
        status="ok",
        run_name="main",
        time_per_sample=tps,
        metrics=met,
        command_result=result,
        output_path=out_path,
    )


def run_seq2seq_extraction(base_dir: Path, workdir: Path) -> ModelRun:
    script = base_dir / "seq2seq_extraction.py"
    cmd = [sys.executable, str(script)]
    result = run_command(cmd, cwd=workdir)

    if result.returncode != 0:
        return ModelRun(method="Longformer", status="failed", run_name="main", command_result=result)

    out_path = workdir / "test_predictions.csv"
    if not out_path.exists():
        return ModelRun(method="Longformer", status="failed", run_name="main", command_result=result, note="missing test_predictions.csv")

    out_df = pd.read_csv(out_path)
    met = extraction_metrics_from_seq2seq(out_df)
    tps = parse_time_from_output(result.stdout, [r"Average time per sample:\s*([0-9.eE+-]+)"])

    return ModelRun(
        method="Longformer",
        status="ok",
        run_name="main",
        time_per_sample=tps,
        metrics=met,
        command_result=result,
        output_path=out_path,
    )


def run_llm_extraction(
    base_dir: Path,
    workdir: Path,
    script_name: str,
    method_name: str,
    run_label: str,
    model: Optional[str],
    temperature: float,
    window_size: int,
) -> ModelRun:
    script = base_dir / script_name
    out_name = f"{sanitize_label(method_name.lower())}_{sanitize_label(run_label)}.csv"

    cmd = [
        sys.executable,
        str(script),
        "--input_csv",
        "test_extraction.csv",
        "--window_size",
        str(window_size),
        "--temperature",
        str(temperature),
        "--output_filename",
        out_name,
    ]
    if model:
        cmd.extend(["--model", model])

    result = run_command(cmd, cwd=workdir)
    if result.returncode != 0:
        return ModelRun(method=method_name, status="failed", run_name=run_label, temperature=temperature, command_result=result)

    out_path = workdir / out_name
    if not out_path.exists():
        return ModelRun(method=method_name, status="failed", run_name=run_label, temperature=temperature, command_result=result, note="missing extraction output")

    out_df = pd.read_csv(out_path)
    if "nearby_pump" in out_df.columns:
        mask = pd.to_numeric(out_df["nearby_pump"], errors="coerce").fillna(0).astype(int) == 1
    else:
        mask = None

    met = extraction_metrics_from_bool_df(out_df, total_mask=mask)
    tps = parse_time_from_output(result.stdout, [r"Average time per sample:\s*([0-9.eE+-]+)"])

    return ModelRun(
        method=method_name,
        status="ok",
        run_name=run_label,
        temperature=temperature,
        time_per_sample=tps,
        metrics=met,
        command_result=result,
        output_path=out_path,
    )


# ----------------------------- CV folds ---------------------------------

def build_expanding_folds(
    pretest_df: pd.DataFrame,
    max_folds: int,
    embargo: int,
    min_train_pos: int,
    min_val_pos: int,
) -> Tuple[int, List[Dict[str, int]]]:
    n = len(pretest_df)
    if n < 100:
        raise RuntimeError("Not enough pre-test rows for CV")

    for k in [max_folds, 4, 3]:
        if k <= 0:
            continue
        block = n // (k + 1)
        if block <= 0:
            continue

        folds: List[Dict[str, int]] = []
        ok = True

        for i in range(1, k + 1):
            raw_train_end = i * block
            raw_val_end = n if i == k else (i + 1) * block

            train_end = raw_train_end - embargo
            val_start = raw_train_end + embargo
            val_end = raw_val_end - (embargo if i < k else 0)

            if train_end <= 0 or val_end <= val_start:
                ok = False
                break

            train_slice = pretest_df.iloc[:train_end]
            val_slice = pretest_df.iloc[val_start:val_end]

            train_pos = int(pd.to_numeric(train_slice["nearby_pump"], errors="coerce").fillna(0).sum())
            val_pos = int(pd.to_numeric(val_slice["nearby_pump"], errors="coerce").fillna(0).sum())

            if train_pos < min_train_pos or val_pos < min_val_pos:
                ok = False
                break

            folds.append(
                {
                    "fold_id": i,
                    "train_start": 0,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "train_rows": int(len(train_slice)),
                    "val_rows": int(len(val_slice)),
                    "train_pos": train_pos,
                    "val_pos": val_pos,
                }
            )

        if ok and len(folds) == k:
            return k, folds

    raise RuntimeError("Could not build feasible CV folds under current constraints")


# ----------------------------- delay analysis ---------------------------

def compute_delay_metrics(test_df: pd.DataFrame, y_pred: Sequence[int], method: str, window_type: str, window_size: int) -> Dict[str, object]:
    if len(test_df) != len(y_pred):
        raise ValueError(f"prediction length mismatch for {method}/{window_type}: {len(test_df)} vs {len(y_pred)}")

    df = test_df.copy().reset_index(drop=True)
    df["y_pred"] = np.array(y_pred, dtype=int)

    if "date_ts" not in df.columns:
        df["date_ts"] = pd.to_datetime(df["date"], errors="coerce")

    onsets = df[df["Pump"] == 1].copy()

    delays_msg: List[float] = []
    delays_sec: List[float] = []
    misses = 0

    for _, onset in tqdm(
        onsets.iterrows(),
        total=len(onsets),
        desc=f"Delay {method} {window_type}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        mask = (
            (df["group"] == onset["group"])
            & (df["y_pred"] == 1)
            & (pd.to_numeric(df["start_idx"], errors="coerce") <= int(onset["center_idx"]))
            & (pd.to_numeric(df["end_idx"], errors="coerce") >= int(onset["center_idx"]))
        )
        candidates = df[mask].copy()

        if candidates.empty:
            misses += 1
            continue

        chosen = candidates.sort_values("center_idx").iloc[0]
        d_msg = int(chosen["center_idx"]) - int(onset["center_idx"])

        onset_ts = pd.to_datetime(onset["date"], errors="coerce") if pd.isna(onset.get("date_ts")) else onset["date_ts"]
        pred_ts = pd.to_datetime(chosen["date"], errors="coerce") if pd.isna(chosen.get("date_ts")) else chosen["date_ts"]

        d_sec = float((pred_ts - onset_ts).total_seconds()) if pd.notna(onset_ts) and pd.notna(pred_ts) else float("nan")

        delays_msg.append(float(d_msg))
        delays_sec.append(d_sec)

    total = int(len(onsets))
    detected = int(total - misses)

    valid_sec = [x for x in delays_sec if not math.isnan(x)]

    def pct(vals: List[float], p: float) -> float:
        if not vals:
            return float("nan")
        return float(np.percentile(np.array(vals), p))

    def hit_rate(vals: List[float], th: float) -> float:
        if total == 0:
            return float("nan")
        hit = sum(1 for x in vals if x <= th)
        return hit / total

    summary = {
        "method": method,
        "window_type": window_type,
        "window_size": window_size,
        "total_events": total,
        "detected_events": detected,
        "missed_events": int(misses),
        "miss_rate": (misses / total) if total else float("nan"),
        "median_delay_messages": pct(delays_msg, 50),
        "p90_delay_messages": pct(delays_msg, 90),
        "median_delay_seconds": pct(valid_sec, 50),
        "p90_delay_seconds": pct(valid_sec, 90),
        "hit_rate_msg_le_0": hit_rate(delays_msg, 0),
        "hit_rate_msg_le_1": hit_rate(delays_msg, 1),
        "hit_rate_msg_le_5": hit_rate(delays_msg, 5),
        "hit_rate_sec_le_0": hit_rate(valid_sec, 0),
        "hit_rate_sec_le_30": hit_rate(valid_sec, 30),
        "hit_rate_sec_le_60": hit_rate(valid_sec, 60),
        "min_delay_messages": min(delays_msg) if delays_msg else float("nan"),
        "max_delay_messages": max(delays_msg) if delays_msg else float("nan"),
    }

    # Sanity expectation for causal windows
    if window_type == "causal" and delays_msg:
        min_delay = min(delays_msg)
        summary["causal_negative_delay_violation"] = int(min_delay < 0)
    else:
        summary["causal_negative_delay_violation"] = 0

    return summary


# ----------------------------- report helpers ---------------------------

def aggregate_classification_runs(method: str, runs: List[ModelRun], run_name: str) -> Dict[str, object]:
    ok_runs = [r for r in runs if r.status == "ok" and r.metrics is not None]
    if not ok_runs:
        return {
            "method": method,
            "run_name": run_name,
            "status": "skipped",
            "n_runs": 0,
        }

    fields = ["precision", "recall", "f1", "balanced_accuracy", "accuracy", "time_per_sample"]
    out: Dict[str, object] = {
        "method": method,
        "run_name": run_name,
        "status": "ok",
        "n_runs": len(ok_runs),
    }

    for f in fields:
        vals = [float(r.metrics[f]) if f in r.metrics else float(r.time_per_sample or float("nan")) for r in ok_runs]
        if f == "time_per_sample":
            vals = [float(r.time_per_sample) for r in ok_runs if r.time_per_sample is not None]
        out[f"{f}_mean"] = safe_mean(vals)
        out[f"{f}_std"] = safe_std(vals)

    # confusion values only make sense for deterministic single-run (or sum for visibility)
    out["tp_sum"] = int(sum(int(r.metrics["tp"]) for r in ok_runs if r.metrics))
    out["tn_sum"] = int(sum(int(r.metrics["tn"]) for r in ok_runs if r.metrics))
    out["fp_sum"] = int(sum(int(r.metrics["fp"]) for r in ok_runs if r.metrics))
    out["fn_sum"] = int(sum(int(r.metrics["fn"]) for r in ok_runs if r.metrics))
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        if columns:
            for c in columns:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[columns]
    else:
        df = pd.DataFrame(columns=columns or [])
    df.to_csv(path, index=False)


# ----------------------------- main -------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer-focused reproduction extension")
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--chat_data_root", type=Path, default=None)
    parser.add_argument("--reports_dir", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window_sizes", type=str, default="7,11,15")
    parser.add_argument("--window_types", type=str, default="centered,causal")
    parser.add_argument("--base_window_size", type=int, default=11)

    parser.add_argument("--cv_max_folds", type=int, default=5)
    parser.add_argument("--cv_embargo", type=int, default=10)
    parser.add_argument("--cv_min_train_pos", type=int, default=1000)
    parser.add_argument("--cv_min_val_pos", type=int, default=200)

    parser.add_argument("--llm_n_runs", type=int, default=5)
    parser.add_argument("--llm_temps", type=str, default="0.0,0.2,0.5")
    parser.add_argument(
        "--llm_providers",
        type=str,
        default="all",
        help="Comma-separated LLM providers to run: all, openai, deepseek, gemini",
    )

    parser.add_argument("--gpt_detection_model", type=str, default="gpt-5.4")
    parser.add_argument("--gpt_detection_neg_samples", type=int, default=100)
    parser.add_argument("--gpt_detection_pos_samples", type=int, default=5)

    parser.add_argument("--lightgbm_features", type=int, default=15000)
    parser.add_argument("--bge_epochs", type=int, default=2)
    parser.add_argument("--bge_batch_size", type=int, default=16)
    parser.add_argument("--bge_max_length", type=int, default=1024)
    parser.add_argument("--bge_learning_rate", type=float, default=3e-5)

    parser.add_argument("--run_bge", action="store_true", default=False)
    parser.add_argument("--run_seq2seq", action="store_true", default=False)
    parser.add_argument("--run_api", action="store_true", default=False)
    parser.add_argument("--preflight_checks", action="store_true", default=False)
    parser.add_argument("--preflight_only", action="store_true", default=False)
    parser.add_argument("--clean_workdir", action="store_true", default=False)

    args = parser.parse_args()

    project_root = args.project_root.resolve()
    chat_data_root = (args.chat_data_root or (project_root / "chat_data")).resolve()
    reports_dir = (args.reports_dir or (project_root / "reports")).resolve()
    work_root = reports_dir / "work"

    if args.clean_workdir and work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "project_root": str(project_root),
        "chat_data_root": str(chat_data_root),
        "reports_dir": str(reports_dir),
        "config": {
            "seed": args.seed,
            "window_sizes": args.window_sizes,
            "window_types": args.window_types,
            "base_window_size": args.base_window_size,
            "cv_max_folds": args.cv_max_folds,
            "cv_embargo": args.cv_embargo,
            "cv_min_train_pos": args.cv_min_train_pos,
            "cv_min_val_pos": args.cv_min_val_pos,
            "llm_n_runs": args.llm_n_runs,
            "llm_temps": args.llm_temps,
            "llm_providers": args.llm_providers,
            "run_bge": args.run_bge,
            "run_seq2seq": args.run_seq2seq,
            "run_api": args.run_api,
            "preflight_checks": args.preflight_checks,
            "preflight_only": args.preflight_only,
        },
        "stages": [],
        "skipped_models": [],
        "warnings": [],
    }

    def stage_record(name: str, status: str, extra: Optional[Dict[str, object]] = None) -> None:
        rec = {"name": name, "status": status}
        if extra:
            rec.update(extra)
        manifest["stages"].append(rec)

    raw_providers = [x.strip().lower() for x in args.llm_providers.split(",") if x.strip()]
    valid_providers = {"openai", "deepseek", "gemini"}
    if not raw_providers or "all" in raw_providers:
        selected_llm_providers = set(valid_providers)
    else:
        invalid = sorted(set(raw_providers) - valid_providers)
        if invalid:
            raise ValueError(f"Invalid --llm_providers values: {invalid}. Valid: {sorted(valid_providers)} or 'all'")
        selected_llm_providers = set(raw_providers)
    provider_enabled = {p: (p in selected_llm_providers) for p in valid_providers}
    manifest["llm_provider_selection"] = {p: provider_enabled[p] for p in sorted(valid_providers)}

    try:
        if args.preflight_checks or args.preflight_only:
            log("Running preflight checks for API keys and GPU...")
            preflight = run_preflight_checks(selected_providers=selected_llm_providers)
            manifest["preflight"] = preflight
            stage_record("preflight_checks", "ok", preflight.get("summary", {}))
            (reports_dir / "preflight_checks.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
            log(f"Preflight summary: {preflight.get('summary', {})}")
            if args.preflight_only:
                manifest["finished_at"] = datetime.now().isoformat()
                manifest["status"] = "ok"
                (reports_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                log(f"Wrote manifest: {reports_dir / 'run_manifest.json'}")
                return

        log("Loading raw message corpus from result11.csv files...")
        corpus, corpus_stats = load_message_corpus(chat_data_root)
        manifest["corpus_stats"] = corpus_stats
        manifest["warnings"].extend(corpus_stats.get("warnings", []))
        stage_record("load_corpus", "ok", {"messages": corpus_stats["total_messages"]})

        window_sizes = [int(x.strip()) for x in args.window_sizes.split(",") if x.strip()]
        window_types = [x.strip() for x in args.window_types.split(",") if x.strip()]
        llm_temps = [float(x.strip()) for x in args.llm_temps.split(",") if x.strip()]

        openai_ready, openai_env = has_any_env_var(["OPENAI_API_KEY"])
        deepseek_ready, deepseek_env = has_any_env_var(["DEEPSEEK_API_KEY"])
        gemini_ready, gemini_env = has_any_env_var(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
        provider_ready = {
            "openai": openai_ready,
            "deepseek": deepseek_ready,
            "gemini": gemini_ready,
        }
        manifest["api_credentials"] = {
            "openai": {"available": openai_ready, "env_var": openai_env},
            "deepseek": {"available": deepseek_ready, "env_var": deepseek_env},
            "gemini": {"available": gemini_ready, "env_var": gemini_env},
        }

        variant_cache: Dict[Tuple[str, int], Dict[str, object]] = {}

        def prepare_variant(window_type: str, window_size: int) -> Dict[str, object]:
            key = (window_type, window_size)
            if key in variant_cache:
                return variant_cache[key]

            log(f"Building windows for variant: {window_type}, size={window_size}")
            full = build_windows(corpus, window_size=window_size, window_type=window_type)
            train_df, val_df, test_df = chronological_split(full)
            variant_dir = work_root / f"{window_type}_w{window_size}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            write_split_csvs(train_df, val_df, test_df, variant_dir)

            record = {
                "full": full,
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "workdir": variant_dir,
                "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            }
            variant_cache[key] = record
            return record

        # Main baseline variant
        main_variant = prepare_variant("centered", args.base_window_size)
        main_workdir = main_variant["workdir"]
        stage_record("build_windows_main", "ok", main_variant["split_sizes"])

        # Run detection main
        detection_main_rows: List[Dict[str, object]] = []
        detection_gpt_runs: List[ModelRun] = []
        detection_models_for_delay: Dict[Tuple[str, str, int], ModelRun] = {}

        log("Running main detection models...")
        lgm_main = run_lightgbm(project_root, main_workdir, features=args.lightgbm_features)
        detection_models_for_delay[("LightGBM", "centered", args.base_window_size)] = lgm_main

        if lgm_main.status == "ok":
            row = {
                "method": "LightGBM",
                "window_type": "centered",
                "window_size": args.base_window_size,
                "precision_mean": lgm_main.metrics["precision"],
                "recall_mean": lgm_main.metrics["recall"],
                "f1_mean": lgm_main.metrics["f1"],
                "balanced_accuracy_mean": lgm_main.metrics["balanced_accuracy"],
                "accuracy_mean": lgm_main.metrics["accuracy"],
                "time_per_sample_mean": lgm_main.time_per_sample,
                "precision_std": 0.0,
                "recall_std": 0.0,
                "f1_std": 0.0,
                "balanced_accuracy_std": 0.0,
                "accuracy_std": 0.0,
                "time_per_sample_std": 0.0,
                "tp": lgm_main.metrics["tp"],
                "tn": lgm_main.metrics["tn"],
                "fp": lgm_main.metrics["fp"],
                "fn": lgm_main.metrics["fn"],
                "n_runs": 1,
                "status": "ok",
            }
        else:
            row = {"method": "LightGBM", "status": lgm_main.status, "window_type": "centered", "window_size": args.base_window_size}
        detection_main_rows.append(row)

        if args.run_bge:
            bge_main = run_bge(
                project_root,
                main_workdir,
                epochs=args.bge_epochs,
                batch_size=args.bge_batch_size,
                max_length=args.bge_max_length,
                learning_rate=args.bge_learning_rate,
            )
            detection_models_for_delay[("BGE-M3", "centered", args.base_window_size)] = bge_main
            if bge_main.status == "ok":
                row = {
                    "method": "BGE-M3",
                    "window_type": "centered",
                    "window_size": args.base_window_size,
                    "precision_mean": bge_main.metrics["precision"],
                    "recall_mean": bge_main.metrics["recall"],
                    "f1_mean": bge_main.metrics["f1"],
                    "balanced_accuracy_mean": bge_main.metrics["balanced_accuracy"],
                    "accuracy_mean": bge_main.metrics["accuracy"],
                    "time_per_sample_mean": bge_main.time_per_sample,
                    "precision_std": 0.0,
                    "recall_std": 0.0,
                    "f1_std": 0.0,
                    "balanced_accuracy_std": 0.0,
                    "accuracy_std": 0.0,
                    "time_per_sample_std": 0.0,
                    "tp": bge_main.metrics["tp"],
                    "tn": bge_main.metrics["tn"],
                    "fp": bge_main.metrics["fp"],
                    "fn": bge_main.metrics["fn"],
                    "n_runs": 1,
                    "status": "ok",
                }
            else:
                row = {"method": "BGE-M3", "status": bge_main.status, "window_type": "centered", "window_size": args.base_window_size}
            detection_main_rows.append(row)
        else:
            detection_main_rows.append({"method": "BGE-M3", "status": "skipped", "note": "run_bge flag not set", "window_type": "centered", "window_size": args.base_window_size})

        # GPT detection variability (sampled)
        if args.run_api:
            if not provider_enabled["openai"]:
                note = "provider disabled via --llm_providers"
                manifest["skipped_models"].append({"model": "LLM (GPT sampled)", "reason": note})
                detection_main_rows.append(
                    {
                        "method": "LLM (GPT sampled)",
                        "status": "skipped",
                        "note": note,
                        "window_type": "centered",
                        "window_size": args.base_window_size,
                    }
                )
            elif openai_ready:
                log("Running sampled GPT detection variability runs...")
                for i in tqdm(
                    range(args.llm_n_runs),
                    desc="GPT detection variability",
                    leave=False,
                    disable=not sys.stdout.isatty(),
                ):
                    run = run_openai_detection_sample(
                        project_root,
                        main_workdir,
                        model_name=args.gpt_detection_model,
                        seed=args.seed + i,
                        temperature=0.0,
                        neg_samples=args.gpt_detection_neg_samples,
                        pos_samples=args.gpt_detection_pos_samples,
                        run_label=f"variability_run_{i+1}",
                    )
                    detection_gpt_runs.append(run)

                agg = aggregate_classification_runs("LLM (GPT sampled)", detection_gpt_runs, "variability")
                agg["window_type"] = "centered"
                agg["window_size"] = args.base_window_size
                detection_main_rows.append(agg)
            else:
                note = "missing OPENAI_API_KEY"
                manifest["skipped_models"].append({"model": "LLM (GPT sampled)", "reason": note})
                detection_main_rows.append(
                    {
                        "method": "LLM (GPT sampled)",
                        "status": "skipped",
                        "note": note,
                        "window_type": "centered",
                        "window_size": args.base_window_size,
                    }
                )
        else:
            detection_main_rows.append(
                {
                    "method": "LLM (GPT sampled)",
                    "status": "skipped",
                    "note": "run_api flag not set",
                    "window_type": "centered",
                    "window_size": args.base_window_size,
                }
            )

        stage_record("run_detection_main", "ok")

        # Build extraction split for main variant
        log("Creating extraction datasets...")
        extraction_split_run = run_create_extraction_dataset(project_root, main_workdir)
        stage_record("create_extraction_split", extraction_split_run.status, {"note": extraction_split_run.note})

        # Extraction main
        extraction_main_rows: List[Dict[str, object]] = []
        extraction_llm_variability_rows: List[Dict[str, object]] = []
        extraction_llm_temp_rows: List[Dict[str, object]] = []

        log("Running extraction main methods...")
        baseline_main = run_baseline_extraction(project_root, main_workdir)
        if baseline_main.status == "ok":
            extraction_main_rows.append(
                {
                    "method": "Rule-based",
                    "status": "ok",
                    "coin_accuracy": baseline_main.metrics["coin_accuracy"],
                    "exchange_accuracy": baseline_main.metrics["exchange_accuracy"],
                    "joint_accuracy": baseline_main.metrics["joint_accuracy"],
                    "time_per_sample": baseline_main.time_per_sample,
                }
            )
        else:
            extraction_main_rows.append({"method": "Rule-based", "status": baseline_main.status})

        if args.run_seq2seq:
            seq_main = run_seq2seq_extraction(project_root, main_workdir)
            if seq_main.status == "ok":
                extraction_main_rows.append(
                    {
                        "method": "Longformer",
                        "status": "ok",
                        "coin_accuracy": seq_main.metrics["coin_accuracy"],
                        "exchange_accuracy": seq_main.metrics["exchange_accuracy"],
                        "joint_accuracy": seq_main.metrics["joint_accuracy"],
                        "time_per_sample": seq_main.time_per_sample,
                    }
                )
            else:
                extraction_main_rows.append({"method": "Longformer", "status": seq_main.status})
        else:
            extraction_main_rows.append({"method": "Longformer", "status": "skipped", "note": "run_seq2seq flag not set"})

        llm_specs = [
            {
                "script": "deepseek_extraction.py",
                "method": "DeepSeek",
                "model": "deepseek-chat",
                "provider": "deepseek",
                "missing_key_note": "missing DEEPSEEK_API_KEY",
            },
            {
                "script": "gemini_extraction.py",
                "method": "Gemini",
                "model": "gemini-3.1-pro-preview",
                "provider": "gemini",
                "missing_key_note": "missing GEMINI_API_KEY/GOOGLE_API_KEY",
            },
            {
                "script": "openai_extraction.py",
                "method": "GPT",
                "model": "gpt-5.4",
                "provider": "openai",
                "missing_key_note": "missing OPENAI_API_KEY",
            },
        ]

        llm_main_runs: Dict[str, List[ModelRun]] = {x["method"]: [] for x in llm_specs}

        if args.run_api:
            for spec in tqdm(
                llm_specs,
                desc="Extraction LLM methods",
                leave=False,
                disable=not sys.stdout.isatty(),
            ):
                if not provider_enabled.get(spec["provider"], False):
                    note = "provider disabled via --llm_providers"
                    extraction_main_rows.append({"method": spec["method"], "status": "skipped", "note": note})
                    extraction_llm_variability_rows.append(
                        {
                            "method": spec["method"],
                            "run_name": "skipped",
                            "temperature": 0.0,
                            "status": "skipped",
                        }
                    )
                    extraction_llm_temp_rows.append(
                        {
                            "method": spec["method"],
                            "temperature": float("nan"),
                            "run_name": "skipped",
                            "status": "skipped",
                        }
                    )
                    manifest["skipped_models"].append({"model": spec["method"], "reason": note})
                    continue

                if not provider_ready.get(spec["provider"], False):
                    extraction_main_rows.append({"method": spec["method"], "status": "skipped", "note": spec["missing_key_note"]})
                    extraction_llm_variability_rows.append(
                        {
                            "method": spec["method"],
                            "run_name": "skipped",
                            "temperature": 0.0,
                            "status": "skipped",
                        }
                    )
                    extraction_llm_temp_rows.append(
                        {
                            "method": spec["method"],
                            "temperature": float("nan"),
                            "run_name": "skipped",
                            "status": "skipped",
                        }
                    )
                    manifest["skipped_models"].append({"model": spec["method"], "reason": spec["missing_key_note"]})
                    continue

                # main single run (temperature 0.0)
                main_run = run_llm_extraction(
                    project_root,
                    main_workdir,
                    script_name=spec["script"],
                    method_name=spec["method"],
                    run_label="main",
                    model=spec["model"],
                    temperature=0.0,
                    window_size=args.base_window_size,
                )
                llm_main_runs[spec["method"]].append(main_run)

                if main_run.status == "ok":
                    extraction_main_rows.append(
                        {
                            "method": spec["method"],
                            "status": "ok",
                            "coin_accuracy": main_run.metrics["coin_accuracy"],
                            "exchange_accuracy": main_run.metrics["exchange_accuracy"],
                            "joint_accuracy": main_run.metrics["joint_accuracy"],
                            "time_per_sample": main_run.time_per_sample,
                        }
                    )
                else:
                    extraction_main_rows.append({"method": spec["method"], "status": main_run.status})

                # variability at fixed production settings (T=0)
                for i in tqdm(
                    range(args.llm_n_runs),
                    desc=f"{spec['method']} variability",
                    leave=False,
                    disable=not sys.stdout.isatty(),
                ):
                    run = run_llm_extraction(
                        project_root,
                        main_workdir,
                        script_name=spec["script"],
                        method_name=spec["method"],
                        run_label=f"variability_run_{i+1}",
                        model=spec["model"],
                        temperature=0.0,
                        window_size=args.base_window_size,
                    )
                    if run.status == "ok":
                        extraction_llm_variability_rows.append(
                            {
                                "method": spec["method"],
                                "run_name": run.run_name,
                                "temperature": run.temperature,
                                "coin_accuracy": run.metrics["coin_accuracy"],
                                "exchange_accuracy": run.metrics["exchange_accuracy"],
                                "joint_accuracy": run.metrics["joint_accuracy"],
                                "time_per_sample": run.time_per_sample,
                                "status": run.status,
                            }
                        )
                    else:
                        extraction_llm_variability_rows.append(
                            {
                                "method": spec["method"],
                                "run_name": run.run_name,
                                "temperature": run.temperature,
                                "status": run.status,
                            }
                        )

                # temperature sensitivity runs
                temp_run_grid = [(temp, i) for temp in llm_temps for i in range(args.llm_n_runs)]
                for temp, i in tqdm(
                    temp_run_grid,
                    desc=f"{spec['method']} temp sweep",
                    leave=False,
                    disable=not sys.stdout.isatty(),
                ):
                        run = run_llm_extraction(
                            project_root,
                            main_workdir,
                            script_name=spec["script"],
                            method_name=spec["method"],
                            run_label=f"temp_{temp}_run_{i+1}",
                            model=spec["model"],
                            temperature=temp,
                            window_size=args.base_window_size,
                        )
                        extraction_llm_temp_rows.append(
                            {
                                "method": spec["method"],
                                "temperature": temp,
                                "run_name": run.run_name,
                                "coin_accuracy": run.metrics["coin_accuracy"] if run.metrics else float("nan"),
                                "exchange_accuracy": run.metrics["exchange_accuracy"] if run.metrics else float("nan"),
                                "joint_accuracy": run.metrics["joint_accuracy"] if run.metrics else float("nan"),
                                "time_per_sample": run.time_per_sample,
                                "status": run.status,
                            }
                        )
        else:
            for spec in llm_specs:
                extraction_main_rows.append({"method": spec["method"], "status": "skipped", "note": "run_api flag not set"})

        # add variability summaries
        if extraction_llm_variability_rows:
            var_df = pd.DataFrame(extraction_llm_variability_rows)
            ok = var_df[var_df["status"] == "ok"].copy()
            if not ok.empty:
                for method, g in ok.groupby("method"):
                    extraction_llm_variability_rows.append(
                        {
                            "method": method,
                            "run_name": "summary_mean",
                            "temperature": 0.0,
                            "coin_accuracy": g["coin_accuracy"].mean(),
                            "exchange_accuracy": g["exchange_accuracy"].mean(),
                            "joint_accuracy": g["joint_accuracy"].mean(),
                            "time_per_sample": g["time_per_sample"].mean(),
                            "status": "ok",
                        }
                    )
                    extraction_llm_variability_rows.append(
                        {
                            "method": method,
                            "run_name": "summary_std",
                            "temperature": 0.0,
                            "coin_accuracy": g["coin_accuracy"].std(ddof=1) if len(g) > 1 else 0.0,
                            "exchange_accuracy": g["exchange_accuracy"].std(ddof=1) if len(g) > 1 else 0.0,
                            "joint_accuracy": g["joint_accuracy"].std(ddof=1) if len(g) > 1 else 0.0,
                            "time_per_sample": g["time_per_sample"].std(ddof=1) if len(g) > 1 else 0.0,
                            "status": "ok",
                        }
                    )

        if extraction_llm_temp_rows:
            temp_df = pd.DataFrame(extraction_llm_temp_rows)
            ok = temp_df[temp_df["status"] == "ok"].copy()
            if not ok.empty:
                for (method, temp), g in ok.groupby(["method", "temperature"]):
                    extraction_llm_temp_rows.append(
                        {
                            "method": method,
                            "temperature": temp,
                            "run_name": "summary_mean",
                            "coin_accuracy": g["coin_accuracy"].mean(),
                            "exchange_accuracy": g["exchange_accuracy"].mean(),
                            "joint_accuracy": g["joint_accuracy"].mean(),
                            "time_per_sample": g["time_per_sample"].mean(),
                            "status": "ok",
                        }
                    )
                    extraction_llm_temp_rows.append(
                        {
                            "method": method,
                            "temperature": temp,
                            "run_name": "summary_std",
                            "coin_accuracy": g["coin_accuracy"].std(ddof=1) if len(g) > 1 else 0.0,
                            "exchange_accuracy": g["exchange_accuracy"].std(ddof=1) if len(g) > 1 else 0.0,
                            "joint_accuracy": g["joint_accuracy"].std(ddof=1) if len(g) > 1 else 0.0,
                            "time_per_sample": g["time_per_sample"].std(ddof=1) if len(g) > 1 else 0.0,
                            "status": "ok",
                        }
                    )

        stage_record("run_extraction", "ok")

        # Time-series CV for detection
        detection_cv_rows: List[Dict[str, object]] = []
        log("Running detection CV...")
        pretest_df = pd.concat([main_variant["train"], main_variant["val"]], ignore_index=True)

        try:
            chosen_k, folds = build_expanding_folds(
                pretest_df,
                max_folds=args.cv_max_folds,
                embargo=args.cv_embargo,
                min_train_pos=args.cv_min_train_pos,
                min_val_pos=args.cv_min_val_pos,
            )
            manifest["cv"] = {"chosen_k": chosen_k, "folds": folds}

            for fold in tqdm(
                folds,
                desc="Detection CV folds",
                leave=False,
                disable=not sys.stdout.isatty(),
            ):
                fold_id = int(fold["fold_id"])
                fold_dir = work_root / f"cv_fold_{fold_id}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                tr = pretest_df.iloc[fold["train_start"] : fold["train_end"]].copy().reset_index(drop=True)
                va = pretest_df.iloc[fold["val_start"] : fold["val_end"]].copy().reset_index(drop=True)
                write_split_csvs(tr, va, va, fold_dir)

                lg = run_lightgbm(project_root, fold_dir, features=args.lightgbm_features)
                if lg.status == "ok":
                    detection_cv_rows.append(
                        {
                            "model": "LightGBM",
                            "fold_id": fold_id,
                            "k": chosen_k,
                            "train_rows": fold["train_rows"],
                            "val_rows": fold["val_rows"],
                            "train_pos": fold["train_pos"],
                            "val_pos": fold["val_pos"],
                            "precision": lg.metrics["precision"],
                            "recall": lg.metrics["recall"],
                            "f1": lg.metrics["f1"],
                            "time_per_sample": lg.time_per_sample,
                            "status": "ok",
                        }
                    )
                else:
                    detection_cv_rows.append({"model": "LightGBM", "fold_id": fold_id, "k": chosen_k, "status": lg.status})

                if args.run_bge:
                    bg = run_bge(
                        project_root,
                        fold_dir,
                        epochs=args.bge_epochs,
                        batch_size=args.bge_batch_size,
                        max_length=args.bge_max_length,
                        learning_rate=args.bge_learning_rate,
                    )
                    if bg.status == "ok":
                        detection_cv_rows.append(
                            {
                                "model": "BGE-M3",
                                "fold_id": fold_id,
                                "k": chosen_k,
                                "train_rows": fold["train_rows"],
                                "val_rows": fold["val_rows"],
                                "train_pos": fold["train_pos"],
                                "val_pos": fold["val_pos"],
                                "precision": bg.metrics["precision"],
                                "recall": bg.metrics["recall"],
                                "f1": bg.metrics["f1"],
                                "time_per_sample": bg.time_per_sample,
                                "status": "ok",
                            }
                        )
                    else:
                        detection_cv_rows.append({"model": "BGE-M3", "fold_id": fold_id, "k": chosen_k, "status": bg.status})

            # summaries
            if detection_cv_rows:
                cv_df = pd.DataFrame(detection_cv_rows)
                ok_cv = cv_df[cv_df["status"] == "ok"].copy()
                if not ok_cv.empty:
                    for model, g in ok_cv.groupby("model"):
                        detection_cv_rows.append(
                            {
                                "model": model,
                                "fold_id": "mean",
                                "k": chosen_k,
                                "precision": g["precision"].mean(),
                                "recall": g["recall"].mean(),
                                "f1": g["f1"].mean(),
                                "time_per_sample": g["time_per_sample"].mean(),
                                "status": "ok",
                            }
                        )
                        detection_cv_rows.append(
                            {
                                "model": model,
                                "fold_id": "std",
                                "k": chosen_k,
                                "precision": g["precision"].std(ddof=1) if len(g) > 1 else 0.0,
                                "recall": g["recall"].std(ddof=1) if len(g) > 1 else 0.0,
                                "f1": g["f1"].std(ddof=1) if len(g) > 1 else 0.0,
                                "time_per_sample": g["time_per_sample"].std(ddof=1) if len(g) > 1 else 0.0,
                                "status": "ok",
                            }
                        )
            stage_record("run_detection_cv", "ok", {"chosen_k": chosen_k})
        except Exception as exc:
            stage_record("run_detection_cv", "failed", {"error": str(exc)})
            manifest["warnings"].append(f"CV failed: {exc}")

        # Detection ablations (window type + size)
        detection_ablation_rows: List[Dict[str, object]] = []
        log("Running detection window ablations...")
        baseline_by_model: Dict[str, Dict[str, float]] = {}

        ablation_combos = [(wtype, wsize) for wtype in window_types for wsize in window_sizes]
        for wtype, wsize in tqdm(
            ablation_combos,
            desc="Detection window ablations",
            leave=False,
            disable=not sys.stdout.isatty(),
        ):
                var = prepare_variant(wtype, wsize)
                workdir = var["workdir"]

                lg = run_lightgbm(project_root, workdir, features=args.lightgbm_features)
                if lg.status == "ok":
                    detection_ablation_rows.append(
                        {
                            "model": "LightGBM",
                            "window_type": wtype,
                            "window_size": wsize,
                            "precision": lg.metrics["precision"],
                            "recall": lg.metrics["recall"],
                            "f1": lg.metrics["f1"],
                            "time_per_sample": lg.time_per_sample,
                            "status": "ok",
                        }
                    )
                    if wtype == "centered" and wsize == args.base_window_size:
                        baseline_by_model["LightGBM"] = {
                            "precision": lg.metrics["precision"],
                            "recall": lg.metrics["recall"],
                            "f1": lg.metrics["f1"],
                            "time_per_sample": lg.time_per_sample or float("nan"),
                        }
                        detection_models_for_delay[("LightGBM", wtype, wsize)] = lg
                    if wtype == "causal" and wsize == args.base_window_size:
                        detection_models_for_delay[("LightGBM", wtype, wsize)] = lg
                else:
                    detection_ablation_rows.append(
                        {"model": "LightGBM", "window_type": wtype, "window_size": wsize, "status": lg.status}
                    )

                if args.run_bge:
                    bg = run_bge(
                        project_root,
                        workdir,
                        epochs=args.bge_epochs,
                        batch_size=args.bge_batch_size,
                        max_length=args.bge_max_length,
                        learning_rate=args.bge_learning_rate,
                    )
                    if bg.status == "ok":
                        detection_ablation_rows.append(
                            {
                                "model": "BGE-M3",
                                "window_type": wtype,
                                "window_size": wsize,
                                "precision": bg.metrics["precision"],
                                "recall": bg.metrics["recall"],
                                "f1": bg.metrics["f1"],
                                "time_per_sample": bg.time_per_sample,
                                "status": "ok",
                            }
                        )
                        if wtype == "centered" and wsize == args.base_window_size:
                            baseline_by_model["BGE-M3"] = {
                                "precision": bg.metrics["precision"],
                                "recall": bg.metrics["recall"],
                                "f1": bg.metrics["f1"],
                                "time_per_sample": bg.time_per_sample or float("nan"),
                            }
                            detection_models_for_delay[("BGE-M3", wtype, wsize)] = bg
                        if wtype == "causal" and wsize == args.base_window_size:
                            detection_models_for_delay[("BGE-M3", wtype, wsize)] = bg
                    else:
                        detection_ablation_rows.append(
                            {"model": "BGE-M3", "window_type": wtype, "window_size": wsize, "status": bg.status}
                        )

        # add deltas vs centered-11 baseline
        if detection_ablation_rows:
            for row in detection_ablation_rows:
                if row.get("status") != "ok":
                    continue
                base = baseline_by_model.get(row["model"])
                if not base:
                    continue
                row["delta_precision_vs_centered11"] = row["precision"] - base["precision"]
                row["delta_recall_vs_centered11"] = row["recall"] - base["recall"]
                row["delta_f1_vs_centered11"] = row["f1"] - base["f1"]
                row["delta_time_vs_centered11"] = (
                    (row["time_per_sample"] - base["time_per_sample"])
                    if row.get("time_per_sample") is not None and not math.isnan(base["time_per_sample"])
                    else float("nan")
                )

        stage_record("run_detection_ablations", "ok")

        # Delay analysis (centered-11 and causal-11)
        delay_rows: List[Dict[str, object]] = []
        for model in ["LightGBM", "BGE-M3"]:
            if model == "BGE-M3" and not args.run_bge:
                continue
            for wtype in ["centered", "causal"]:
                key = (model, wtype, args.base_window_size)
                run = detection_models_for_delay.get(key)
                if not run or run.status != "ok" or run.predictions is None:
                    delay_rows.append(
                        {
                            "method": model,
                            "window_type": wtype,
                            "window_size": args.base_window_size,
                            "status": "skipped",
                        }
                    )
                    continue

                test_df = variant_cache[(wtype, args.base_window_size)]["test"].copy()
                try:
                    summary = compute_delay_metrics(test_df, run.predictions, model=model, window_type=wtype, window_size=args.base_window_size)
                    summary["status"] = "ok"
                    delay_rows.append(summary)
                except Exception as exc:
                    delay_rows.append(
                        {
                            "method": model,
                            "window_type": wtype,
                            "window_size": args.base_window_size,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

        stage_record("run_delay_analysis", "ok")

        # ---------------- write reports ----------------
        detection_main_columns = [
            "method",
            "window_type",
            "window_size",
            "status",
            "n_runs",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "accuracy_mean",
            "accuracy_std",
            "time_per_sample_mean",
            "time_per_sample_std",
            "tp",
            "tn",
            "fp",
            "fn",
            "tp_sum",
            "tn_sum",
            "fp_sum",
            "fn_sum",
            "note",
        ]

        detection_cv_columns = [
            "model",
            "fold_id",
            "k",
            "train_rows",
            "val_rows",
            "train_pos",
            "val_pos",
            "precision",
            "recall",
            "f1",
            "time_per_sample",
            "status",
        ]

        detection_ablation_columns = [
            "model",
            "window_type",
            "window_size",
            "status",
            "precision",
            "recall",
            "f1",
            "time_per_sample",
            "delta_precision_vs_centered11",
            "delta_recall_vs_centered11",
            "delta_f1_vs_centered11",
            "delta_time_vs_centered11",
        ]

        delay_columns = [
            "method",
            "window_type",
            "window_size",
            "status",
            "total_events",
            "detected_events",
            "missed_events",
            "miss_rate",
            "median_delay_messages",
            "p90_delay_messages",
            "median_delay_seconds",
            "p90_delay_seconds",
            "hit_rate_msg_le_0",
            "hit_rate_msg_le_1",
            "hit_rate_msg_le_5",
            "hit_rate_sec_le_0",
            "hit_rate_sec_le_30",
            "hit_rate_sec_le_60",
            "min_delay_messages",
            "max_delay_messages",
            "causal_negative_delay_violation",
            "error",
        ]

        extraction_main_columns = [
            "method",
            "status",
            "coin_accuracy",
            "exchange_accuracy",
            "joint_accuracy",
            "time_per_sample",
            "note",
        ]

        extraction_var_columns = [
            "method",
            "run_name",
            "temperature",
            "status",
            "coin_accuracy",
            "exchange_accuracy",
            "joint_accuracy",
            "time_per_sample",
        ]

        extraction_temp_columns = [
            "method",
            "temperature",
            "run_name",
            "status",
            "coin_accuracy",
            "exchange_accuracy",
            "joint_accuracy",
            "time_per_sample",
        ]

        write_csv(reports_dir / "detection_main.csv", detection_main_rows, detection_main_columns)
        write_csv(reports_dir / "detection_cv.csv", detection_cv_rows, detection_cv_columns)
        write_csv(reports_dir / "detection_ablation_windows.csv", detection_ablation_rows, detection_ablation_columns)
        write_csv(reports_dir / "delay_analysis.csv", delay_rows, delay_columns)
        write_csv(reports_dir / "extraction_main.csv", extraction_main_rows, extraction_main_columns)
        write_csv(reports_dir / "extraction_llm_variability.csv", extraction_llm_variability_rows, extraction_var_columns)
        write_csv(reports_dir / "extraction_llm_temp_sensitivity.csv", extraction_llm_temp_rows, extraction_temp_columns)

        manifest["finished_at"] = datetime.now().isoformat()
        manifest["status"] = "ok"

    except Exception as exc:
        manifest["finished_at"] = datetime.now().isoformat()
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        log(f"Pipeline failed: {exc}")

    (reports_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {reports_dir / 'run_manifest.json'}")

    if manifest.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
