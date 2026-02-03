#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare upstream CLI predictions (predict/aop_predict.py) with FastAPI API outputs.

Usage:
  python3 tests/compare_cli_vs_api.py \
    --model-path predict/model/best_model_Oct13.pth \
    --api-url http://localhost:8000/api/v1/predict/batch
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_DIR = PROJECT_ROOT / "predict"
sys.path.insert(0, str(PROJECT_ROOT))
if str(PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICT_DIR))


def write_input_csv(sequences, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["SEQUENCE", "label"])
        writer.writeheader()
        for seq in sequences:
            writer.writerow({"SEQUENCE": seq, "label": 0})


def call_api(api_url, sequences, in_process=False):
    if in_process:
        from app.services.predictor import PredictionService

        service = PredictionService()
        return service.predict_batch(sequences)

    payload = json.dumps({"sequences": sequences}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def read_cli_output(csv_path):
    results = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("SEQUENCE", "").strip()
            if not seq:
                continue
            prob = float(row.get("probs", "0"))
            pred = int(float(row.get("preds", "0")))
            results[seq] = {"probability": prob, "prediction": pred}
    return results


def run_cli_via_adapter(model_path, csv_path):
    from app.adapters.core_adapter import get_core_adapter
    import torch
    from aop_dataloader import get_data_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = get_core_adapter()
    model = adapter.load_model(model_path, device)

    test_loader = get_data_loader(csv_path, batch_size=500, seq_length=50, shuffle=False)
    test_prob_list = []
    test_pred_list = []

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch["sequences"].to(device)
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_idx_tensor = batch["batch"].to(device)

            _, _, _, _, _, outputs = model(
                sequences, x, edge_index, edge_attr, batch_idx_tensor
            )
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(float)

            test_prob_list.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            test_pred_list.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])

    return test_prob_list, test_pred_list


def compare(cli_results, api_results, tolerance):
    mismatches = []
    for seq, cli in cli_results.items():
        api = api_results.get(seq)
        if not api:
            mismatches.append((seq, "missing_in_api", cli, None))
            continue
        prob_diff = abs(cli["probability"] - api["probability"])
        pred_same = cli["prediction"] == api["prediction"]
        if (prob_diff > tolerance) or (not pred_same):
            mismatches.append((seq, "diff", cli, api))
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Compare CLI vs API predictions")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1/predict/batch",
        help="Batch prediction API endpoint",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Use FastAPI TestClient instead of external HTTP",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="Probability tolerance"
    )
    parser.add_argument(
        "--use-adapter-cli",
        action="store_true",
        help="Use Adapter-based CLI flow for CPU compatibility",
    )
    args = parser.parse_args()

    sequences = [
        "MKLLVVVFCLVLAAP",
        "ACDEFGHIKLMNPQRSTVWY",
        "TTTTTTTTTTTTTTTTTTTT",
        "MKLLVVVFCLVLAAPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_csv = tmpdir_path / "input.csv"
        output_csv = tmpdir_path / "cli_output.csv"

        print("[1/3] Writing input CSV")
        write_input_csv(sequences, input_csv)

        print("[2/3] Running upstream CLI prediction")
        if args.use_adapter_cli:
            probs, preds = run_cli_via_adapter(str(args.model_path), str(input_csv))
            # Write adapter CLI output in aop_predict format
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["SEQUENCE", "probs", "preds"])
                writer.writeheader()
                for seq, prob, pred in zip(sequences, probs, preds):
                    writer.writerow({"SEQUENCE": seq, "probs": prob, "preds": pred})
        else:
            try:
                import seq_model_def as _seq_model_def
                _backend = getattr(_seq_model_def.cfg.slstm_block.slstm, "backend", None)
                if _backend == "cpu":
                    _seq_model_def.cfg.slstm_block.slstm.backend = "vanilla"
                from predict.aop_predict import aop_predict
                aop_predict(str(args.model_path), str(input_csv), str(output_csv))
            except Exception as e:
                print(f"CLI import failed: {e}")
                sys.exit(2)

        print("[3/3] Calling API and comparing")
        api_resp = call_api(args.api_url, sequences, in_process=args.in_process)
        api_results = {r["sequence"]: r for r in api_resp.get("results", [])}

        cli_results = read_cli_output(output_csv)
        mismatches = compare(cli_results, api_results, args.tolerance)

        total = len(cli_results)
        print(f"Total sequences: {total}")
        if mismatches:
            print(f"Mismatches: {len(mismatches)}")
            for seq, reason, cli, api in mismatches:
                print(f"- {seq}: {reason}")
                print(f"  CLI: {cli}")
                print(f"  API: {api}")
            sys.exit(1)
        else:
            print("All predictions match within tolerance ✅")


if __name__ == "__main__":
    main()
