"""
Phase 2 — train the Subtask 1 risk classifier.

Usage (from esrd_project/ root, esrd2026 env active, on the A100 machine):
    # original fixed 85/15 split (backward compatible default)
    python scripts/train_risk_classifier.py

    # one fold of the 5-fold CV (see scripts/make_kfold_splits.py) — run
    # this for fold in {0,1,2,3,4} to get an honest, averaged performance
    # estimate and 5 models you can ensemble
    python scripts/train_risk_classifier.py --fold 0

    # a second/third seed on the SAME split, for seed-ensembling (mirrors
    # what train_factor_classifier.py already supports)
    python scripts/train_risk_classifier.py --seed 123 --suffix _seed123

    # final submission model: train on ALL labeled data (train+val, no
    # held-out eval), once you've picked an epoch count from your --fold
    # CV runs above and set training.final_epochs in config.yaml
    python scripts/train_risk_classifier.py --full_data
"""
import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_risk import RiskDataset, LABEL2ID, ID2LABEL
from src.models.train_utils import get_class_weights, WeightedLossTrainer, compute_metrics, full_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=None,
                    help="Use data/processed/folds/fold{N}_{train,val}.csv instead of the fixed split.")
    p.add_argument("--full_data", action="store_true",
                    help="Train on train_clean+val_clean combined, no real eval set. For the final submission model only.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed (for seed-ensembling).")
    p.add_argument("--suffix", type=str, default="", help="Suffix appended to output checkpoint/log/report dirs.")
    p.add_argument("--encoder", type=str, default="", help="Override the config encoder (e.g. microsoft/deberta-v3-base).")
    p.add_argument("--batch_size", type=int, default=0, help="Override config batch size (VRAM headroom control).")
    return p.parse_args()


def main():
    cli_args = parse_args()
    if cli_args.fold is not None and cli_args.full_data:
        raise ValueError("--fold and --full_data are mutually exclusive.")

    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
        if cli_args.seed is not None:
            cfg["project"]["seed"] = cli_args.seed

    suffix = cli_args.suffix
    if cli_args.fold is not None:
        suffix = suffix + f"_fold{cli_args.fold}"
    elif cli_args.full_data:
        suffix = suffix + "_fulldata"

    processed_dir = root / cfg["paths"]["processed_dir"]
    ckpt_dir = root / cfg["paths"]["checkpoints_dir"] / f"risk_classifier{suffix}"
    log_dir = root / cfg["paths"]["logs_dir"] / f"risk_classifier{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed data ...")
    if cli_args.fold is not None:
        fold_dir = processed_dir / "folds"
        train_df = pd.read_csv(fold_dir / f"fold{cli_args.fold}_train.csv")
        val_df = pd.read_csv(fold_dir / f"fold{cli_args.fold}_val.csv")
        print(f"  [fold {cli_args.fold}] train: {len(train_df)} rows | val: {len(val_df)} rows")
    elif cli_args.full_data:
        train_df = pd.concat([
            pd.read_csv(processed_dir / "train_clean.csv"),
            pd.read_csv(processed_dir / "val_clean.csv"),
        ], ignore_index=True)
        val_df = None
        print(f"  [full_data] train: {len(train_df)} rows (100% of labeled data). No eval set — "
              f"by design, since the point is training on everything. Epoch count comes from "
              f"training.final_epochs (set this from your --fold CV runs first, don't leave it "
              f"at the default).")
    else:
        train_df = pd.read_csv(processed_dir / "train_clean.csv")
        val_df = pd.read_csv(processed_dir / "val_clean.csv")
        print(f"  train: {len(train_df)} rows | val: {len(val_df)} rows")

    model_name = cli_args.encoder or cfg["model"]["subtask1_encoder"]
    if cli_args.batch_size:
        cfg["training"]["batch_size"] = cli_args.batch_size
        cfg["training"]["gradient_accumulation_steps"] = max(1, 16 // cli_args.batch_size)
    max_len = cfg["model"]["max_seq_len"]
    print(f"Loading tokenizer + model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = RiskDataset(train_df, tokenizer, max_length=max_len)
    val_ds = RiskDataset(val_df, tokenizer, max_length=max_len) if val_df is not None else None

    class_weights = get_class_weights(train_df)
    print(f"Class weights (Indicator, Ideation, Behavior, Attempt): {class_weights.tolist()}")

    tr_cfg = cfg["training"]
    device_has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {device_has_cuda}")

    if cli_args.full_data:
        # No real eval set to pick a "best" checkpoint against — train a
        # fixed number of epochs instead (see training.final_epochs in
        # config.yaml; falls back to training.epochs if unset), and don't
        # evaluate at all during/after training (there's nothing valid to
        # evaluate against — see val_ds = None above).
        n_epochs = tr_cfg.get("final_epochs", tr_cfg["epochs"])
        load_best = False
        eval_strategy = "no"
        save_strategy = "no"
    else:
        n_epochs = tr_cfg["epochs"]
        load_best = True
        eval_strategy = "epoch"
        save_strategy = "epoch"

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        logging_dir=str(log_dir),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=tr_cfg["batch_size"],
        per_device_eval_batch_size=tr_cfg["eval_batch_size"],
        learning_rate=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        fp16=tr_cfg["fp16"] and device_has_cuda,
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        save_safetensors=False,
        load_best_model_at_end=load_best,
        metric_for_best_model="macro_f1" if load_best else None,
        greater_is_better=True if load_best else None,
        logging_steps=20,
        report_to=["tensorboard"],
        seed=cfg["project"]["seed"],
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=(lambda p: compute_metrics(p, id2label=ID2LABEL)) if val_ds is not None else None,
        class_weights=class_weights,
    )

    print("\n=== Training ===")
    trainer.train()

    if val_ds is not None:
        print("\n=== Final evaluation on val set ===")
        metrics = trainer.evaluate()
        print(metrics)

        # detailed classification report + confusion matrix
        preds_output = trainer.predict(val_ds)
        import numpy as np
        preds = np.argmax(preds_output.predictions, axis=-1)
        labels = preds_output.label_ids
        report, cm = full_report(labels, preds, ID2LABEL)
        print("\n" + report)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
    else:
        print("\n=== full_data mode: no eval set, skipping evaluation ===")
        metrics = {"note": "full_data mode — trained on 100% of labeled data, no held-out evaluation performed. "
                            "Rely on your --fold CV runs for performance estimates of this configuration."}
        report, cm = None, None

    # save everything needed for later phases
    best_model_dir = root / "outputs" / "checkpoints" / f"risk_classifier_best{suffix}"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"\nSaved best model to {best_model_dir}")

    report_path = root / "reports" / f"phase2_risk_classifier_report{suffix}.md"
    with open(report_path, "w") as f:
        f.write("# Phase 2 — Risk Classifier (Subtask 1) Report\n\n")
        f.write(f"Model: {model_name}\n\n")
        if cli_args.fold is not None:
            f.write(f"Fold: {cli_args.fold} (of {cfg['cross_validation']['n_folds']})\n\n")
        if cli_args.full_data:
            f.write("**Trained on 100% of labeled data — no held-out evaluation was performed "
                    "(there's no valid held-out set left by design). See your --fold CV reports "
                    "for this configuration's actual performance estimate.**\n\n")
        f.write(f"Final val metrics: {json.dumps(metrics, indent=2)}\n\n")
        if report is not None:
            f.write("## Classification report\n```\n" + report + "\n```\n\n")
            f.write("## Confusion matrix (rows=true, cols=pred)\n```\n" + str(cm) + "\n```\n")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
