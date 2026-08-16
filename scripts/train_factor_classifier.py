"""
Phase 4 — train the Subtask 2 multi-label factor classifier.

Usage (from esrd_project/ root, esrd2026 env active, on the A100 machine):
    python scripts/train_factor_classifier.py                              # original fixed split
    python scripts/train_factor_classifier.py --fold 0                     # one of 5 CV folds
    python scripts/train_factor_classifier.py --seed 123 --suffix _seed123 # seed ensemble (as before)
    python scripts/train_factor_classifier.py --full_data                  # final submission model

    # legacy positional form (kept working): seed, then suffix
    python scripts/train_factor_classifier.py 123 _seed123
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_factors import FactorDataset, FACTOR_LIST, NUM_FACTORS
from src.models.train_utils import (
    get_factor_pos_weights,
    WeightedBCETrainer,
    compute_multilabel_metrics,
    multilabel_full_report,
)


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


def main(seed_override=None, output_suffix="", fold=None, full_data=False, encoder="", batch_size=0):
    if fold is not None and full_data:
        raise ValueError("--fold and --full_data are mutually exclusive.")

    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
        if seed_override is not None:
            cfg["project"]["seed"] = seed_override

    suffix = output_suffix
    if fold is not None:
        suffix = suffix + f"_fold{fold}"
    elif full_data:
        suffix = suffix + "_fulldata"

    processed_dir = root / cfg["paths"]["processed_dir"]
    ckpt_dir = root / cfg["paths"]["checkpoints_dir"] / f"factor_classifier{suffix}"
    log_dir = root / cfg["paths"]["logs_dir"] / f"factor_classifier{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed data ...")
    if fold is not None:
        fold_dir = processed_dir / "folds"
        train_df = pd.read_csv(fold_dir / f"fold{fold}_train.csv")
        val_df = pd.read_csv(fold_dir / f"fold{fold}_val.csv")
        print(f"  [fold {fold}] train: {len(train_df)} rows | val: {len(val_df)} rows")
    elif full_data:
        train_df = pd.concat([
            pd.read_csv(processed_dir / "train_clean.csv"),
            pd.read_csv(processed_dir / "val_clean.csv"),
        ], ignore_index=True)
        val_df = None
        print(f"  [full_data] train: {len(train_df)} rows (100% of labeled data). No eval set — "
              f"by design. Epoch count comes from training.final_epochs (set this from your "
              f"--fold CV runs first, don't leave it at the default).")
    else:
        train_df = pd.read_csv(processed_dir / "train_clean.csv")
        val_df = pd.read_csv(processed_dir / "val_clean.csv")
        print(f"  train: {len(train_df)} rows | val: {len(val_df)} rows")

    model_name = encoder or cfg["model"]["subtask2_encoder"]
    if batch_size:
        cfg["training"]["batch_size"] = batch_size
        cfg["training"]["gradient_accumulation_steps"] = max(1, 16 // batch_size)
    max_len = cfg["model"]["max_seq_len"]
    print(f"Loading tokenizer + model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_FACTORS,
        problem_type="multi_label_classification",
        id2label={i: f for i, f in enumerate(FACTOR_LIST)},
        label2id={f: i for i, f in enumerate(FACTOR_LIST)},
    )

    train_ds = FactorDataset(train_df, tokenizer, max_length=max_len)
    val_ds = FactorDataset(val_df, tokenizer, max_length=max_len) if val_df is not None else None

    pos_weight = get_factor_pos_weights(train_ds, NUM_FACTORS)
    print("Per-factor pos_weight (dampened, sqrt of raw neg/pos ratio):")
    for name, w in zip(FACTOR_LIST, pos_weight.tolist()):
        print(f"  {name:<45} {w:.2f}")

    tr_cfg = cfg["training"]
    device_has_cuda = torch.cuda.is_available()
    print(f"\nCUDA available: {device_has_cuda}")

    if full_data:
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
        save_safetensors=False,  # same non-contiguous-tensor issue as Phases 2-3
        load_best_model_at_end=load_best,
        metric_for_best_model="factors_macro_f1" if load_best else None,
        greater_is_better=True if load_best else None,
        logging_steps=20,
        report_to=["tensorboard"],
        seed=cfg["project"]["seed"],
    )

    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_multilabel_metrics if val_ds is not None else None,
        pos_weight=pos_weight,
    )

    print("\n=== Training ===")
    trainer.train()

    if val_ds is not None:
        print("\n=== Final evaluation on val set ===")
        metrics = trainer.evaluate()
        print(metrics)

        preds_output = trainer.predict(val_ds)
        probs = 1 / (1 + np.exp(-preds_output.predictions))
        preds = (probs >= 0.5).astype(int)
        labels = preds_output.label_ids
        report = multilabel_full_report(labels, preds, FACTOR_LIST)
        print("\n" + report)
    else:
        print("\n=== full_data mode: no eval set, skipping evaluation ===")
        metrics = {"note": "full_data mode — trained on 100% of labeled data, no held-out evaluation performed. "
                            "Rely on your --fold CV runs for performance estimates of this configuration."}
        report = None

    best_model_dir = root / "outputs" / "checkpoints" / f"factor_classifier_best{suffix}"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"\nSaved best model to {best_model_dir}")

    # NOTE: previously hardcoded to phase4_factor_classifier_report.md regardless
    # of output_suffix, so every seed-ensemble run silently overwrote the same
    # file. Fixed to include suffix like the checkpoint dir already did.
    report_path = root / "reports" / f"phase4_factor_classifier_report{suffix}.md"
    with open(report_path, "w") as f:
        f.write("# Phase 4 — Factor Classifier (Subtask 2) Report\n\n")
        f.write(f"Model: {model_name}\n\n")
        if fold is not None:
            f.write(f"Fold: {fold} (of {cfg['cross_validation']['n_folds']})\n\n")
        if full_data:
            f.write("**Trained on 100% of labeled data — no held-out evaluation was performed. "
                    "See your --fold CV reports for this configuration's actual performance estimate.**\n\n")
        f.write(f"Final val metrics: {json.dumps(metrics, indent=2)}\n\n")
        if report is not None:
            f.write("## Per-factor classification report\n```\n" + report + "\n```\n")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    # backward-compatible: `python train_factor_classifier.py 123 _seed123`
    # (positional seed/suffix, no dashes) still works exactly as before, for
    # scripts/ensemble_factors.py's documented invocation.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
        suffix = sys.argv[2] if len(sys.argv) > 2 else ""
        main(seed_override=seed, output_suffix=suffix)
    else:
        cli_args = parse_args()
        main(seed_override=cli_args.seed, output_suffix=cli_args.suffix,
             fold=cli_args.fold, full_data=cli_args.full_data,
             encoder=cli_args.encoder, batch_size=cli_args.batch_size)