"""
Phase 3 — train the evidence span extraction model (BIO token classification).

Usage (from esrd_project/ root, esrd2026 env active, on the A100 machine):
    python scripts/train_evidence_extractor.py                  # original fixed split
    python scripts/train_evidence_extractor.py --fold 0          # one of 5 CV folds
    python scripts/train_evidence_extractor.py --seed 123 --suffix _seed123   # seed ensemble
    python scripts/train_evidence_extractor.py --full_data        # final submission model
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
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_evidence import EvidenceDataset, LABEL2ID, ID2LABEL
from src.models.train_utils import (
    compute_token_metrics,
    token_report,
    get_token_class_weights,
    WeightedTokenTrainer,
    compute_token_level_prf,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=None,
                    help="Use data/processed/folds/fold{N}_{train,val}.csv instead of the fixed split.")
    p.add_argument("--full_data", action="store_true",
                    help="Train on train_clean+val_clean combined, no real eval set. For the final submission model only.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed (for seed-ensembling).")
    p.add_argument("--suffix", type=str, default="", help="Suffix appended to output checkpoint/log/report dirs.")
    p.add_argument("--use_fuzzy_evidence_match", action="store_true",
                    help="Enable tier-3 fuzzy span matching in dataset_evidence.py. Off by default — "
                         "run scripts/audit_evidence_matches.py and review flagged rows first, see that "
                         "script's docstring for why (found a real annotation error, row P01511).")
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
    ckpt_dir = root / cfg["paths"]["checkpoints_dir"] / f"evidence_extractor{suffix}"
    log_dir = root / cfg["paths"]["logs_dir"] / f"evidence_extractor{suffix}"
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
              f"by design. Epoch count comes from training.final_epochs (set this from your "
              f"--fold CV runs first, don't leave it at the default).")
    else:
        train_df = pd.read_csv(processed_dir / "train_clean.csv")
        val_df = pd.read_csv(processed_dir / "val_clean.csv")
        print(f"  train: {len(train_df)} rows | val: {len(val_df)} rows")

    # reuse the same encoder as Subtask 1 (config-driven, swappable)
    model_name = cfg["model"]["subtask1_encoder"]
    max_len = cfg["model"]["max_seq_len"]
    print(f"Loading tokenizer + model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = EvidenceDataset(train_df, tokenizer, max_length=max_len,
                                use_fuzzy=cli_args.use_fuzzy_evidence_match)
    val_ds = EvidenceDataset(val_df, tokenizer, max_length=max_len,
                              use_fuzzy=cli_args.use_fuzzy_evidence_match) if val_df is not None else None

    # sanity check: how many spans actually got tagged (some evidence,
    # per Phase 1's ~6% residual paraphrasing, won't align to any tokens)
    n_with_evidence_tag = sum(
        1 for i in range(len(train_ds)) if (train_ds[i]["labels"] != 0).any()
        and (train_ds[i]["labels"] != -100).any()
    )
    print(f"Train rows with at least one B/I-EVID tag: {n_with_evidence_tag}/{len(train_ds)}")

    class_weights = get_token_class_weights(train_ds)
    print(f"Token class weights (O, B-EVID, I-EVID): {class_weights.tolist()}")

    tr_cfg = cfg["training"]
    device_has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {device_has_cuda}")

    if cli_args.full_data:
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
        save_safetensors=False,   # same non-contiguous-tensor issue as Phase 2
        load_best_model_at_end=load_best,
        metric_for_best_model="evidence_macro_f1" if load_best else None,
        greater_is_better=True if load_best else None,
        logging_steps=20,
        report_to=["tensorboard"],
        seed=cfg["project"]["seed"],
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=(lambda p: {
            **compute_token_metrics(p, id2label=ID2LABEL),
            **compute_token_level_prf(p, id2label=ID2LABEL),
        }) if val_ds is not None else None,
        class_weights=class_weights,
    )

    print("\n=== Training ===")
    trainer.train()

    if val_ds is not None:
        print("\n=== Final evaluation on val set ===")
        metrics = trainer.evaluate()
        print(metrics)

        preds_output = trainer.predict(val_ds)
        report = token_report(
            (preds_output.predictions, preds_output.label_ids), ID2LABEL
        )
        print("\n" + report)
    else:
        print("\n=== full_data mode: no eval set, skipping evaluation ===")
        metrics = {"note": "full_data mode — trained on 100% of labeled data, no held-out evaluation performed. "
                            "Rely on your --fold CV runs for performance estimates of this configuration."}
        report = None

    best_model_dir = root / "outputs" / "checkpoints" / f"evidence_extractor_best{suffix}"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"\nSaved best model to {best_model_dir}")

    report_path = root / "reports" / f"phase3_evidence_extractor_report{suffix}.md"
    with open(report_path, "w") as f:
        f.write("# Phase 3 — Evidence Span Extraction Report\n\n")
        f.write(f"Model: {model_name}\n\n")
        if cli_args.fold is not None:
            f.write(f"Fold: {cli_args.fold} (of {cfg['cross_validation']['n_folds']})\n\n")
        if cli_args.full_data:
            f.write("**Trained on 100% of labeled data — no held-out evaluation was performed. "
                    "See your --fold CV reports for this configuration's actual performance estimate.**\n\n")
        f.write(f"Fuzzy evidence matching: {'ON' if cli_args.use_fuzzy_evidence_match else 'off (default)'}\n\n")
        f.write(f"Rows with at least one taggable span: {n_with_evidence_tag}/{len(train_ds)}\n\n")
        f.write(f"Final val metrics: {json.dumps(metrics, indent=2)}\n\n")
        if report is not None:
            f.write("## seqeval classification report\n```\n" + report + "\n```\n")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
