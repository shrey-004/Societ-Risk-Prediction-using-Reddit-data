"""
Phase 2 — shared training utilities: class-weighted loss (Attempt is only
~7% of the data, weight ~4x per Phase 1 stats) and Macro-F1 metrics.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer


def get_class_weights(train_df, label_order=("Indicator", "Ideation", "Behavior", "Attempt")):
    y = train_df["risk_label"].values
    weights = compute_class_weight("balanced", classes=np.array(label_order), y=y)
    return torch.tensor(weights, dtype=torch.float)


class WeightedLossTrainer(Trainer):
    """HF Trainer subclass that applies class weights to the loss —
    plain fine-tuning would otherwise under-predict 'Attempt' since it's
    the rarest class by a wide margin."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred, id2label=None):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    all_label_ids = sorted(id2label) if id2label is not None else None
    macro_f1 = f1_score(labels, preds, labels=all_label_ids, average="macro")
    weighted_f1 = f1_score(labels, preds, labels=all_label_ids, average="weighted")
    result = {"macro_f1": macro_f1, "weighted_f1": weighted_f1}
    if id2label is not None:
        # explicit labels= + zip against all_label_ids (not bare enumerate)
        # so a class missing from THIS eval batch still gets a correctly
        # labeled 0.0 entry instead of silently shifting every subsequent
        # class's name by one position.
        per_class = f1_score(labels, preds, labels=all_label_ids, average=None, zero_division=0)
        for class_id, f1 in zip(all_label_ids, per_class):
            result[f"f1_{id2label[class_id]}"] = f1
    return result


def full_report(labels, preds, id2label):
    target_names = [id2label[i] for i in sorted(id2label)]
    all_label_ids = sorted(id2label)
    # explicit labels= avoids a crash if this particular val/test slice
    # happens not to contain every class (e.g. a small CV fold with very
    # few Attempt examples) — sklearn infers labels from the data by
    # default, which breaks the moment target_names and the data disagree
    # on class count.
    report = classification_report(
        labels, preds, labels=all_label_ids, target_names=target_names,
        digits=3, zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=all_label_ids)
    return report, cm


# =========================================================
# Phase 3 additions — evidence span extraction (token classification)
# =========================================================
from seqeval.metrics import f1_score as seqeval_f1, classification_report as seqeval_report


def compute_token_metrics(eval_pred, id2label):
    """Entity-level Macro-F1 for BIO-tagged evidence spans, via seqeval.
    Ignores -100 (special/padding) positions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        seq_labels, seq_preds = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            seq_labels.append(id2label[l])
            seq_preds.append(id2label[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {"evidence_macro_f1": seqeval_f1(true_labels, true_preds, average="macro")}


def token_report(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        seq_labels, seq_preds = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            seq_labels.append(id2label[l])
            seq_preds.append(id2label[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)
    return seqeval_report(true_labels, true_preds, digits=3)


def get_token_class_weights(dataset, num_labels: int = 3, dampen: float = 0.5) -> torch.Tensor:
    """Compute class weights over actual token labels (ignoring -100).

    IMPORTANT: full 'balanced' inverse-frequency weighting badly overcorrected
    in practice here — recall went up but precision collapsed (0.165->0.074),
    and strict entity-level F1 got WORSE (0.197->0.119), because seqeval's
    entity matching gives zero credit for boundary-imprecise predictions, and
    the over-weighted loss pushed the model to over-predict evidence tokens.

    `dampen` < 1.0 compresses the weight spread toward 1.0 (e.g. dampen=0.5
    takes the sqrt of the balanced weights) — keeps the *direction* of the
    correction without letting it dominate the loss. This is a standard
    practical fix for token-tagging class imbalance."""
    all_labels = []
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        all_labels.extend(labels[labels != -100].tolist())
    all_labels = np.array(all_labels)
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_labels), y=all_labels
    )
    weights = weights ** dampen
    return torch.tensor(weights, dtype=torch.float)


def compute_token_level_prf(eval_pred, id2label):
    """Lenient TOKEN-level (not entity-level) precision/recall/F1 for the
    merged EVID class (B-EVID + I-EVID vs O). Complements the strict
    seqeval entity metric — useful for seeing whether the model is learning
    *something* even when boundary-strict entity F1 looks bad."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = fp = fn = 0
    for pred_row, label_row in zip(preds, labels):
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_is_evid = id2label[p] != "O"
            true_is_evid = id2label[l] != "O"
            if pred_is_evid and true_is_evid:
                tp += 1
            elif pred_is_evid and not true_is_evid:
                fp += 1
            elif not pred_is_evid and true_is_evid:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"token_precision": precision, "token_recall": recall, "token_f1": f1}


class WeightedTokenTrainer(Trainer):
    """Trainer subclass for token classification with class-weighted loss.
    Same idea as WeightedLossTrainer (Phase 2) but reshapes for the extra
    sequence-length dimension token classification has."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, num_labels)
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# =========================================================
# Phase 4 additions — multi-label factor classification (Subtask 2)
# =========================================================
from sklearn.metrics import f1_score as sk_f1_multilabel, classification_report as sk_classification_report


def get_factor_pos_weights(dataset, num_labels: int, dampen: float = 0.5) -> torch.Tensor:
    """Per-class pos_weight for BCEWithLogitsLoss, dampened the same way as
    Phase 3's token weights (sqrt of the raw neg/pos ratio) — the factor
    taxonomy ranges from 46.7% down to 0.5% prevalence (~90x range), and
    full undamped weighting would almost certainly overcorrect the same
    way it did for evidence extraction (recall up, precision destroyed)."""
    all_labels = torch.stack([dataset[i]["labels"] for i in range(len(dataset))])
    n_total = all_labels.shape[0]
    n_pos = all_labels.sum(dim=0)
    n_neg = n_total - n_pos
    raw_weight = n_neg / n_pos.clamp(min=1)
    return raw_weight.pow(dampen)


class WeightedBCETrainer(Trainer):
    """Trainer subclass for multi-label classification with per-class
    pos_weight in BCEWithLogitsLoss."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_multilabel_metrics(eval_pred, threshold: float = 0.5):
    """Macro-F1 across all 24 factor categories — this IS the official
    Subtask 2 metric, applied at a fixed 0.5 sigmoid threshold."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= threshold).astype(int)
    macro_f1 = sk_f1_multilabel(labels, preds, average="macro", zero_division=0)
    micro_f1 = sk_f1_multilabel(labels, preds, average="micro", zero_division=0)
    return {"factors_macro_f1": macro_f1, "factors_micro_f1": micro_f1}


def multilabel_full_report(labels, preds, factor_names):
    return sk_classification_report(
        labels, preds, target_names=factor_names, zero_division=0, digits=3
    )