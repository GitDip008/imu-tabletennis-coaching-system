import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json

from preprocessing import load_and_prepare, get_loso_splits, save_scaler
from model import build_model


#  #########################   Helpers   #########################

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts)          # inverse-frequency
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    acc  = correct / total
    loss = total_loss / total
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss, acc, f1, np.array(all_preds), np.array(all_labels)


# ######################### Main LOSO training loop #########################

def run_loso(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    df, X, y, groups = load_and_prepare(cfg["data"]["raw_path"])
    os.makedirs(cfg["evaluation"]["checkpoint_dir"], exist_ok=True)

    fold_results = []
    all_preds_global, all_labels_global = [], []

    for fold_idx, (X_tr, X_val, y_tr, y_val, subj_id, scaler) in enumerate(
        get_loso_splits(X, y, groups)
    ):
        print(f"── Fold {fold_idx+1:02d} | Subject {subj_id:03d} "
              f"| Train: {len(y_tr):,}  Val: {len(y_val):,}")

        train_loader = make_loader(X_tr, y_tr, cfg["training"]["batch_size"], shuffle=True)
        val_loader   = make_loader(X_val, y_val, cfg["training"]["batch_size"], shuffle=False)

        # Class-weighted loss to handle imbalance
        class_weights = compute_class_weights(y_tr, cfg["model"]["num_classes"], device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model     = build_model(cfg, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

        best_f1, patience_counter = 0.0, 0
        best_ckpt_path = os.path.join(
            cfg["evaluation"]["checkpoint_dir"], f"best_subj{subj_id:03d}.pt"
        )

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, preds, labels = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt_path)
                save_scaler(scaler, best_ckpt_path.replace(".pt", "_scaler.pkl"))
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:02d} | "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"  Early stop at epoch {epoch}")
                break

        # Reload the best checkpoint for final fold metrics
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        _, final_acc, final_f1, final_preds, final_labels = evaluate(
            model, val_loader, criterion, device
        )
        cm = confusion_matrix(final_labels, final_preds)

        fold_results.append({
            "subject_id": int(subj_id),
            "accuracy":   round(final_acc, 4),
            "macro_f1":   round(final_f1, 4),
            "confusion_matrix": cm.tolist(),
        })
        all_preds_global.extend(final_preds)
        all_labels_global.extend(final_labels)

        print(f"  ✓ Best → acc={final_acc:.3f}  macro-F1={final_f1:.3f}\n")

    # ── Aggregate results ──────────────────────────────────────────────────
    overall_acc = accuracy_score(all_labels_global, all_preds_global)
    overall_f1  = f1_score(all_labels_global, all_preds_global, average="macro", zero_division=0)
    global_cm   = confusion_matrix(all_labels_global, all_preds_global)

    summary = {
        "overall_accuracy": round(overall_acc, 4),
        "overall_macro_f1": round(overall_f1, 4),
        "global_confusion_matrix": global_cm.tolist(),
        "per_fold": fold_results,
    }

    results_path = os.path.join(cfg["evaluation"]["checkpoint_dir"], "loso_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("══════════════════════════════════════════")
    print(f"LOSO Overall Accuracy : {overall_acc:.4f}")
    print(f"LOSO Overall Macro-F1 : {overall_f1:.4f}")
    print(f"Results saved to      : {results_path}")
    print("══════════════════════════════════════════")
    print("\nGlobal Confusion Matrix:")
    print(global_cm)

    return summary


if __name__ == "__main__":
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    config_path = root / "config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["raw_path"]           = str(root / cfg["data"]["raw_path"])
    cfg["data"]["processed_dir"]      = str(root / cfg["data"]["processed_dir"])
    cfg["evaluation"]["checkpoint_dir"] = str(root / cfg["evaluation"]["checkpoint_dir"])

    run_loso(cfg)