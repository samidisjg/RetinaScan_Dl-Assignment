# src/engine/trainer.py
from __future__ import annotations
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.engine.metrics import multitask_loss, compute_metrics
from src.utils.common import AvgMeter, save_checkpoint


def _to_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


class Trainer:
    def __init__(self, cfg, model, train_loader, valid_loader, out_dir, device: str = "cuda"):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.out_dir = out_dir

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Coerce YAML values to correct types
        lr = _to_float(getattr(cfg, "lr", 3e-4), 3e-4)
        wd = _to_float(getattr(cfg, "weight_decay", 1e-4), 1e-4)
        epochs = _to_int(getattr(cfg, "epochs", 30), 30)
        self.label_smoothing = _to_float(getattr(cfg, "label_smoothing", 0.0), 0.0)

        # GradScaler (new API) – enabled only when CUDA is available and amp:true
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(getattr(cfg, "amp", True) and torch.cuda.is_available())
        )
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Early stopping
        self.best_f1 = -1.0
        self.no_improve = 0

        # Optional class weights
        self.grade_weights = None
        self.edema_pos_weight = None
        if getattr(cfg, "loss", None):
            if getattr(cfg.loss, "grade_class_weights", None):
                vals = [float(v) for v in cfg.loss.grade_class_weights]
                self.grade_weights = torch.tensor(vals, dtype=torch.float32, device=self.device)
            if getattr(cfg.loss, "edema_pos_weight", None) is not None:
                self.edema_pos_weight = torch.tensor(
                    [float(cfg.loss.edema_pos_weight)], dtype=torch.float32, device=self.device
                )

    def train_epoch(self, epoch: int):
        self.model.train()
        loss_meter = AvgMeter()
        grade_meter = AvgMeter()
        edema_meter = AvgMeter()

        # accumulate for train metrics
        all_g_true, all_g_pred = [], []
        all_e_true, all_e_prob = [], []

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")
        for imgs, y_grade, y_edema in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            y_grade = y_grade.to(self.device)
            y_edema = y_edema.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                self.device.type,
                enabled=bool(getattr(self.cfg, "amp", True) and torch.cuda.is_available())
            ):
                logits_g, logit_e = self.model(imgs)
                loss, lg, le = multitask_loss(
                    logits_g, logit_e, y_grade, y_edema,
                    label_smoothing=self.label_smoothing,
                    grade_weights=self.grade_weights,
                    edema_pos_weight=self.edema_pos_weight,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # meters for losses
            bs = imgs.size(0)
            loss_meter.update(loss.item(), bs)
            grade_meter.update(lg, bs)
            edema_meter.update(le, bs)

            # accumulate for metrics (detach -> cpu)
            all_g_true.append(y_grade.detach().cpu())
            all_g_pred.append(logits_g.detach().argmax(1).cpu())
            all_e_true.append(y_edema.detach().cpu())
            all_e_prob.append(torch.sigmoid(logit_e.detach()).cpu())

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                grade=f"{grade_meter.avg:.4f}",
                edema=f"{edema_meter.avg:.4f}",
            )

        self.scheduler.step()

        # compute TRAIN metrics for the epoch
        y_true_g = torch.cat(all_g_true).numpy()
        y_pred_g = torch.cat(all_g_pred).numpy()
        y_true_e = torch.cat(all_e_true).numpy()
        y_prob_e = torch.cat(all_e_prob).numpy()
        m = compute_metrics(y_true_g, y_pred_g, y_true_e, y_prob_e)

        return {
            "loss": loss_meter.avg,
            "loss_grade": grade_meter.avg,
            "loss_edema": edema_meter.avg,
            "grade_acc": float(m["grade_acc"]),
            "grade_f1_macro": float(m["grade_f1_macro"]),
            "edema_auc": float(m["edema_auc"]),
        }

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        all_g_true, all_g_pred = [], []
        all_e_true, all_e_prob = [], []
        loss_meter = AvgMeter()

        for imgs, y_grade, y_edema in tqdm(self.valid_loader, desc=f"Valid {epoch}"):
            imgs = imgs.to(self.device, non_blocking=True)
            y_grade = y_grade.to(self.device)
            y_edema = y_edema.to(self.device)

            logits_g, logit_e = self.model(imgs)
            loss, _, _ = multitask_loss(
                logits_g, logit_e, y_grade, y_edema,
                label_smoothing=0.0,
                grade_weights=self.grade_weights,
                edema_pos_weight=self.edema_pos_weight,
            )
            loss_meter.update(loss.item(), imgs.size(0))

            pred_g = logits_g.argmax(1)
            prob_e = torch.sigmoid(logit_e)
            all_g_true.append(y_grade.cpu())
            all_g_pred.append(pred_g.cpu())
            all_e_true.append(y_edema.cpu())
            all_e_prob.append(prob_e.cpu())

        y_true_g = torch.cat(all_g_true).numpy()
        y_pred_g = torch.cat(all_g_pred).numpy()
        y_true_e = torch.cat(all_e_true).numpy()
        y_prob_e = torch.cat(all_e_prob).numpy()

        metrics = compute_metrics(y_true_g, y_pred_g, y_true_e, y_prob_e)
        metrics["val_loss"] = loss_meter.avg
        return metrics

    def fit(self):
        history = {"train": [], "valid": []}
        patience = _to_int(getattr(self.cfg, "patience", 7), 7)
        keep_k = _to_int(getattr(self.cfg, "save_top_k", 3), 3)

        for epoch in range(1, _to_int(getattr(self.cfg, "epochs", 30), 30) + 1):
            tr = self.train_epoch(epoch)
            va = self.validate(epoch)
            history["train"].append(tr)
            history["valid"].append(va)

            f1 = va.get("grade_f1_macro", -1.0)
            is_best = f1 > self.best_f1
            if is_best:
                self.best_f1 = f1
                self.no_improve = 0
            else:
                self.no_improve += 1

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_f1": self.best_f1,
                    "cfg": self.cfg.d,
                    "history": history,
                },
                is_best=is_best,
                folder=self.out_dir,
                max_keep=keep_k,
            )

            if self.no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience})")
                break

        return history
