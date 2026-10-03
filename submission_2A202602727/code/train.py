"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.3,                    # chọn bằng val (Part 2: lưới 0.003..0.3, cao nhất val macro-F1)
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)

TRAIN_LOSS_SUBSET = 50_000   # số mẫu cố định của X_tr dùng để đo train_loss mỗi epoch


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    pred_pos = cm.sum(axis=0)          # số mẫu được dự đoán là lớp c
    true_pos = cm.sum(axis=1)          # số mẫu thật thuộc lớp c
    prec = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    rec = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    raise NotImplementedError  # TODO


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    n, n_classes = len(X), 7
    total_loss = 0.0
    correct = 0
    cm = torch.zeros(n_classes * n_classes, dtype=torch.int64, device=X.device)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb).float()
        # compute_loss trả trung bình theo lô -> nhân lại số mẫu để được tổng, chia N ở cuối
        total_loss += compute_loss(logits, yb, loss_name).item() * len(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        cm += torch.bincount(yb * n_classes + pred, minlength=n_classes * n_classes)
    cm = cm.view(n_classes, n_classes).cpu().numpy()
    return {"loss": total_loss / n, "acc": correct / n, "macro_f1": macro_f1_from_confusion(cm)}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        # như nn.MSELoss: không có hệ số 1/2, trung bình trên MỌI phần tử (B * 7)
        target = F.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
        return F.mse_loss(logits, target)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.

    Lựa chọn ghi lại:
      - train_loss đo ở chế độ eval() trên một tập con CỐ ĐỊNH gồm TRAIN_LOSS_SUBSET mẫu đầu của X_tr
        (X_tr đã được xáo ngẫu nhiên khi tách val nên đây là một mẫu ngẫu nhiên, giống nhau ở mọi epoch/lần chạy).
      - epoch_time_s chỉ tính phần huấn luyện (vòng qua các lô), không tính bước đánh giá cuối epoch.
      - Lô cuối nhỏ hơn batch vẫn được dùng (xem data.iterate_batches).
    """
    assert cfg.get("lr") is not None, "cfg['lr'] chưa được đặt (chọn bằng val)"
    device = data["X_tr"].device
    use_cuda = device.type == "cuda"
    hidden = tuple(cfg["hidden"])

    # ---- 0. seed, model, optimizer
    set_seed(cfg["seed"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], count_params(model)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg.get("momentum", 0.9))
    precision = cfg.get("precision", "fp32")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])

    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    X_tr_sub, y_tr_sub = X_tr[:TRAIN_LOSS_SUBSET], y_tr[:TRAIN_LOSS_SUBSET]
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 1. loss bước 0 (trước bước cập nhật đầu tiên)
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    # grad_norm_max: chuẩn lớn nhất trong epoch (thấy "gai"); clip_frac: tỉ lệ bước có chuẩn > clip_norm
    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                               "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")}
    best_val_loss, best_epoch, best_state = math.inf, None, None
    diverged = False

    # ---- 2. huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        grad_norms = []
        n_steps = n_clipped = 0
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=generator):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)
                loss = compute_loss(logits.float(), yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)      # gradient về thang thật TRƯỚC khi đo/clip
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])
            if scaler is not None:
                scaler.step(optimizer)          # tự bỏ qua bước nếu gradient có inf/NaN
                scaler.update()
            else:
                optimizer.step()
            if math.isfinite(gn):
                grad_norms.append(gn)
            n_steps += 1
            if cfg["clip_norm"] is not None and gn > cfg["clip_norm"]:
                n_clipped += 1
        if use_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - t0
        if diverged:
            print(f"[{cfg['exp_id']}] loss = NaN/inf ở epoch {epoch} -> dừng sớm (diverged)")
            break

        # cuối epoch: đánh giá ở chế độ eval()
        tr = evaluate(model, X_tr_sub, y_tr_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_macro_f1"].append(va["macro_f1"])
        history["grad_norm"].append(float(np.mean(grad_norms)) if grad_norms else float("nan"))
        history["grad_norm_max"].append(float(np.max(grad_norms)) if grad_norms else float("nan"))
        history["clip_frac"].append(n_clipped / n_steps if n_steps else 0.0)
        history["epoch_time_s"].append(epoch_time)

        if not math.isfinite(va["loss"]):
            diverged = True
            print(f"[{cfg['exp_id']}] val loss = NaN/inf ở epoch {epoch} -> dừng sớm (diverged)")
            break
        if va["loss"] < best_val_loss:
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- 3. tóm tắt tại best_epoch
    if best_epoch is not None:
        bi = history["epoch"].index(best_epoch)
        val_acc, val_f1 = history["val_acc"][bi], history["val_macro_f1"][bi]
    else:
        val_acc = val_f1 = None
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else None,
        "peak_mem_MB": torch.cuda.max_memory_allocated(device) / 2**20 if use_cuda else None,
        "diverged": diverged,
    }
    return {"cfg": copy.deepcopy(cfg), "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    raise NotImplementedError  # TODO


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    raise NotImplementedError  # TODO
