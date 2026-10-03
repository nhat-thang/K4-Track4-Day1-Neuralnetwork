"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    best = s.get("best_epoch")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ax = axes[0]
    ax.plot(ep, h["train_loss"], marker="o", ms=3, label="train loss (eval mode)")
    ax.plot(ep, h["val_loss"], marker="o", ms=3, label="val loss")
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.set_ylabel(f"{cfg['loss']} loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], marker="o", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], marker="o", ms=3, label="val macro-F1")
    ax.set_title("Val metric"); ax.set_xlabel("epoch"); ax.set_ylabel("giá trị")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], marker="o", ms=3, color="tab:red", label="grad_norm TB (trước clip)")
    ax.set_title("Chuẩn gradient"); ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖₂ trung bình epoch")

    for ax in axes:
        if best is not None:
            ax.axvline(best, color="gray", ls="--", lw=1, label=f"best epoch = {best}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    title = (f"{cfg['exp_id']} — {cfg['optimizer']}, lr={cfg['lr']:g}, batch={cfg['batch']}, "
             f"epochs={cfg['epochs']}, hidden={tuple(cfg['hidden'])}, init={cfg['init']}, "
             f"dropout={cfg['dropout']}, wd={cfg['weight_decay']}, clip={cfg['clip_norm']}, "
             f"{cfg['precision']}, seed={cfg['seed']}")
    if s.get("diverged"):
        title += "  [DIVERGED]"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    raise NotImplementedError  # TODO
