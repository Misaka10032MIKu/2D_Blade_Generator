"""
正向代理模型（MLP）+ 逆向生成模型（条件扩散 DDPM）
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from pipeline import CascadePipeline, BladeParams
from pathlib import Path

PARAM_COLS = [
    "beta1", "beta2",
    # "t_max", "r_le", "r_te",
    "x_tmax", "front_chord", "front_camber",
    "mach_in"
]
PERF_COLS = ["loss", "pressure_rise"]


# ═══════════════════════════════════════════════════════════
#  正向代理模型
# ═══════════════════════════════════════════════════════════

class SurrogateModel(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, n_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor,
                                  n_mc: int = 50
                                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout 不确定性估计，推理时保持 dropout 激活"""
        self.train()
        preds = torch.stack([self(x) for _ in range(n_mc)])  # (n_mc, B, out)
        self.eval()
        return preds.mean(0), preds.std(0)


class SurrogateTrainer:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df[df["converged"] == True].dropna()

        X = torch.tensor(df[PARAM_COLS].values, dtype=torch.float32)
        y = torch.tensor(df[PERF_COLS].values,  dtype=torch.float32)

        # 标准化（保存统计量供推理使用）
        self.X_mean, self.X_std = X.mean(0), X.std(0).clamp(min=1e-8)
        self.y_mean, self.y_std = y.mean(0), y.std(0).clamp(min=1e-8)
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        dataset = TensorDataset(X, y)
        n_val = max(1, int(0.15 * len(dataset)))
        self.train_set, self.val_set = random_split(
            dataset, [len(dataset) - n_val, n_val]
        )
        print(f"训练集：{len(self.train_set)}，验证集：{len(self.val_set)}")

    def train(self, epochs: int = 800, lr: float = 1e-3,
              batch_size: int = 64) -> SurrogateModel:

        model = SurrogateModel(n_in=len(PARAM_COLS), n_out=len(PERF_COLS))
        opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(self.val_set,   batch_size=256)

        best_val, best_state = float("inf"), None
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                loss = loss_fn(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = sum(
                        loss_fn(model(xb), yb).item()
                        for xb, yb in val_loader
                    ) / len(val_loader)
                print(f"  Epoch {epoch+1:4d} | val loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone()
                                  for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        torch.save({
            "model": best_state,
            "X_mean": self.X_mean, "X_std": self.X_std,
            "y_mean": self.y_mean, "y_std": self.y_std,
        }, "surrogate.pt")
        print(f"正向代理模型保存至 surrogate.pt（最佳验证损失 {best_val:.4f}）")
        return model


# ═══════════════════════════════════════════════════════════
#  逆向生成模型：条件 DDPM
# ═══════════════════════════════════════════════════════════

class SinusoidalTimeEmbed(nn.Module):
    """时间步正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -torch.arange(half, device=t.device) * np.log(10000) / (half - 1)
        )
        emb = t[:, None].float() * freq[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalDenoiser(nn.Module):
    """
    去噪网络：给定噪声叶型参数 x_t、时间步 t、性能条件 c
    预测噪声 ε
    """
    def __init__(self, n_params: int, n_perf: int,
                 hidden: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_embed  = SinusoidalTimeEmbed(time_dim)
        self.cond_proj   = nn.Sequential(
            nn.Linear(n_perf, 64), nn.SiLU(), nn.Linear(64, 128)
        )
        self.net = nn.Sequential(
            nn.Linear(n_params + time_dim + 128, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_params)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        ce = self.cond_proj(cond)
        h  = torch.cat([x_t, te, ce], dim=-1)
        return self.net(h)


class CascadeDiffusion:
    """
    DDPM 训练 + 采样（含 Classifier-Free Guidance）
    """
    def __init__(self, n_params: int, n_perf: int,
                 T: int = 1000, device: str = "cpu"):
        self.T = T
        self.device = device
        self.model = ConditionalDenoiser(n_params, n_perf).to(device)

        # 噪声调度（线性）
        beta = torch.linspace(1e-4, 0.02, T, device=device)
        alpha = 1 - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

    def q_sample(self, x0: torch.Tensor,
                 t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向加噪：q(x_t | x_0)"""
        ab = self.alpha_bar[t].unsqueeze(-1)
        eps = torch.randn_like(x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps

    def train(self, csv_path: str, epochs: int = 500,
              lr: float = 2e-4, batch_size: int = 128,
              cfg_drop: float = 0.15):
        """
        cfg_drop: Classifier-Free Guidance 的条件随机丢弃概率
        """
        df = pd.read_csv(csv_path)
        df = df[df["converged"] == True].dropna()

        X = torch.tensor(df[PARAM_COLS].values, dtype=torch.float32)
        C = torch.tensor(df[PERF_COLS].values,  dtype=torch.float32)

        # 标准化
        self.X_low = X.min(0).values
        self.X_high = X.max(0).values

        X = (X - self.X_low) / (self.X_high - self.X_low)

        self.C_low = C.min(0).values
        self.C_high = C.max(0).values

        C = (C - self.C_low) / (self.C_high - self.C_low)

        loader = DataLoader(TensorDataset(X, C),
                            batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

        print(f"开始训练扩散模型，样本数：{len(X)}")
        for epoch in range(epochs):
            self.model.train()
            ep_loss = 0.0
            for x0, cond in loader:
                x0, cond = x0.to(self.device), cond.to(self.device)
                t = torch.randint(0, self.T, (x0.size(0),), device=self.device)
                x_t, eps = self.q_sample(x0, t)

                # CFG：随机丢弃条件
                mask = torch.rand(cond.size(0), device=self.device) < cfg_drop
                cond_in = cond.clone()
                cond_in[mask] = 0.0   # 用零向量表示"无条件"

                eps_pred = self.model(x_t, t, cond_in)
                loss = nn.functional.mse_loss(eps_pred, eps)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:4d} | loss {ep_loss/len(loader):.4f}")

        self.n_params = X.shape[1]

        torch.save({
            "model": self.model.state_dict(),

            # ⭐ bounded normalization（替代 mean/std）
            "X_low": self.X_low,
            "X_high": self.X_high,

            "C_low": self.C_low,
            "C_high": self.C_high,

            "T": self.T,
        }, "diffusion.pt")

    @torch.no_grad()
    def sample(self, target_perf: dict,
               n_samples: int = 10,
               cfg_scale: float = 3.0) -> np.ndarray:
        """
        给定目标性能，生成叶型参数
        target_perf: {"loss": 0.03, "pressure_rise": 1.34}
        cfg_scale:   引导强度（越大越靠近目标，但多样性降低）
        """
        self.model.eval()
        n_params = self.n_params
        device = self.device

        # 条件向量（标准化）
        c_raw = torch.tensor(
            [[target_perf.get(k, 0.0) for k in PERF_COLS]],
            dtype=torch.float32, device=device
        ).repeat(n_samples, 1)
        c_norm = (c_raw - self.C_low) / (self.C_high - self.C_low)
        c_null = torch.zeros_like(c_norm)  # 无条件引导

        # 从纯噪声开始逆向去噪
        x = torch.randn(n_samples, n_params, device=device)
        for t_val in reversed(range(self.T)):
            t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
            ab   = self.alpha_bar[t_val]
            ab_prev = self.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0)

            # CFG：条件 + 无条件预测加权
            eps_cond = self.model(x, t, c_norm)
            eps_uncond = self.model(x, t, c_null)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # DDPM 逆向更新
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            # ✔ bounded projection
            x0_pred = torch.sigmoid(x0_pred)

            # 标准后验计算
            if t_val > 0:
                noise = torch.randn_like(x)
                # 计算当前步的 alpha_t 和 beta_t
                alpha_t = ab / ab_prev
                beta_t = 1 - alpha_t

                # 计算 DDPM 标准后验均值
                mean = (ab_prev.sqrt() * beta_t / (1 - ab)) * x0_pred + \
                       (alpha_t.sqrt() * (1 - ab_prev) / (1 - ab)) * x
                # 计算后验方差
                variance = beta_t * (1 - ab_prev) / (1 - ab)

                x = mean + variance.sqrt() * noise
            else:
                x = x0_pred

        # 反标准化回物理空间
        x_phys = x * (self.X_high - self.X_low) + self.X_low
        x_phys = x_phys.cpu().numpy()

        return x_phys  # shape: (n_samples, n_params)

    def export_ta_dat(candidates, output_dir="./TA_outputs"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        pipe = CascadePipeline(root_dir=".")

        for i, params in enumerate(candidates):
            # --- 1. 安全约束（非常重要）---
            params = np.clip(params, -10, 60)

            t_max_val = max(params[2], 0.05)

            blade = BladeParams(
                beta1=float(params[0]),
                beta2=float(params[1]),
                # t_max=float(t_max_val),
                # r_le=float(params[3]),
                # r_te=float(params[4]),
                x_tmax=float(params[2]),
                front_chord=float(params[3]),
                front_camber=float(params[4])
            )

            # --- 2. 输出文件 ---
            file_path = output_dir / f"ta_{i:02d}"
            file_path.mkdir(parents=True, exist_ok=True)

            pipe.write_ta_dat(
                blade,
                file_path,
                template_path=Path("TA_template.DAT")
            )

            print(f"[OK] {file_path}")

# ═══════════════════════════════════════════════════════════
#  使用示例
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- 正向代理模型 ---
    print("=== 训练正向代理模型 ===")
    trainer = SurrogateTrainer("dataset.csv")
    surrogate = trainer.train(epochs=600)

    # --- 逆向扩散模型 ---
    print("\n=== 训练逆向扩散模型 ===")
    diffusion = CascadeDiffusion(
        n_params=len(PARAM_COLS),
        n_perf=len(PERF_COLS),
        T=1000
    )
    diffusion.train("dataset.csv", epochs=300)

    # --- 逆向设计推理 ---
    print("\n=== 逆向设计：生成低损失叶型 ===")
    candidates = diffusion.sample(
        target_perf={"loss": 0.03, "pressure_rise": 1.2},
        n_samples=20,
        cfg_scale=4.0
    )
    print(f"生成 {len(candidates)} 个候选叶型参数")
    CascadeDiffusion.export_ta_dat(candidates)

    print("候选参数（前3个）：")
    for i, row in enumerate(candidates[:3]):
        params = dict(zip(PARAM_COLS, row))
        # print(f"  [{i}] beta1={params['beta1']:.2f}° "
        #       f"beta2={params['beta2']:.2f}° "
        #       f"t_max={params['t_max']:.4f} "
        #       f"mach_in={params['mach_in']:.4f}")

        print(f"  [{i}] beta1={params['beta1']:.2f}° "
              f"beta2={params['beta2']:.2f}° "
              f" x_tmax={params['x_tmax']:.4f} "
              f"front_chord={params['front_chord']:.4f} "
              f"front_camber={params['front_camber']:.4f} "
              f"mach_in={params['mach_in']:.4f}")
