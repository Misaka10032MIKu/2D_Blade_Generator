"""
拉丁超立方采样 + 批量数据集生成
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pipeline import CascadePipeline, BladeParams


# ── 参数空间定义（根据叶型实际范围调整）─────────────────────────
PARAM_BOUNDS = {
    # 参数名        [下界,    上界]
    "beta1":       [48.0,   50.0],   # 进口气流角 (度)
    "beta2":       [ -2.0,   2.0],   # 出口气流角 (度)
    # "t_max":       [ 0.05,   0.12],  # 最大相对厚度
    # "r_le":        [5e-5,   3e-4],   # 前缘半径 (米)
    # "r_te":        [5e-5,   3e-4],   # 后缘半径 (米)
    "x_tmax":      [ 0.30,   0.70],  # 最大厚度位置 x/c
    "front_chord": [ 0.30,   0.70],  # 前段弦长比
    "front_camber":[ 0.30,   0.70],  # 前段弯度比
}

MACH_CONDITIONS = [0.8, 0.825, 0.85]   # 多工况


def sample_params(n: int) -> list[BladeParams]:
    """拉丁超立方采样，返回 BladeParams 列表"""
    keys = list(PARAM_BOUNDS.keys())
    bounds = np.array(list(PARAM_BOUNDS.values()))

    sampler = qmc.LatinHypercube(d=len(keys), seed=42)
    raw = sampler.random(n=n)
    scaled = qmc.scale(raw, bounds[:, 0], bounds[:, 1])

    return [BladeParams(**{keys[i]: float(scaled[j, i])
                           for i in range(len(keys))})
            for j in range(n)]


def _run_single_task(args):
    """用于多进程的独立包装函数"""
    params, case_id, mach, root_dir, tpl_ta, tpl_ises = args
    pipe = CascadePipeline(root_dir=root_dir)
    return pipe.run_case(params, case_id, mach_in=mach, template_ta=tpl_ta, template_ises=tpl_ises)


def generate_dataset(n_samples: int = 500, root_dir: str = ".",
                     n_workers: int = 8, save_path: str = "dataset.csv") -> pd.DataFrame:
    template_ta = Path(root_dir) / "TA_template.DAT"
    template_ises = Path(root_dir) / "input" / "ises.dat"
    params_list = sample_params(n_samples)

    tasks = []
    case_id = 0
    for params in params_list:
        for mach in MACH_CONDITIONS:
            tasks.append((params, case_id, mach, root_dir, template_ta, template_ises))
            case_id += 1

    records = []
    failed = 0
    print(f"开始生成 {n_samples} 个样本，共 {len(tasks)} 个任务。启动 {n_workers} 个进程...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_task, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                records.append(result)
            else:
                failed += 1

            if (i + 1) % 200 == 0:
                print(f"  进度 {i + 1}/{len(tasks)} | 有效样本 {len(records)} | 失败 {failed}")

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"\n完成！有效样本：{len(df)} / {len(tasks)}")
    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=2000, root_dir=".")
    print(df.describe())
