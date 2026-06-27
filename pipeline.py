"""
二维叶栅自动化流水线
适配：Airfoil.exe + mises + LOST.dat 输出
"""
import subprocess, shutil, time, os, re
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class BladeParams:
    """对应 TA.DAT 中的可变几何参数"""
    beta1:      float   # 叶片进口气流角 (度)
    beta2:      float   # 叶片出口气流角 (度)
    # t_max:      float   # 叶片最大相对厚度
    # r_le:       float   # 前缘小圆半径 (米)
    # r_te:       float   # 后缘小圆半径 (米)
    x_tmax:     float   # 最大厚度相对位置
    front_chord:float   # 前段弦长比总弦长
    front_camber:float  # 前段弯度比总弯度

@dataclass
class AeroPerf:
    """对应 LOST.dat 输出"""
    loss:       float     # 总压损失系数
    pressure_rise: float  # 静压升系数
    converged:  bool      # 是否收敛


class CascadePipeline:
    def __init__(self, root_dir: str, work_base: str = "./runs"):
        self.root = Path(root_dir)       # 存放各exe的根目录
        self.work_base = Path(work_base)
        self.work_base.mkdir(parents=True, exist_ok=True)

        # 验证必要文件存在
        for exe in ["Airfoil.exe", "iset.exe", "winmises.exe", "iplot.exe"]:
            if not (self.root / exe).exists():
                raise FileNotFoundError(f"缺少 {exe}，请检查 root_dir 路径")

    # ------------------------------------------------------------------ #
    #  Step 1: 生成 TA.DAT                                                #
    # ------------------------------------------------------------------ #
    def write_ta_dat(self, params: BladeParams, case_dir: Path,
                     template_path: Path = None):
        """
        根据模板 TA.DAT 替换可变参数后写入 case_dir/TA.DAT
        如果没有模板则用内置默认值
        """
        # 读模板（建议保留一份你调好的基准叶型作为模板）
        if template_path and template_path.exists():
            lines = template_path.read_text().splitlines()
        else:
            lines = self._default_ta_dat_lines()

        # 根据文件格式，第2行包含主要气动参数
        # 格式：r_le  锥角  叶片数  弦长  beta1  beta2  t_max
        # # 只修改需要变化的字段，其余保持模板值
        parts = lines[1].split()
        parts[4] = f"{params.beta1:.6f}"
        parts[5] = f"{params.beta2:.6f}"
        # parts[6] = f"{params.t_max:.6f}"
        lines[1] = "   ".join(parts)

        # # 第3行：攻角 IDEV 落后角 点数 r_le r_te 落后角修正
        # parts3 = lines[2].split()
        # parts3[4] = f"{params.r_le:.6f}"
        # parts3[5] = f"{params.r_te:.6f}"
        # lines[2] = "   ".join(parts3)

        # 第16行：最大厚度位置
        parts16 = lines[15].split()
        parts16[0] = f"{params.x_tmax:.6f}"
        lines[15] = "   ".join(parts16)

        # 第20行：前段弦长/弯度比（多圆弧中线时使用）
        parts20 = lines[19].split()
        parts20[0] = f"{params.front_chord:.6f}"
        parts20[1] = f"{params.front_camber:.6f}"
        lines[19] = "   ".join(parts20)

        (case_dir / "TA.DAT").write_text("\n".join(lines))

    # ------------------------------------------------------------------ #
    #  Step 2: 运行 Airfoil.exe 生成叶型坐标                              #
    # ------------------------------------------------------------------ #
    def run_airfoil(self, case_dir: Path) -> bool:
        """调用 Airfoil.exe，输入 TA.DAT，生成 blade.DAT"""
        # 将 Airfoil.exe 复制到工作目录（或用绝对路径调用）
        shutil.copy(self.root / "Airfoil.exe", case_dir / "Airfoil.exe")

        result = subprocess.run(
            ["Airfoil.exe"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        blade_dat = case_dir / "blade.DAT"
        if not blade_dat.exists() or blade_dat.stat().st_size == 0:
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Step 3: 更新 ises.dat 中的马赫数（如需多工况）                     #
    # ------------------------------------------------------------------ #
    def write_ises_dat(self, case_dir: Path, mach_in: float,
                       beta_in: float, template_ises: Path = None):   # 有进口几何角月输入
    # def write_ises_dat(self, case_dir: Path, mach_in: float,
    #                    template_ises: Path = None):                     # 无进口几何角输入
        """从 input/ 文件夹的模板 ises.dat 复制并修改进口马赫数"""
        src = template_ises or (self.root / "input" / "ises.dat")
        target_file = case_dir / "ises.dat"

        lines = src.read_text().splitlines()

        # 修改第三行 (索引为 2)
        if len(lines) >= 3:
            parts = lines[2].split()
            # 确保行内至少有 3 个参数，避免索引越界
            if len(parts) >= 3:
                parts[0] = f"{mach_in:.6f}"  # 修改第一个参数为马赫数
                tan_beta1 = np.tan(np.deg2rad(beta_in))
                parts[2] = f"{tan_beta1:.6f}"  # 修改第三个参数为进口几何角
                lines[2] = "   ".join(parts)  # 使用多个空格重组，保持格式整洁

        # 将修改后的内容写入目标文件
        target_file.write_text("\n".join(lines) + "\n")

        # 同时更新 Machnumber.dat
        (case_dir / "Machnumber.dat").write_text(f"{mach_in:.6f}\n")

    # ------------------------------------------------------------------ #
    #  Step 4: 运行 MISES 三件套                                          #
    # ------------------------------------------------------------------ #
    def run_mises(self, case_dir: Path, timeout: int = 120) -> bool:
        """顺序运行 iset → winmises → iplot"""
        exes = ["iset.exe", "winmises.exe", "iplot.exe", "mises.exe"]
        for exe in exes:
            shutil.copy(self.root / exe, case_dir / exe)

        # 复制依赖文件
        for f in ["Bee.dll", "Bee.lib", "gridpar.dat", "LOST.dat"]:
            src = self.root / f
            if src.exists():
                shutil.copy(src, case_dir / f)

        # mises：流场求解
        r = subprocess.run(
            ["mises.exe"],
            cwd=case_dir,
            capture_output=True,
            text=True,  # 加这个才能读 stdout/stderr
            timeout=timeout
        )

        if r.returncode != 0:
            return False

        return (case_dir / "LOST.dat").exists()

    # ------------------------------------------------------------------ #
    #  Step 5: 解析 LOST.dat                                              #
    # ------------------------------------------------------------------ #
    def parse_lost(self, case_dir: Path) -> AeroPerf:
        lost_file = case_dir / "LOST.dat"
        if not lost_file.exists():
            return AeroPerf(loss=999.0, pressure_rise=0.0, converged=False)

        try:
            # 使用 skiprows 配合尝试机制，防止头部字符干扰
            # 尝试读取文件，如果包含非数字字符，则跳过首行再读
            try:
                data = np.loadtxt(lost_file)
            except ValueError:
                data = np.loadtxt(lost_file, skiprows=1)

            if data.size == 0:
                return AeroPerf(loss=999.0, pressure_rise=0.0, converged=False)

            # 确保 data 是二维数组（处理单行输出的情况）
            if data.ndim == 1:
                data = data.reshape(1, -1)

            loss = float(data[-1, 0])
            prise = float(data[-1, 1])

            # 增加对物理极端值的检查
            # 很多未收敛的案例会产出无穷大 (inf) 或 NaN
            if not np.isfinite(loss) or not np.isfinite(prise):
                return AeroPerf(loss=999.0, pressure_rise=0.0, converged=False)

            converged = 0.001 < loss < 0.5
            return AeroPerf(loss=loss, pressure_rise=prise, converged=converged)

        except Exception as e:
            # 调试：如果有报错，打印出来看看，防止被掩盖
            # print(f"DEBUG: 解析 LOST.dat 失败: {e}")
            return AeroPerf(loss=999.0, pressure_rise=0.0, converged=False)

    # ------------------------------------------------------------------ #
    #  主接口：运行单个案例                                               #
    # ------------------------------------------------------------------ #
    def run_case(self, params: BladeParams, case_id: int,
                 mach_in: float = 0.8,
                 template_ta: Path = None,
                 template_ises: Path = None) -> dict | None:

        case_dir = self.work_base / f"case_{case_id:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.write_ta_dat(params, case_dir, template_ta)
            if not self.run_airfoil(case_dir):
                print(f"Case {case_id}: 叶型生成 失败")
                return None
            self.write_ises_dat(case_dir, mach_in, params.beta1, template_ises)
            # self.write_ises_dat(case_dir, mach_in, template_ises)
            if not self.run_mises(case_dir):
                print(f"Case {case_id}: MISES 失败")
                return None
            perf = self.parse_lost(case_dir)
            if not perf.converged:
                print(f"Case {case_id}: 收敛 失败")
                return None

            return {**asdict(params), "mach_in": mach_in, **asdict(perf)}

        except Exception as e:
            # 引入 traceback 模块来打印详细的堆栈跟踪
            import traceback
            print(f"--- Case {case_id} 发生错误 ---")
            traceback.print_exc()  # 打印出报错的具体文件名、行号和错误类型
            return None

    # ------------------------------------------------------------------ #
    #  工具函数                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_line_idx(lines: list, keyword: str) -> int:
        """找到注释行之后的数据行索引"""
        for i, line in enumerate(lines):
            if keyword in line and i + 1 < len(lines):
                return i + 1
        return -1

    def _default_ta_dat_lines(self) -> list:
        """你的基准叶型 TA.DAT 内容，作为兜底模板"""
        # 直接把你调好的基准 TA.DAT 内容粘贴在这里
        # 或者在初始化时从文件读取
        raise FileNotFoundError(
            "请提供 template_ta 路径，或将基准 TA.DAT 放在 root_dir 下"
        )
