"""
================================================================================
SB-BDI  ·  HE HO TRO RA QUYET DINH PHAN TICH RAO CAN   (file don - app.py)
Secretary Bird, Barrier Dependency Interpretation
--------------------------------------------------------------------------------
Framework tich hop BWM + DEMATEL + ISM voi nguong cat alpha noi sinh, duoc toi uu
bang Secretary Bird Optimization Algorithm (SBOA) tren chi so CSI.

Tac gia framework : Ton Nguyen Trong Hien
ORCID             : https://orcid.org/0000-0002-6970-0799

Chay:  streamlit run app.py
Can:   streamlit, numpy, pandas, scipy, plotly
       (openpyxl chi can neu muon TAI LEN file .xlsx; khong bat buoc de chay app)
================================================================================

MUC LUC
  PHAN A. ENGINE THUAT TOAN
    A0. Hang so                      A5. SBOA, Secretary Bird Optimization
    A1. BWM (Linear Best-Worst)      A6. Pipeline tong
    A2. DEMATEL                      A7. Du lieu mau (12 rao can DBSCL)
    A3. ISM va MICMAC                A8. Sinh tinh huong demo ngau nhien
    A4. CSI, Causal Structure Index  A9. Phan tich ket qua tu dong
  PHAN B. TRUC QUAN HOA
    B1. Ban do nhan qua DEMATEL      B5. Heatmap va bieu do cot
    B2. So do phan tang ISM (DOT)    B6. So do quy trinh framework
    B3. MICMAC                       B7. So do quy trinh rut gon
    B4. Landscape CSI va hoi tu
  PHAN C. GIAO DIEN STREAMLIT
    Tab 0. Gioi thieu      Tab 3. DEMATEL
    Tab 1. Cau hinh        Tab 4. Toi uu nguong alpha
    Tab 2. BWM             Tab 5. Ket qua va dien giai
================================================================================
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog


# ##############################################################################
# ##############################################################################
##                     PHAN A.  ENGINE THUAT TOAN SB-BDI                      ##
# ##############################################################################
# ##############################################################################

# ==============================================================================
# 0. HANG SO
# ==============================================================================

# Bang chi so nhat quan CI cua Rezaei (2015) theo gia tri a_BW
CI_TABLE: Dict[int, float] = {
    1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30,
    6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23,
}

# Nguong chap nhan CR
CR_THRESHOLD = 0.10

EPS = 1e-12


# ==============================================================================
# 1. BWM - BEST WORST METHOD (Linear BWM, Rezaei 2016)
# ==============================================================================

def solve_bwm_linear(
    bo: Sequence[float],
    wo: Sequence[float],
    best_idx: Optional[int] = None,
    worst_idx: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Giai Linear BWM bang quy hoach tuyen tinh.

        min  xi
        s.t. | w_B - a_Bj * w_j | <= xi     for all j != B
             | w_j - a_jW * w_W | <= xi     for all j != W
             sum(w) = 1,  w >= 0

    Bien quyet dinh: x = [w_1, ..., w_n, xi]  (n+1 bien)

    Tra ve: (w, xi*)
    """
    bo = np.asarray(bo, dtype=float)
    wo = np.asarray(wo, dtype=float)
    n = len(bo)

    B = int(np.argmin(bo)) if best_idx is None else int(best_idx)
    W = int(np.argmin(wo)) if worst_idx is None else int(worst_idx)

    nv = n + 1
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    # Rang buoc Best-to-Others
    for j in range(n):
        if j == B:
            continue
        a = bo[j]
        r = np.zeros(nv); r[B] = 1.0;  r[j] = -a; r[-1] = -1.0
        A_ub.append(r); b_ub.append(0.0)
        r = np.zeros(nv); r[B] = -1.0; r[j] = a;  r[-1] = -1.0
        A_ub.append(r); b_ub.append(0.0)

    # Rang buoc Others-to-Worst
    for j in range(n):
        if j == W:
            continue
        a = wo[j]
        r = np.zeros(nv); r[j] = 1.0;  r[W] = -a; r[-1] = -1.0
        A_ub.append(r); b_ub.append(0.0)
        r = np.zeros(nv); r[j] = -1.0; r[W] = a;  r[-1] = -1.0
        A_ub.append(r); b_ub.append(0.0)

    A_eq = np.zeros((1, nv)); A_eq[0, :n] = 1.0
    c = np.zeros(nv); c[-1] = 1.0

    res = linprog(
        c,
        A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=A_eq, b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n + [(0.0, None)],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP BWM that bai: {res.message}")

    w = np.asarray(res.x[:n], dtype=float)
    w = w / w.sum()
    return w, float(res.x[-1])


def consistency_ratio(bo: Sequence[float], wo: Sequence[float], xi: float,
                      best_idx: Optional[int] = None,
                      worst_idx: Optional[int] = None) -> Tuple[float, int]:
    """
    CR = xi* / CI(a_BW),  voi a_BW = max(BO[W], WO[B]).
    Tra ve (CR, a_BW).
    """
    bo = np.asarray(bo, dtype=float)
    wo = np.asarray(wo, dtype=float)
    B = int(np.argmin(bo)) if best_idx is None else int(best_idx)
    W = int(np.argmin(wo)) if worst_idx is None else int(worst_idx)

    a_bw = int(round(max(bo[W], wo[B])))
    a_bw = max(1, min(9, a_bw))
    ci = CI_TABLE.get(a_bw, 5.23)
    if ci <= 0:
        return 0.0, a_bw
    return float(xi / ci), a_bw


def aggregate_weights(W_matrix: np.ndarray, method: str = "geometric") -> np.ndarray:
    """
    Tong hop trong so cua nhieu chuyen gia.
      - "geometric": AIJ - trung binh nhan (Aczel & Saaty, 1983)  [mac dinh]
      - "arithmetic": trung binh cong
    """
    W_matrix = np.atleast_2d(np.asarray(W_matrix, dtype=float))
    if method == "arithmetic":
        w = W_matrix.mean(axis=0)
    else:
        w = np.exp(np.mean(np.log(W_matrix + EPS), axis=0))
    return w / w.sum()


def run_bwm(BO: np.ndarray, WO: np.ndarray,
            best_idx: Optional[int] = None,
            worst_idx: Optional[int] = None,
            agg: str = "geometric") -> Dict:
    """
    Chay BWM cho toan bo panel chuyen gia.
    BO, WO: mang (n_experts, n_barriers).
    """
    BO = np.atleast_2d(np.asarray(BO, dtype=float))
    WO = np.atleast_2d(np.asarray(WO, dtype=float))
    k = BO.shape[0]

    w_all, xi_all, cr_all, abw_all = [], [], [], []
    for e in range(k):
        w, xi = solve_bwm_linear(BO[e], WO[e], best_idx, worst_idx)
        cr, abw = consistency_ratio(BO[e], WO[e], xi, best_idx, worst_idx)
        w_all.append(w); xi_all.append(xi); cr_all.append(cr); abw_all.append(abw)

    W_matrix = np.array(w_all)
    return {
        "W_experts": W_matrix,
        "xi": np.array(xi_all),
        "CR": np.array(cr_all),
        "a_BW": np.array(abw_all),
        "weights": aggregate_weights(W_matrix, agg),
        "all_consistent": bool(np.all(np.array(cr_all) < CR_THRESHOLD)),
    }


# ==============================================================================
# 2. DEMATEL
# ==============================================================================

def dematel_pipeline(Z: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict:
    """
    DEMATEL co tich hop trong so BWM.

        s      = max( max_i sum_j z_ij , max_j sum_i z_ij )
        N      = Z / s
        T_unw  = N (I - N)^-1
        T[i,j] = T_unw[i,j] * (n * w_j)     <-- trong so theo COT (barrier nhan anh huong)

    He so n giu T o cung thang do voi DEMATEL khong trong so (vi sum(w)=1).
    """
    Z = np.asarray(Z, dtype=float).copy()
    n = Z.shape[0]
    np.fill_diagonal(Z, 0.0)

    s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max())
    if s <= 0:
        raise ValueError("Ma tran quan he truc tiep Z rong (tong = 0).")

    N = Z / s

    # Truong hop suy bien: neu MOI hang deu co tong = s (vi du ma tran dong nhat),
    # thi (I - N) suy bien. Thu nho N mot luong khong dang ke de bao dam kha nghich.
    rho = float(np.max(np.abs(np.linalg.eigvals(N))))
    if rho >= 1.0 - 1e-9:
        N = N * ((1.0 - 1e-6) / rho)

    I = np.eye(n)
    T_unw = np.linalg.solve((I - N).T, N.T).T   # = N (I - N)^-1, on dinh hon inv()

    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    T = T_unw * (w[None, :] * n)

    R = T.sum(axis=1)          # anh huong PHAT ra (row sums)
    C = T.sum(axis=0)          # anh huong NHAN vao (column sums)

    return {
        "Z": Z, "s": float(s), "N": N,
        "T_unweighted": T_unw, "T": T,
        "R": R, "C": C,
        "prominence": R + C,       # Do noi bat (Importance)
        "relation": R - C,         # Quan he (Cause/Effect)
        "group": np.where(R - C > 0, "Nguyên nhân", "Hệ quả"),
    }


# ==============================================================================
# 3. ISM - INTERPRETIVE STRUCTURAL MODELING
# ==============================================================================

def reachability_matrix(T: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    binary = (T >= alpha), duong cheo = 1 (tinh phan xa)
    reach  = be bao bac cau Warshall cua binary
    Tra ve (binary, reach).
    """
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    binary = (T >= alpha).astype(int)
    np.fill_diagonal(binary, 1)

    reach = binary.copy()
    for k in range(n):
        for i in range(n):
            if reach[i, k]:
                reach[i] = np.maximum(reach[i], reach[k])
    return binary, reach


def n_arrows(T: np.ndarray, alpha: float) -> int:
    """So mui ten truc tiep (khong tinh duong cheo) tai nguong alpha."""
    n = T.shape[0]
    return int(((np.asarray(T) >= alpha) & (~np.eye(n, dtype=bool))).sum())


def level_partition(reach: np.ndarray) -> Tuple[int, np.ndarray, Dict[int, List[int]]]:
    """
    Phan tang Warfield (1974): nut i thuoc tang hien tai khi R(i) subset A(i).

    QUY UOC: Tang 1 = DINH = ket qua/trieu chung (bi phu thuoc nhieu nhat).
             Tang L = DAY  = nguyen nhan goc (root causes).

    Tra ve (so_tang, vector_tang_theo_nut[1..L], dict {tang: [chi so nut]}).
    """
    n = reach.shape[0]
    remaining = set(range(n))
    levels: Dict[int, List[int]] = {}
    part = np.zeros(n, dtype=int)
    lvl = 1

    while remaining:
        current = []
        for i in remaining:
            R_i = {j for j in remaining if reach[i, j]}
            A_i = {j for j in remaining if reach[j, i]}
            if R_i.issubset(A_i):
                current.append(i)
        if not current:  # chu trinh -> gom phan con lai vao 1 tang
            current = sorted(remaining)
        levels[lvl] = sorted(current)
        for nd in current:
            part[nd] = lvl
            remaining.discard(nd)
        lvl += 1

    return len(levels), part, levels


def micmac(reach: np.ndarray, split_mode: str = "adaptive") -> Dict:
    """
    Phan tich MICMAC tren ma tran kha dat cuoi cung.
      driving power = tong hang, dependence = tong cot.

    split_mode:
      - "classic"  : chia doi tai n/2 tren ca hai truc (quy uoc co dien).
      - "adaptive" : chia tai trung diem dai gia tri quan sat cua tung truc.
        Do SB-BDI chon alpha kha cao (do thi thua), nguong n/2 thuong khien
        toan bo nut roi vao o "Tu tri" va mat het thong tin -> mac dinh adaptive.
    """
    n = reach.shape[0]
    dp = reach.sum(axis=1)
    dep = reach.sum(axis=0)

    if split_mode == "classic":
        sx = sy = n / 2.0
    else:
        sy = (float(dp.min()) + float(dp.max())) / 2.0     # truc sức dẫn dắt
        sx = (float(dep.min()) + float(dep.max())) / 2.0   # truc phụ thuộc

    cls = []
    for d, p in zip(dp, dep):
        if d > sy and p > sx:
            cls.append("Liên kết (Linkage)")
        elif d > sy and p <= sx:
            cls.append("Độc lập / Dẫn dắt (Independent)")
        elif d <= sy and p > sx:
            cls.append("Phụ thuộc (Dependent)")
        else:
            cls.append("Tự trị (Autonomous)")
    return {"driving_power": dp, "dependence": dep, "classification": cls,
            "split_x": sx, "split_y": sy, "split": sy, "mode": split_mode}


def run_ism(T: np.ndarray, alpha: float, split_mode: str = "adaptive") -> Dict:
    binary, reach = reachability_matrix(T, alpha)
    L, part, levels = level_partition(reach)
    mm = micmac(reach, split_mode)
    return {
        "alpha": float(alpha),
        "binary": binary, "reach": reach,
        "n_levels": L, "partition": part, "levels": levels,
        "n_arrows": n_arrows(T, alpha),
        "micmac": mm,
    }


# ==============================================================================
# 4. CSI - CAUSAL STRUCTURE INDEX (ham muc tieu noi sinh)
# ==============================================================================

@dataclass
class CSIContext:
    """
    Ngu canh chuan hoa cho CSI. S duoc "dong bang" tu T CHUA cat nguong
    (tranh phu thuoc vong tron vao alpha).

        CS(alpha) = S / N_arrows(alpha)
        L(alpha)  = so tang ISM
        CSI       = sqrt( norm(CS) * norm(L) )
    """
    T: np.ndarray
    S: float
    cs_min: float
    cs_max: float
    l_min: float
    l_max: float
    alpha_low: float
    alpha_high: float
    n: int
    candidates: np.ndarray = field(default_factory=lambda: np.array([]))


def build_csi_context(T: np.ndarray) -> CSIContext:
    T = np.asarray(T, dtype=float)
    n = T.shape[0]

    R = T.sum(axis=1); C = T.sum(axis=0)
    rel = R - C
    pos = abs(float(np.mean(rel[rel > 0]))) if np.any(rel > 0) else 0.0
    neg = abs(float(np.mean(rel[rel < 0]))) if np.any(rel < 0) else 0.0
    S = pos + neg

    off = T[~np.eye(n, dtype=bool)]
    off_pos = off[off > 0]
    if off_pos.size == 0:
        raise ValueError("Ma tran T khong co quan he duong nao ngoai duong cheo.")

    lo, hi = float(off_pos.min()), float(off_pos.max())
    cands = np.unique(off_pos)   # CSI la ham bac thang theo alpha -> quet chinh xac

    cs_vals, l_vals = [], []
    for a in cands:
        na = n_arrows(T, a)
        if na < n:                       # rang buoc kha thi: N_arrows >= n
            continue
        _, reach = reachability_matrix(T, a)
        L, _, _ = level_partition(reach)
        cs_vals.append(S / (na + EPS))
        l_vals.append(L)

    if not cs_vals:                      # khong co alpha kha thi -> noi long
        for a in cands:
            na = n_arrows(T, a)
            if na < 1:
                continue
            _, reach = reachability_matrix(T, a)
            L, _, _ = level_partition(reach)
            cs_vals.append(S / (na + EPS)); l_vals.append(L)

    return CSIContext(
        T=T, S=S,
        cs_min=float(min(cs_vals)), cs_max=float(max(cs_vals)),
        l_min=float(min(l_vals)), l_max=float(max(l_vals)),
        alpha_low=lo, alpha_high=hi, n=n, candidates=cands,
    )


def evaluate_csi(alpha: float, ctx: CSIContext) -> float:
    """CSI(alpha) = sqrt( CS_norm * L_norm ). Tra ve -1e9 neu khong kha thi."""
    na = n_arrows(ctx.T, alpha)
    if na < ctx.n:
        return -1e9
    _, reach = reachability_matrix(ctx.T, alpha)
    L, _, _ = level_partition(reach)

    cs = ctx.S / (na + EPS)

    def norm(v: float, lo: float, hi: float) -> float:
        # Neu dai gia tri suy bien (moi alpha cho cung ket qua) thi tieu chi do
        # khong con kha nang phan biet -> tra ve 1.0 thay vi 0 de khong triet tieu CSI.
        if hi - lo < 1e-12:
            return 1.0
        return min(1.0, max(0.0, (v - lo) / (hi - lo)))

    cs_t = norm(cs, ctx.cs_min, ctx.cs_max)
    l_t = norm(float(L), ctx.l_min, ctx.l_max)
    return float(math.sqrt(cs_t * l_t))


def csi_profile(ctx: CSIContext, n_points: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """Duong bieu dien CSI(alpha) de ve landscape."""
    grid = np.linspace(ctx.alpha_low, ctx.alpha_high, n_points)
    vals = np.array([evaluate_csi(float(a), ctx) for a in grid])
    vals = np.where(vals < -1e8, np.nan, vals)
    return grid, vals


# ==============================================================================
# 5. SBOA - SECRETARY BIRD OPTIMIZATION ALGORITHM (Fu et al., 2024)
# ==============================================================================

def levy_flight(beta: float = 1.5) -> float:
    sigma_u = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    return float(u / (abs(v) ** (1 / beta) + EPS))


def secretary_bird_optimization(
    ctx: CSIContext,
    iterations: int = 80,
    pop_size: int = 25,
    seed: Optional[int] = 42,
    progress_cb=None,
) -> Dict:
    """
    Toi uu alpha bang SBOA (bai toan 1 chieu, maximize CSI).

    Giai doan 1 - SAN MOI (3 pha theo tien do t/T):
      Pha 1 (t < T/3)      Tim moi   : X_new = X_i + (X_r1 - X_r2) * R1
      Pha 2 (T/3<=t<2T/3)  An moi    : X_new = X_best + exp(-(t/T)^4)*(RB-0.5)*(X_best-X_i)
      Pha 3 (t >= 2T/3)    Tan cong  : X_new = X_best + (1-t/T)^(2t/T) * X_i * Levy

    Giai doan 2 - TRON KE THU (2 chien luoc, chon ngau nhien):
      C1 Nguy trang : X_new = X_best + (2R-1)*(1-t/T)*X_i
      C2 Bo chay    : X_new = X_i + R2*(X_rand - K*X_i),  K in {1,2}
    """
    if seed is not None:
        np.random.seed(seed)

    low, high = ctx.alpha_low, ctx.alpha_high
    pop = np.random.uniform(low, high, pop_size)
    fit = np.array([evaluate_csi(float(a), ctx) for a in pop])

    bi = int(np.argmax(fit))
    best, best_f = float(pop[bi]), float(fit[bi])
    history = [best_f]

    for t in range(1, iterations + 1):
        # ---- Giai doan 1: San moi ----
        for i in range(pop_size):
            if t < iterations / 3.0:
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                new = pop[i] + (pop[r1] - pop[r2]) * np.random.rand()
            elif t < 2 * iterations / 3.0:
                RB = np.random.normal(0, 1)
                new = best + math.exp(-((t / iterations) ** 4)) * (RB - 0.5) * (best - pop[i])
            else:
                new = best + ((1 - t / iterations) ** (2 * t / iterations)) * pop[i] * levy_flight()
            new = float(np.clip(new, low, high))
            f = evaluate_csi(new, ctx)
            if f > fit[i]:
                pop[i], fit[i] = new, f
                if f > best_f:
                    best, best_f = new, f

        # ---- Giai doan 2: Tron ke thu ----
        for i in range(pop_size):
            if np.random.rand() < 0.5:
                R = np.random.rand()
                new = best + (2 * R - 1) * (1 - t / iterations) * pop[i]
            else:
                K = np.round(1 + np.random.rand())
                rr = int(np.random.choice(pop_size))
                new = pop[i] + np.random.normal(0, 1) * (pop[rr] - K * pop[i])
            new = float(np.clip(new, low, high))
            f = evaluate_csi(new, ctx)
            if f > fit[i]:
                pop[i], fit[i] = new, f
                if f > best_f:
                    best, best_f = new, f

        history.append(best_f)
        if progress_cb is not None:
            progress_cb(t / iterations)

    # Tinh chinh: CSI la ham bac thang -> chieu nghiem ve ung vien chinh xac gan nhat
    refined, refined_f = best, best_f
    for a in ctx.candidates:
        f = evaluate_csi(float(a), ctx)
        if f > refined_f + 1e-9:
            refined, refined_f = float(a), f
    if refined_f > best_f:
        best, best_f = refined, refined_f

    return {
        "alpha": float(best),
        "CSI": float(best_f),
        "history": history,
        "n_arrows": n_arrows(ctx.T, best),
        "CS": ctx.S / (n_arrows(ctx.T, best) + EPS),
    }


# ==============================================================================
# 6. PIPELINE TONG
# ==============================================================================

def run_full_pipeline(Z: np.ndarray, weights: np.ndarray,
                      iterations: int = 80, pop_size: int = 25,
                      seed: Optional[int] = 42,
                      alpha_override: Optional[float] = None,
                      progress_cb=None) -> Dict:
    dm = dematel_pipeline(Z, weights)
    ctx = build_csi_context(dm["T"])

    if alpha_override is not None:
        alpha = float(alpha_override)
        sb = {
            "alpha": alpha, "CSI": evaluate_csi(alpha, ctx),
            "history": [], "n_arrows": n_arrows(dm["T"], alpha),
            "CS": ctx.S / (n_arrows(dm["T"], alpha) + EPS),
        }
    else:
        sb = secretary_bird_optimization(ctx, iterations, pop_size, seed, progress_cb)

    ism = run_ism(dm["T"], sb["alpha"])
    return {"dematel": dm, "ctx": ctx, "sbo": sb, "ism": ism}


# ==============================================================================
# 7. DU LIEU MAU - Case study: Rao can lua huu co DBSCL (12 barriers)
# ==============================================================================

DEMO_CODES = ["EC1", "SO1", "SO2", "SO3", "IN1", "IN2",
              "IN3", "IN4", "IN5", "TE1", "TE2", "TE3"]

DEMO_NAMES = [
    "Chi phí đầu tư ban đầu cao",
    "Nhu cầu tiêu dùng thấp",
    "Nhận thức hạn chế",
    "Cảm nhận kém hiệu quả",
    "Chính sách chưa phù hợp",
    "Thủ tục chứng nhận phức tạp",
    "Kiểm soát thị trường yếu",
    "Thiếu doanh nghiệp dẫn dắt",
    "Quy hoạch vùng manh mún",
    "Vận hành phức tạp",
    "Thói quen độc canh",
    "Thiếu vật tư chuyên dụng",
]

DEMO_DIMS = ["Kinh tế", "Xã hội", "Xã hội", "Xã hội",
             "Thể chế", "Thể chế", "Thể chế", "Thể chế", "Thể chế",
             "Kỹ thuật", "Kỹ thuật", "Kỹ thuật"]

# Ma tran quan he truc tiep Z (trung binh 10 chuyen gia, thang 0-4)
DEMO_Z = np.array([
    [0.0, 1.2, 0.8, 1.5, 0.5, 1.1, 1.4, 0.7, 0.6, 2.1, 0.9, 1.3],
    [1.1, 0.0, 1.4, 1.8, 0.4, 0.6, 2.5, 1.3, 0.7, 0.8, 0.5, 0.6],
    [0.7, 2.1, 0.0, 2.4, 0.3, 0.5, 1.8, 0.9, 0.4, 0.6, 1.1, 0.5],
    [1.4, 1.7, 1.2, 0.0, 0.2, 0.4, 1.6, 0.8, 0.5, 1.9, 1.5, 1.2],
    [3.2, 2.5, 2.1, 1.9, 0.0, 3.4, 3.6, 2.8, 3.5, 2.1, 1.8, 2.4],
    [1.8, 1.1, 0.9, 0.7, 0.6, 0.0, 2.1, 1.5, 1.2, 2.4, 0.8, 1.6],
    [2.5, 3.4, 2.2, 2.8, 0.4, 1.2, 0.0, 2.6, 1.5, 1.4, 0.9, 1.1],
    [1.9, 2.2, 1.4, 1.6, 0.5, 1.8, 2.9, 0.0, 2.1, 1.3, 0.7, 1.2],
    [1.5, 1.3, 0.8, 1.1, 0.4, 1.2, 2.4, 1.9, 0.0, 2.6, 1.4, 1.8],
    [2.1, 0.8, 0.6, 1.9, 0.3, 0.7, 1.2, 1.1, 0.9, 0.0, 1.6, 2.5],
    [1.3, 0.6, 0.5, 1.4, 0.2, 0.4, 0.8, 0.7, 1.1, 2.4, 0.0, 2.1],
    [1.6, 0.7, 0.5, 1.3, 0.3, 0.6, 0.9, 1.1, 0.8, 2.7, 1.8, 0.0],
])

# Best = IN3 (index 6), Worst = SO2 (index 2)
DEMO_BEST, DEMO_WORST = 6, 2

DEMO_BO = np.array([
    [7, 6, 9, 4, 5, 5, 1, 4, 3, 3, 6, 4],
    [5, 6, 9, 4, 4, 5, 1, 5, 3, 2, 4, 3],
    [6, 7, 8, 3, 5, 5, 1, 4, 3, 4, 5, 4],
    [7, 6, 9, 5, 3, 5, 1, 5, 2, 3, 6, 4],
    [5, 5, 9, 4, 4, 6, 1, 3, 4, 3, 5, 5],
    [7, 5, 9, 4, 4, 5, 1, 4, 3, 2, 6, 5],
    [6, 7, 8, 3, 5, 4, 1, 5, 2, 3, 5, 4],
    [7, 5, 9, 4, 4, 6, 1, 4, 4, 4, 5, 5],
    [6, 6, 9, 4, 3, 5, 1, 4, 3, 3, 6, 4],
    [7, 6, 9, 4, 5, 5, 1, 4, 3, 3, 5, 4],
], dtype=float)

DEMO_WO = np.array([
    [2, 3, 1, 5, 4, 4, 9, 5, 6, 6, 3, 5],
    [4, 3, 1, 5, 5, 4, 9, 4, 6, 7, 5, 6],
    [3, 2, 1, 4, 4, 4, 8, 5, 6, 5, 4, 5],
    [2, 3, 1, 5, 6, 4, 9, 4, 7, 6, 3, 5],
    [4, 3, 1, 5, 4, 3, 9, 6, 5, 6, 4, 4],
    [2, 4, 1, 5, 5, 4, 9, 5, 6, 7, 3, 4],
    [3, 2, 1, 6, 4, 5, 8, 4, 7, 6, 4, 5],
    [2, 4, 1, 5, 5, 3, 9, 5, 5, 5, 4, 6],
    [3, 3, 1, 5, 6, 4, 9, 5, 6, 6, 3, 5],
    [2, 3, 1, 4, 5, 4, 9, 5, 6, 6, 4, 4],
], dtype=float)

# Trong so BWM da cong bo (Bang 3 cua bai bao)
DEMO_WEIGHTS = np.array([0.062, 0.067, 0.022, 0.082, 0.078, 0.073,
                         0.212, 0.076, 0.085, 0.093, 0.071, 0.079])


# ==============================================================================
# 8. SINH TINH HUONG DEMO NGAU NHIEN
# ==============================================================================

DEMO_SCENARIOS: List[Dict] = [
    {
        "title": "Rào cản chuyển đổi số của doanh nghiệp sản xuất",
        "barriers": [
            ("TC1", "Ngân sách đầu tư công nghệ hạn chế", "Tài chính"),
            ("TC2", "Khó chứng minh hiệu quả đầu tư", "Tài chính"),
            ("NL1", "Thiếu nhân sự có kỹ năng số", "Nhân lực"),
            ("NL2", "Nhân viên phản ứng với thay đổi", "Nhân lực"),
            ("NL3", "Lãnh đạo thiếu cam kết dài hạn", "Nhân lực"),
            ("QT1", "Quy trình nội bộ chưa chuẩn hoá", "Quy trình"),
            ("QT2", "Dữ liệu phân mảnh giữa các phòng ban", "Quy trình"),
            ("QT3", "Thiếu lộ trình chuyển đổi rõ ràng", "Quy trình"),
            ("CN1", "Hệ thống cũ khó tích hợp", "Công nghệ"),
            ("CN2", "Lo ngại an toàn thông tin", "Công nghệ"),
            ("CN3", "Phụ thuộc nhà cung cấp bên ngoài", "Công nghệ"),
        ],
    },
    {
        "title": "Rào cản triển khai ESG trong chuỗi cung ứng",
        "barriers": [
            ("CP1", "Chi phí tuân thủ tăng cao", "Chi phí"),
            ("CP2", "Khó chuyển chi phí sang giá bán", "Chi phí"),
            ("TT1", "Khách hàng chưa sẵn sàng trả thêm", "Thị trường"),
            ("TT2", "Thiếu áp lực từ đối thủ cạnh tranh", "Thị trường"),
            ("CS1", "Quy định pháp lý chưa rõ ràng", "Chính sách"),
            ("CS2", "Thiếu ưu đãi cho doanh nghiệp tiên phong", "Chính sách"),
            ("CS3", "Tiêu chuẩn báo cáo không thống nhất", "Chính sách"),
            ("NCU1", "Nhà cung cấp cấp 2 không đáp ứng", "Nhà cung ứng"),
            ("NCU2", "Khó truy xuất nguồn gốc nguyên liệu", "Nhà cung ứng"),
            ("NL1", "Thiếu chuyên gia đánh giá nội bộ", "Nhân lực"),
        ],
    },
    {
        "title": "Rào cản mở rộng thị trường xuất khẩu",
        "barriers": [
            ("TC1", "Vốn lưu động không đủ cho đơn hàng lớn", "Tài chính"),
            ("TC2", "Rủi ro tỷ giá và thanh toán quốc tế", "Tài chính"),
            ("SP1", "Chất lượng sản phẩm chưa ổn định", "Sản phẩm"),
            ("SP2", "Bao bì chưa đạt chuẩn thị trường đích", "Sản phẩm"),
            ("SP3", "Năng lực sản xuất chưa đủ quy mô", "Sản phẩm"),
            ("TT1", "Thiếu kênh phân phối tại nước sở tại", "Thị trường"),
            ("TT2", "Thương hiệu chưa được nhận diện", "Thị trường"),
            ("PL1", "Hàng rào kỹ thuật và kiểm dịch", "Pháp lý"),
            ("PL2", "Thủ tục chứng nhận kéo dài", "Pháp lý"),
            ("NL1", "Thiếu nhân sự am hiểu thị trường quốc tế", "Nhân lực"),
            ("NL2", "Hạn chế năng lực ngoại ngữ và đàm phán", "Nhân lực"),
        ],
    },
    {
        "title": "Rào cản giữ chân nhân sự chất lượng cao",
        "barriers": [
            ("TN1", "Mức lương chưa cạnh tranh thị trường", "Thu nhập"),
            ("TN2", "Chính sách thưởng thiếu minh bạch", "Thu nhập"),
            ("PT1", "Lộ trình thăng tiến không rõ ràng", "Phát triển"),
            ("PT2", "Ít cơ hội đào tạo chuyên sâu", "Phát triển"),
            ("PT3", "Công việc lặp lại, thiếu thử thách", "Phát triển"),
            ("MT1", "Văn hoá nội bộ thiếu gắn kết", "Môi trường"),
            ("MT2", "Áp lực công việc kéo dài", "Môi trường"),
            ("QL1", "Quản lý trực tiếp thiếu kỹ năng dẫn dắt", "Quản lý"),
            ("QL2", "Phản hồi hiệu suất không kịp thời", "Quản lý"),
            ("QL3", "Phân công công việc chồng chéo", "Quản lý"),
        ],
    },
]


def random_case(seed: Optional[int] = None, n: Optional[int] = None) -> Dict:
    """
    Sinh mot tinh huong demo ngau nhien nhung CO CAU TRUC:
    gan cho moi rao can mot "do sau" tiem an, rao can sau hon anh huong
    manh len rao can nong hon. Nho vay ISM cho ra cau truc nhieu tang
    thay vi mot dam quan he ngau nhien khong doc duoc.
    """
    rng = np.random.default_rng(seed)

    sc = DEMO_SCENARIOS[int(rng.integers(len(DEMO_SCENARIOS)))]
    pool = list(sc["barriers"])
    n_max = len(pool)
    n = int(n) if n else int(rng.integers(min(8, n_max), n_max + 1))
    n = max(4, min(n, n_max))

    pick = sorted(rng.choice(n_max, size=n, replace=False))
    chosen = [pool[i] for i in pick]
    codes = [c for c, _, _ in chosen]
    names = [nm for _, nm, _ in chosen]
    dims = [d for _, _, d in chosen]

    # --- do sau tiem an: tang 0 la ngon (he qua), tang cao la goc ---
    n_layers = int(rng.integers(3, 5))
    depth = np.zeros(n, dtype=int)
    order = rng.permutation(n)
    depth[order[0]] = 0                       # it nhat mot he qua
    depth[order[1]] = n_layers - 1            # it nhat mot nguyen nhan goc
    for k in order[2:]:
        depth[k] = int(rng.integers(0, n_layers))

    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gap = depth[i] - depth[j]
            if gap > 0:                        # i sau hon j  ->  i thuc day j
                base = 3.5 - 0.7 * (gap - 1)
                Z[i, j] = base + rng.normal(0, 0.35)
            elif gap == 0:
                Z[i, j] = rng.uniform(0.3, 1.4)
            else:                              # phan hoi nguoc, yeu
                Z[i, j] = rng.uniform(0.0, 0.7)
    Z = np.clip(np.round(Z, 1), 0.0, 4.0)
    np.fill_diagonal(Z, 0.0)

    # --- trong so: rao can cang sau, cang de duoc danh gia quan trong ---
    alpha_dir = 1.0 + 1.6 * depth + rng.random(n)
    weights = rng.dirichlet(alpha_dir * 2.5)

    # --- sinh BO/WO nhat quan voi trong so de tab BWM cung co du lieu ---
    B = int(np.argmax(weights))
    W = int(np.argmin(weights))
    n_exp = int(rng.integers(4, 8))
    BO = np.ones((n_exp, n))
    WO = np.ones((n_exp, n))
    for e in range(n_exp):
        for j in range(n):
            BO[e, j] = np.clip(round(weights[B] / max(weights[j], 1e-6)
                                     + rng.normal(0, 0.45)), 1, 9)
            WO[e, j] = np.clip(round(weights[j] / max(weights[W], 1e-6)
                                     + rng.normal(0, 0.45)), 1, 9)
        BO[e, B] = 1.0
        WO[e, W] = 1.0

    return {
        "title": sc["title"], "codes": codes, "names": names, "dims": dims,
        "Z": Z, "weights": weights, "BO": BO, "WO": WO,
        "best_idx": B, "worst_idx": W,
    }


# ==============================================================================
# 9. PHAN TICH / NHIN NHAN KET QUA (sinh nhan dinh tu dong)
# ==============================================================================

def interpret_results(res: Dict, codes: Sequence[str], names: Sequence[str],
                      weights: np.ndarray) -> List[Dict]:
    """
    Doc ket qua va sinh cac nhan dinh bang ngon ngu quan ly.
    Moi phan tu: {"tone": good|warn|info|key, "title": str, "text": str}
    tone dung de to mau the tren giao dien.
    """
    dm, sb, ism = res["dematel"], res["sbo"], res["ism"]
    prom, rel = dm["prominence"], dm["relation"]
    levels, part, mm = ism["levels"], ism["partition"], ism["micmac"]
    n = len(codes)
    out: List[Dict] = []

    top_lvl = max(levels)
    roots = levels[top_lvl]
    top_imp = int(np.argmax(prom))
    top_w = int(np.argmax(weights))
    density = ism["n_arrows"] / max(n * (n - 1), 1)

    def nm(i):
        return f"{codes[i]} ({names[i]})"

    # 1. Do sau cau truc
    if ism["n_levels"] <= 1:
        out.append({"tone": "warn", "title": "Cấu trúc phẳng, chưa tách được lớp",
                    "text": "Toàn bộ rào cản nằm cùng một tầng, nghĩa là dữ liệu đầu vào "
                            "chưa đủ phân biệt để chỉ ra cái nào gây ra cái nào. Nên rà "
                            "lại ma trận ảnh hưởng: có thể chuyên gia chấm quá đều tay."})
    elif ism["n_levels"] == 2:
        out.append({"tone": "info", "title": "Cấu trúc hai tầng, phân tách còn nông",
                    "text": "Hệ thống chỉ tách được thành nhóm gây ra và nhóm chịu ảnh "
                            "hưởng, chưa thấy chuỗi truyền dẫn trung gian. Kết luận vẫn "
                            "dùng được nhưng khuyến nghị can thiệp sẽ kém chi tiết."})
    else:
        out.append({"tone": "good", "title": f"Cấu trúc {ism['n_levels']} tầng, đủ sâu để hành động",
                    "text": f"Hệ thống tách được thành {ism['n_levels']} tầng rõ ràng, cho phép "
                            "xác định thứ tự can thiệp thay vì xử lý dàn trải. Đây là mức "
                            "phân giải tốt cho việc lập kế hoạch theo giai đoạn."})

    # 2. So nguyen nhan goc
    if len(roots) == 1:
        out.append({"tone": "key", "title": "Chỉ có một điểm khởi phát duy nhất",
                    "text": f"{nm(roots[0])} là gốc của toàn bộ chuỗi rào cản. Đây là tin tốt "
                            "cho việc phân bổ nguồn lực: chỉ cần một mũi can thiệp tập trung "
                            "thay vì nhiều chương trình song song."})
    elif len(roots) <= 3:
        out.append({"tone": "key", "title": f"Có {len(roots)} điểm khởi phát song song",
                    "text": "Các rào cản " + ", ".join(codes[i] for i in roots) +
                            " cùng nằm ở tầng đáy và độc lập với nhau. Cần xử lý đồng thời; "
                            "giải quyết một cái mà bỏ các cái còn lại thì chuỗi vẫn tái diễn."})
    else:
        out.append({"tone": "warn", "title": f"Có tới {len(roots)} nguyên nhân gốc, nguồn lực sẽ bị phân tán",
                    "text": "Số điểm khởi phát nhiều cho thấy hệ thống rào cản chưa hội tụ. "
                            "Nên chia thành nhiều giai đoạn và ưu tiên các gốc có trọng số cao nhất "
                            "trước, thay vì cố xử lý tất cả cùng lúc."})

    # 3. Quan trong nhat vs nguyen nhan goc
    if top_imp in roots:
        out.append({"tone": "good", "title": "Rào cản nổi bật nhất cũng chính là gốc",
                    "text": f"{nm(top_imp)} vừa có mức độ liên quan cao nhất trong hệ thống vừa "
                            "nằm ở tầng đáy. Ưu tiên đầu tư vào đây là lựa chọn an toàn và "
                            "được cả hai góc phân tích ủng hộ."})
    else:
        out.append({"tone": "key", "title": "Cái nổi bật nhất KHÔNG phải cái cần xử lý trước",
                    "text": f"{nm(top_imp)} có mức độ liên quan cao nhất nhưng chỉ nằm ở tầng "
                            f"{part[top_imp]}, tức phần lớn là hệ quả. Nếu dồn nguồn lực vào đây "
                            f"sẽ tốn kém mà vấn đề tái diễn, vì gốc thật sự là "
                            f"{', '.join(nm(i) for i in roots)}. Đây thường là điểm mà xếp hạng "
                            "thông thường bỏ sót."})

    # 4. Trong so cao nhat nam o dau
    if top_w not in roots and part[top_w] < top_lvl:
        out.append({"tone": "warn", "title": "Đánh giá của chuyên gia lệch khỏi cấu trúc thật",
                    "text": f"Chuyên gia cho {nm(top_w)} trọng số cao nhất "
                            f"({weights[top_w]:.3f}) nhưng rào cản này nằm ở tầng {part[top_w]}, "
                            "không phải tầng gốc. Nên xem đây là cảnh báo: cảm nhận về mức độ "
                            "quan trọng đang bám vào triệu chứng dễ thấy hơn là căn nguyên."})

    # 5. Mat do quan he
    if density > 0.45:
        out.append({"tone": "warn", "title": "Mạng lưới quan hệ khá dày",
                    "text": f"Giữ lại {ism['n_arrows']} quan hệ trên tổng {n*(n-1)} khả năng "
                            f"({density*100:.0f}%). Mọi thứ liên quan tới mọi thứ, nên sơ đồ "
                            "sẽ khó dùng để thuyết trình. Cân nhắc thu hẹp phạm vi phân tích."})
    elif density < 0.08:
        out.append({"tone": "info", "title": "Mạng lưới quan hệ thưa, kết luận rất tập trung",
                    "text": f"Chỉ {ism['n_arrows']} quan hệ được giữ lại. Bức tranh gọn và dễ "
                            "truyền đạt, nhưng cần kiểm tra xem có rào cản nào bị tách rời "
                            "khỏi hệ thống hay không."})
    else:
        out.append({"tone": "good", "title": "Mật độ quan hệ ở mức dễ diễn giải",
                    "text": f"{ism['n_arrows']} quan hệ được giữ lại trên tổng {n*(n-1)} khả năng. "
                            "Đủ để thấy chuỗi nhân quả mà không rối, phù hợp để đưa vào báo cáo."})

    # 6. Nhom lien ket bat on
    linkage = [i for i in range(n) if mm["classification"][i].startswith("Liên kết")]
    if linkage:
        out.append({"tone": "warn", "title": f"{len(linkage)} rào cản thuộc nhóm bất ổn",
                    "text": "Các rào cản " + ", ".join(codes[i] for i in linkage) +
                            " vừa tác động mạnh vừa chịu tác động mạnh. Mọi thay đổi ở đây "
                            "đều dội ngược lại hệ thống, nên cần thí điểm quy mô nhỏ trước "
                            "khi triển khai rộng."})

    # 7. Rao can co the loai khoi pham vi
    isolated = [i for i in range(n)
                if mm["driving_power"][i] <= 1 and mm["dependence"][i] <= 1]
    if isolated:
        out.append({"tone": "info", "title": "Có rào cản gần như tách rời hệ thống",
                    "text": "Các rào cản " + ", ".join(codes[i] for i in isolated) +
                            " gần như không nối với phần còn lại. Có thể xử lý độc lập bằng "
                            "biện pháp riêng, hoặc đưa ra khỏi phạm vi phân tích để tập trung "
                            "nguồn lực."})

    # 8. Can bang nguyen nhan / he qua
    n_cause = int((rel > 0).sum())
    if n_cause == 0 or n_cause == n:
        out.append({"tone": "warn", "title": "Không tách được nhóm nguyên nhân và hệ quả",
                    "text": "Toàn bộ rào cản rơi về cùng một phía. Thường do ma trận ảnh hưởng "
                            "được chấm quá đối xứng. Nên phỏng vấn lại chuyên gia với câu hỏi "
                            "rõ hơn về chiều tác động."})

    # 9. Do tin cay cua nghiem
    if sb["CSI"] >= 0.85:
        out.append({"tone": "good", "title": "Điểm cắt tối ưu rất rõ ràng",
                    "text": f"Chỉ số chất lượng cấu trúc đạt {sb['CSI']:.3f} trên thang 1. "
                            "Thuật toán tìm được một điểm cắt nổi trội hẳn so với các lựa chọn "
                            "khác, nghĩa là cấu trúc thu được ổn định, không phải kết quả may rủi."})
    elif sb["CSI"] >= 0.6:
        out.append({"tone": "info", "title": "Điểm cắt tối ưu ở mức chấp nhận được",
                    "text": f"Chỉ số chất lượng cấu trúc đạt {sb['CSI']:.3f}. Kết quả dùng được, "
                            "nhưng nên thử thay đổi nhẹ dữ liệu đầu vào để xem cấu trúc có giữ "
                            "nguyên hay không trước khi ra quyết định lớn."})
    else:
        out.append({"tone": "warn", "title": "Điểm cắt tối ưu chưa nổi trội",
                    "text": f"Chỉ số chất lượng cấu trúc chỉ đạt {sb['CSI']:.3f}. Dữ liệu đầu vào "
                            "chưa cho phép tách bạch rõ cấu trúc. Nên bổ sung chuyên gia hoặc "
                            "làm rõ định nghĩa từng rào cản rồi chạy lại."})

    return out

# ##############################################################################
# ##############################################################################
##                           PHAN B.  TRUC QUAN HOA                           ##
# ##############################################################################
# ##############################################################################

# ---- Bang mau ----------------------------------------------------------------
C_CAUSE = "#2E86AB"     # xanh - nguyen nhan
C_EFFECT = "#C73E1D"    # do  - ket qua
C_GRID = "rgba(140,140,140,0.25)"
LEVEL_COLORS = ["#C73E1D", "#E8730C", "#B5179E", "#3A86FF",
                "#0EAD69", "#7209B7", "#F4A261", "#264653"]

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13),
    margin=dict(l=60, r=40, t=70, b=60),
    hoverlabel=dict(font_size=13),
)


# ==============================================================================
# 1. CAUSAL MAP DEMATEL  (Prominence R+C  vs  Relation R-C)
# ==============================================================================

def causal_map(codes: Sequence[str], names: Sequence[str],
               prominence: np.ndarray, relation: np.ndarray,
               T: np.ndarray, alpha: float,
               show_arrows: bool = True) -> go.Figure:
    """Bản đồ nhân quả DEMATEL: trục X = độ nổi bật (R+C), trục Y = quan hệ (R-C).
    Mũi tên = quan hệ ảnh hưởng vượt ngưỡng alpha."""
    n = len(codes)
    x, y = np.asarray(prominence, float), np.asarray(relation, float)
    x_mid = float(x.mean())

    fig = go.Figure()

    # --- Nền 4 góc phần tư ---
    x0, x1 = x.min() - 0.35, x.max() + 0.35
    y0, y1 = y.min() - 0.35, y.max() + 0.35
    quads = [
        (x_mid, x1, 0, y1, "rgba(46,134,171,0.10)", "Tác nhân cốt lõi", "top right"),
        (x0, x_mid, 0, y1, "rgba(46,134,171,0.04)", "Nguyên nhân độc lập", "top left"),
        (x_mid, x1, y0, 0, "rgba(199,62,29,0.10)", "Hệ quả cốt lõi", "bottom right"),
        (x0, x_mid, y0, 0, "rgba(199,62,29,0.04)", "Hệ quả độc lập", "bottom left"),
    ]
    for qx0, qx1, qy0, qy1, color, label, pos in quads:
        fig.add_shape(type="rect", x0=qx0, x1=qx1, y0=qy0, y1=qy1,
                      fillcolor=color, line_width=0, layer="below")
        ax = qx1 - 0.06 * (x1 - x0) if "right" in pos else qx0 + 0.06 * (x1 - x0)
        ay = qy1 - 0.06 * (y1 - y0) if "top" in pos else qy0 + 0.06 * (y1 - y0)
        fig.add_annotation(x=ax, y=ay, text=f"<i>{label}</i>", showarrow=False,
                           font=dict(size=11, color="rgba(90,90,90,0.75)"),
                           xanchor="right" if "right" in pos else "left")

    # --- Đường phân chia ---
    fig.add_hline(y=0, line=dict(color="rgba(60,60,60,0.55)", width=1.4, dash="dash"))
    fig.add_vline(x=x_mid, line=dict(color="rgba(60,60,60,0.35)", width=1.2, dash="dot"),
                  annotation_text=f"TB (R+C) = {x_mid:.2f}",
                  annotation_position="top", annotation_font_size=10)

    # --- Mũi tên ảnh hưởng ---
    if show_arrows:
        arrows = [(i, j, T[i, j]) for i in range(n) for j in range(n)
                  if i != j and T[i, j] >= alpha]
        if arrows:
            vmax = max(v for _, _, v in arrows)
            span = max(vmax - alpha, 1e-9)
            for i, j, v in arrows:
                t = (v - alpha) / span
                fig.add_annotation(
                    x=x[j], y=y[j], ax=x[i], ay=y[i],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.1,
                    arrowwidth=0.7 + 2.2 * t,
                    arrowcolor=f"rgba(120,120,140,{0.22 + 0.5 * t:.2f})",
                    standoff=16, startstandoff=16,
                )

    # --- Điểm barrier ---
    is_cause = y > 0
    for mask, color, label in [(is_cause, C_CAUSE, "Nguyên nhân (R-C > 0)"),
                               (~is_cause, C_EFFECT, "Hệ quả (R-C < 0)")]:
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        fig.add_trace(go.Scatter(
            x=x[idx], y=y[idx], mode="markers+text",
            text=[codes[i] for i in idx], textposition="top center",
            textfont=dict(size=12, color="#1a1a1a", family="Inter, sans-serif"),
            marker=dict(size=16 + 20 * (x[idx] - x.min()) / (np.ptp(x) + 1e-9),
                        color=color, line=dict(width=2, color="white"),
                        opacity=0.92),
            name=label,
            customdata=np.column_stack([[names[i] for i in idx], x[idx], y[idx]]),
            hovertemplate="<b>%{text}</b> · %{customdata[0]}<br>"
                          "Độ nổi bật R+C = %{customdata[1]:.3f}<br>"
                          "Quan hệ R-C = %{customdata[2]:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="<b>Bản đồ nhân quả DEMATEL</b><br>"
                        f"<span style='font-size:12px;color:#666'>Mũi tên: quan hệ có T ≥ α* = {alpha:.4f}</span>",
                   x=0.01, xanchor="left"),
        xaxis=dict(title="Độ nổi bật  R + C  (mức độ quan trọng)", gridcolor=C_GRID,
                   range=[x0, x1], zeroline=False),
        yaxis=dict(title="Quan hệ  R - C  (nguyên nhân ↔ hệ quả)", gridcolor=C_GRID,
                   range=[y0, y1], zeroline=False),
        height=620, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        **PLOT_LAYOUT,
    )
    return fig


# ==============================================================================
# 2. SO DO PHAN TANG ISM (DOT / Graphviz)
# ==============================================================================

def ism_dot(codes: Sequence[str], names: Sequence[str],
            levels: Dict[int, List[int]], binary: np.ndarray,
            partition: np.ndarray) -> str:
    """Sinh mã DOT cho sơ đồ phân tầng ISM (Tầng 1 ở đỉnh = hệ quả)."""
    n_levels = len(levels)
    lines = [
        "digraph ISM {",
        '  rankdir=BT; splines=spline; bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Inter,Arial",'
        ' fontsize=12, penwidth=0, height=0.5, margin="0.20,0.10", fontcolor="white"];',
        '  edge [color="#7a8290", penwidth=1.4, arrowsize=0.75];',
    ]

    def level_color(lvl: int) -> str:
        """Đỏ = nguyên nhân gốc (đáy) · Xanh = hệ quả (đỉnh) · Cam/tím = trung gian."""
        if lvl == n_levels:
            return "#C73E1D"
        if lvl == 1:
            return "#3A86FF"
        mids = ["#E8730C", "#B5179E", "#7209B7", "#F4A261", "#0EAD69"]
        return mids[(n_levels - lvl - 1) % len(mids)]

    for lvl in sorted(levels):
        color = level_color(lvl)
        role = "Hệ quả / triệu chứng" if lvl == 1 else (
            "NGUYÊN NHÂN GỐC" if lvl == n_levels else "Trung gian")
        lines.append(f'  subgraph cluster_L{lvl} {{')
        lines.append(f'    label="Tầng {lvl}: {role}"; fontname="Inter,Arial";'
                     f' fontsize=11; fontcolor="#555"; color="#d8dce3"; style="rounded";')
        lines.append("    rank=same;")
        for i in levels[lvl]:
            tip = names[i].replace('"', "'") if i < len(names) else ""
            lines.append(f'    "{codes[i]}" [fillcolor="{color}", tooltip="{tip}"];')
        lines.append("  }")

    # Chỉ vẽ cạnh giữa các tầng khác nhau, hướng từ tầng thấp (nguyên nhân) lên
    n = len(codes)
    for i in range(n):
        for j in range(n):
            if i == j or not binary[i, j]:
                continue
            if partition[i] > partition[j]:
                lines.append(f'  "{codes[j]}" -> "{codes[i]}" [dir=back];')

    lines.append("}")
    return "\n".join(lines)


# ==============================================================================
# 3. MICMAC
# ==============================================================================

def micmac_plot(codes: Sequence[str], names: Sequence[str], mm: Dict, n: int) -> go.Figure:
    """Biểu đồ MICMAC: sức dẫn dắt (driving power) vs mức phụ thuộc (dependence)."""
    dp = np.asarray(mm["driving_power"], float)
    dep = np.asarray(mm["dependence"], float)
    sx = mm.get("split_x", mm["split"])
    sy = mm.get("split_y", mm["split"])

    fig = go.Figure()
    lim = n + 0.6
    quad_bg = [
        (0, sx, sy, lim, "rgba(46,134,171,0.10)", "II. Độc lập / Dẫn dắt"),
        (sx, lim, sy, lim, "rgba(114,9,183,0.10)", "III. Liên kết"),
        (0, sx, 0, sy, "rgba(150,150,150,0.08)", "I. Tự trị"),
        (sx, lim, 0, sy, "rgba(199,62,29,0.10)", "IV. Phụ thuộc"),
    ]
    for qx0, qx1, qy0, qy1, color, label in quad_bg:
        fig.add_shape(type="rect", x0=qx0, x1=qx1, y0=qy0, y1=qy1,
                      fillcolor=color, line_width=0, layer="below")
        ly = qy1 - 0.3 if qy0 >= sy else qy0 + 0.3   # nhãn nằm xa vùng dữ liệu
        fig.add_annotation(x=(qx0 + qx1) / 2, y=ly, text=f"<i>{label}</i>",
                           showarrow=False, font=dict(size=11, color="rgba(80,80,80,0.75)"))

    fig.add_hline(y=sy, line=dict(color="rgba(60,60,60,0.45)", width=1.2, dash="dash"))
    fig.add_vline(x=sx, line=dict(color="rgba(60,60,60,0.45)", width=1.2, dash="dash"))

    # Tách nhãn khi trùng toạ độ
    seen: Dict[tuple, int] = {}
    xs, ys = [], []
    for d, p in zip(dep, dp):
        k = (d, p)
        c = seen.get(k, 0); seen[k] = c + 1
        ang = c * 2.399963            # goc vang, trai deu điểm trùng toạ độ
        rad = 0.20 * np.sqrt(c)
        xs.append(d + rad * np.cos(ang))
        ys.append(p + rad * np.sin(ang))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=list(codes), textposition="top center",
        textfont=dict(size=12), marker=dict(size=15, color="#1d3557",
                                            line=dict(width=2, color="white")),
        customdata=np.column_stack([names, dp, dep, mm["classification"]]),
        hovertemplate="<b>%{text}</b> · %{customdata[0]}<br>Sức dẫn dắt = %{customdata[1]}"
                      "<br>Mức phụ thuộc = %{customdata[2]}<br>Nhóm: %{customdata[3]}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="<b>Phân tích MICMAC</b><br><span style='font-size:12px;color:#666'>"
                        f"Đường chia: phụ thuộc = {sx:.1f} · dẫn dắt = {sy:.1f}</span>",
                   x=0.01, xanchor="left"),
        xaxis=dict(title="Mức phụ thuộc (Dependence)", range=[0, lim], gridcolor=C_GRID, dtick=1),
        yaxis=dict(title="Sức dẫn dắt (Driving power)", range=[0, lim], gridcolor=C_GRID, dtick=1),
        height=560, **PLOT_LAYOUT,
    )
    return fig


# ==============================================================================
# 4. LANDSCAPE CSI + HOI TU SBOA
# ==============================================================================

def csi_landscape(grid: np.ndarray, vals: np.ndarray,
                  alpha_star: float, csi_star: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=vals, mode="lines", name="CSI(α)",
                             line=dict(color="#2E86AB", width=2.6),
                             hovertemplate="α = %{x:.4f}<br>CSI = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[alpha_star], y=[csi_star], mode="markers+text",
                             text=[f"  α* = {alpha_star:.4f}"], textposition="middle right",
                             marker=dict(size=15, color="#C73E1D", symbol="star",
                                         line=dict(width=1.5, color="white")),
                             name="Nghiệm tối ưu SBOA"))
    fig.add_vline(x=alpha_star, line=dict(color="#C73E1D", width=1.2, dash="dot"))
    fig.update_layout(
        title=dict(text="<b>Không gian mục tiêu CSI(α)</b><br>"
                        "<span style='font-size:12px;color:#666'>Vùng trống = α không khả thi "
                        "(số mũi tên &lt; n)</span>", x=0.01, xanchor="left"),
        xaxis=dict(title="Ngưỡng α", gridcolor=C_GRID),
        yaxis=dict(title="CSI(α) = √(C̃S · L̃)", gridcolor=C_GRID),
        height=430,
        legend=dict(orientation="h", y=0.02, x=0.98, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,.75)", bordercolor="#e6e9ef", borderwidth=1),
        **PLOT_LAYOUT,
    )
    return fig


def convergence_plot(history: Sequence[float]) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=list(history), mode="lines", line=dict(color="#0EAD69", width=2.4),
        fill="tozeroy", fillcolor="rgba(14,173,105,0.12)",
        hovertemplate="Vòng lặp %{x}<br>CSI tốt nhất = %{y:.4f}<extra></extra>"))
    fig.update_layout(
        title=dict(text="<b>Đường hội tụ Secretary Bird</b>", x=0.01, xanchor="left"),
        xaxis=dict(title="Vòng lặp", gridcolor=C_GRID),
        yaxis=dict(title="CSI tốt nhất", gridcolor=C_GRID),
        height=430, **PLOT_LAYOUT,
    )
    return fig


# ==============================================================================
# 5. HEATMAP & BAR
# ==============================================================================

def matrix_heatmap(M: np.ndarray, codes: Sequence[str], title: str,
                   colorscale: str = "Blues", zmin=None, zmax=None,
                   text_fmt: str = ".3f") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=M, x=list(codes), y=list(codes), colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        text=np.round(M, 3), texttemplate="%{text:" + text_fmt + "}",
        textfont=dict(size=9),
        hovertemplate="%{y} → %{x}<br>giá trị = %{z:.4f}<extra></extra>",
        colorbar=dict(thickness=12, len=0.85),
    ))
    lay = dict(PLOT_LAYOUT)
    lay["margin"] = dict(l=60, r=40, t=115, b=40)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, xanchor="left", y=0.97, yanchor="top"),
        xaxis=dict(title=dict(text="Chịu ảnh hưởng (j)", standoff=6),
                   side="top", tickangle=0),
        yaxis=dict(title="Gây ảnh hưởng (i)", autorange="reversed"),
        height=max(440, 42 * len(codes)), **lay,
    )
    return fig


def weights_bar(codes: Sequence[str], names: Sequence[str], w: np.ndarray) -> go.Figure:
    order = np.argsort(w)
    fig = go.Figure(go.Bar(
        x=w[order], y=[codes[i] for i in order], orientation="h",
        marker=dict(color=w[order], colorscale="Teal", line=dict(width=0)),
        text=[f"{v:.4f}" for v in w[order]], textposition="outside",
        customdata=[names[i] for i in order],
        hovertemplate="<b>%{y}</b> · %{customdata}<br>Trọng số = %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Trọng số BWM của các rào cản</b>", x=0.01, xanchor="left"),
        xaxis=dict(title="Trọng số w", gridcolor=C_GRID, range=[0, float(w.max()) * 1.2]),
        yaxis=dict(title=""), height=max(380, 34 * len(codes)),
        showlegend=False, **PLOT_LAYOUT,
    )
    return fig


def prominence_bar(codes: Sequence[str], prominence: np.ndarray,
                   relation: np.ndarray) -> go.Figure:
    order = np.argsort(prominence)
    colors = [C_CAUSE if relation[i] > 0 else C_EFFECT for i in order]
    fig = go.Figure(go.Bar(
        x=prominence[order], y=[codes[i] for i in order], orientation="h",
        marker=dict(color=colors), text=[f"{prominence[i]:.3f}" for i in order],
        textposition="outside",
        customdata=[f"{relation[i]:+.3f}" for i in order],
        hovertemplate="<b>%{y}</b><br>R+C = %{x:.3f}<br>R-C = %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Xếp hạng mức độ quan trọng (R+C)</b><br>"
                        "<span style='font-size:12px;color:#666'>Xanh = nguyên nhân · Đỏ = hệ quả</span>",
                   x=0.01, xanchor="left"),
        xaxis=dict(title="Độ nổi bật R + C", gridcolor=C_GRID,
                   range=[0, float(prominence.max()) * 1.18]),
        yaxis=dict(title=""), height=max(380, 34 * len(codes)),
        showlegend=False, **PLOT_LAYOUT,
    )
    return fig


# ==============================================================================
# 6. SO DO QUY TRINH FRAMEWORK
# ==============================================================================

FRAMEWORK_DOT = """
digraph SBBDI {
  rankdir=TB; bgcolor="transparent"; nodesep=0.28; ranksep=0.42;
  node [shape=box, style="rounded,filled", fontname="Inter,Arial", fontsize=11,
        penwidth=0, fontcolor="white", margin="0.22,0.13"];
  edge [color="#8b93a1", penwidth=1.5, arrowsize=0.8];

  subgraph cluster_in {
    label="① Đầu vào chuyên gia"; fontname="Inter,Arial"; fontsize=11;
    fontcolor="#555"; color="#d8dce3"; style="rounded";
    BO [label="Best-to-Others\\nOthers-to-Worst", fillcolor="#264653"];
    Z  [label="Ma trận quan hệ\\ntrực tiếp Z", fillcolor="#264653"];
  }
  BWM [label="② BWM tuyến tính\\nmin ξ  →  trọng số w", fillcolor="#2E86AB"];
  DEM [label="③ DEMATEL\\nT = N(I-N)⁻¹ · (n·wⱼ)", fillcolor="#0EAD69"];
  CSI [label="④ CSI(α) = √(C̃S · L̃)\\nchỉ số cấu trúc nhân quả", fillcolor="#E8730C"];
  SBO [label="⑤ Secretary Bird\\ntối ưu → α*", fillcolor="#C73E1D"];
  ISM [label="⑥ ISM + MICMAC\\nphân tầng, nguyên nhân gốc", fillcolor="#7209B7"];

  BO -> BWM; BWM -> DEM; Z -> DEM; DEM -> CSI; CSI -> SBO; SBO -> ISM;
}
"""


# ==============================================================================
# 7. SO DO QUY TRINH RUT GON (ngon ngu quan ly)
# ==============================================================================

SIMPLE_FLOW_DOT = """
digraph FLOW {
  rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.38;
  node [shape=box, style="rounded,filled", fontname="Inter,Arial", fontsize=11,
        penwidth=0, fontcolor="white", margin="0.24,0.14", width=2.6];
  edge [color="#98a0ae", penwidth=1.5, arrowsize=0.75];

  S1 [label="1. Liệt kê rào cản", fillcolor="#264653"];
  S2 [label="2. Chuyên gia chấm điểm\\nmức quan trọng", fillcolor="#2E86AB"];
  S3 [label="3. Chuyên gia chấm\\nquan hệ ảnh hưởng", fillcolor="#0EAD69"];
  S4 [label="4. Hệ thống dựng\\nbản đồ nhân quả", fillcolor="#E8730C"];
  S5 [label="5. Xếp tầng và chỉ ra\\nthứ tự can thiệp", fillcolor="#C73E1D"];

  S1 -> S2 -> S3 -> S4 -> S5;
}
"""

# ##############################################################################
# ##############################################################################
##                        PHAN C.  GIAO DIEN STREAMLIT                        ##
# ##############################################################################
# ##############################################################################

# ==============================================================================
# CẤU HÌNH TRANG
# ==============================================================================

st.set_page_config(
    page_title="SB-BDI · Phân tích rào cản",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1, h2, h3 {font-family: Inter, "Segoe UI", system-ui, sans-serif; letter-spacing:-0.01em;}
  .hero {
    background: linear-gradient(120deg,#0f2027 0%,#203a43 48%,#2c5364 100%);
    color:#fff; padding: 2.1rem 2.3rem; border-radius: 16px; margin-bottom: 1.4rem;
  }
  .hero h1 {color:#fff; margin:0 0 .35rem 0; font-size: 2.05rem;}
  .hero p {color:#cfe3ee; margin:0; font-size:1.02rem; line-height:1.55;}
  .hero .badge {
    display:inline-block; background:rgba(255,255,255,.14); color:#eaf6ff;
    padding:.25rem .7rem; border-radius:999px; font-size:.78rem; margin-right:.4rem;
    margin-top:.9rem; border:1px solid rgba(255,255,255,.18);
  }
  .card {
    background:#fff; border:1px solid #e6e9ef; border-radius:13px;
    padding:1.15rem 1.3rem; height:100%;
  }
  .card h4 {margin:.1rem 0 .5rem 0; font-size:1.02rem; color:#12304a;}
  .card p  {margin:0; color:#5a6675; font-size:.92rem; line-height:1.55;}
  .keyfind {
    border-left:4px solid #C73E1D; background:#fff5f2; padding:.85rem 1.1rem;
    border-radius:0 10px 10px 0; margin-bottom:.7rem;
  }
  .keyfind.blue  {border-left-color:#2E86AB; background:#f1f8fc;}
  .keyfind.green {border-left-color:#0EAD69; background:#f0fbf6;}
  .keyfind b {color:#12304a;}
  .stTabs [data-baseweb="tab-list"] {gap: 2px;}
  .stTabs [data-baseweb="tab"] {padding: 10px 18px; font-size:.94rem;}
  footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# KHỞI TẠO TRẠNG THÁI
# ==============================================================================

def load_demo():
    n = len(DEMO_CODES)
    st.session_state.update(
        n=n,
        codes=list(DEMO_CODES),
        names=list(DEMO_NAMES),
        dims=list(DEMO_DIMS),
        Z=DEMO_Z.copy(),
        BO=DEMO_BO.copy(),
        WO=DEMO_WO.copy(),
        weights=DEMO_WEIGHTS.copy(),
        best_idx=DEMO_BEST,
        worst_idx=DEMO_WORST,
        results=None,
        bwm_out=None,
        demo_loaded=True,
    )


def load_random_case(iters: int = 60, pop: int = 20):
    """Sinh một tình huống doanh nghiệp ngẫu nhiên rồi chạy trọn bộ phân tích."""
    import random as _random
    sd = _random.randint(1, 999_999)
    c = random_case(seed=sd)
    n = len(c["codes"])
    st.session_state.update(
        n=n, codes=c["codes"], names=c["names"], dims=c["dims"],
        Z=c["Z"], BO=c["BO"], WO=c["WO"], weights=c["weights"],
        best_idx=c["best_idx"], worst_idx=c["worst_idx"],
        results=None, bwm_out=None, demo_loaded=False,
        random_title=c["title"], random_seed=sd,
    )
    try:
        st.session_state["bwm_out"] = run_bwm(
            c["BO"], c["WO"], c["best_idx"], c["worst_idx"])
    except Exception:
        st.session_state["bwm_out"] = None
    st.session_state["results"] = run_full_pipeline(
        c["Z"], c["weights"], iterations=iters, pop_size=pop, seed=sd)
    return c["title"], n


def blank_setup(n: int):
    st.session_state.update(
        n=n,
        codes=[f"B{i+1}" for i in range(n)],
        names=[f"Rào cản {i+1}" for i in range(n)],
        dims=["Chưa phân nhóm"] * n,
        Z=np.zeros((n, n)),
        BO=np.full((3, n), 3.0),
        WO=np.full((3, n), 3.0),
        weights=np.ones(n) / n,
        best_idx=0,
        worst_idx=min(1, n - 1),
        results=None,
        bwm_out=None,
        demo_loaded=False,
    )


if "n" not in st.session_state:
    blank_setup(6)


def S(k):
    return st.session_state[k]


def S_get(k, default=None):
    return st.session_state.get(k, default)


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown("### 🦅 SB-BDI")
    st.caption("Secretary Bird, Barrier Dependency Interpretation")

    st.markdown("---")
    st.markdown("**Dữ liệu**")
    if st.button("🎲 Chạy thử tình huống ngẫu nhiên", use_container_width=True,
                 type="primary",
                 help="Tự dựng một tình huống doanh nghiệp và chạy trọn bộ phân tích "
                      "để xem đầu ra trông ra sao"):
        with st.spinner("Đang dựng tình huống và phân tích..."):
            title, nn = load_random_case()
        st.success(f"Đã tạo: {title} ({nn} rào cản). Mở tab **Kết quả & Diễn giải**.")
    if st.button("📥 Nạp bộ dữ liệu mẫu", use_container_width=True,
                 help="Case study: 12 rào cản sản xuất lúa hữu cơ ở ĐBSCL"):
        load_demo()
        st.success("Đã nạp dữ liệu mẫu (12 rào cản).")
    if st.button("🗑️ Xoá & bắt đầu lại", use_container_width=True):
        blank_setup(6)
        st.session_state.pop("random_title", None)
        st.rerun()

    st.markdown("---")
    st.markdown("**Tham số Secretary Bird**")
    iters = st.slider("Số vòng lặp", 20, 200, 80, 10)
    pop = st.slider("Kích thước quần thể", 10, 60, 25, 5)
    seed = st.number_input("Seed ngẫu nhiên", 0, 9999, 42,
                           help="Cố định seed để kết quả tái lập được.")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.82rem;color:#6b7280;line-height:1.5'>"
        "<b>Tác giả framework</b><br>Tôn Nguyễn Trọng Hiển<br>"
        "<a href='https://orcid.org/0000-0002-6970-0799' target='_blank'>ORCID: 0000-0002-6970-0799</a>"
        "</div>", unsafe_allow_html=True)

    if S_get("random_title"):
        st.caption(f"🎲 Tình huống demo: **{S_get('random_title')}**")
    st.caption(f"Đang cấu hình: **{S('n')} rào cản**")


# ==============================================================================
# HERO
# ==============================================================================

st.markdown("""
<div class="hero">
  <h1>🦅 SB-BDI · Hệ hỗ trợ ra quyết định phân tích rào cản</h1>
  <p>Chỉ ra rào cản nào là <b>nguyên nhân gốc</b>, rào cản nào chỉ là <b>triệu chứng</b>,
  và nên <b>xử lý theo thứ tự nào</b> để nguồn lực bỏ ra tạo hiệu ứng lan toả lớn nhất.</p>
  <span class="badge">Best-Worst Method</span>
  <span class="badge">DEMATEL</span>
  <span class="badge">ISM &amp; MICMAC</span>
  <span class="badge">Secretary Bird Optimization</span>
</div>
""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖  Giới thiệu",
    "⚙️  Cấu hình rào cản",
    "⚖️  BWM · Trọng số",
    "🔗  DEMATEL · Quan hệ",
    "🦅  Tối ưu ngưỡng α",
    "📊  Kết quả & Diễn giải",
])


# ==============================================================================
# TAB 0. GIOI THIEU
# ==============================================================================

with tab0:
    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.markdown("## Công cụ này giúp bạn điều gì?")
        st.markdown("""
Tổ chức nào cũng có danh sách vấn đề tồn đọng dài hơn nguồn lực để xử lý. Câu hỏi khó
không phải *"chúng ta đang vướng những gì"*, vì cái đó ai cũng liệt kê được. Câu hỏi khó là
**bỏ tiền và thời gian vào đâu trước thì cả hệ thống mới chuyển động**.

Đây chính là chỗ hay ra quyết định sai. Vấn đề gây ồn ào nhất, được nhắc nhiều nhất trong
các cuộc họp, thường lại là **hệ quả** của một nguyên nhân sâu hơn nằm ở chỗ khác. Xử lý nó
thì tốn kém mà vài tháng sau vấn đề quay lại y như cũ.

**SB-BDI trả lời ba câu hỏi mà một bảng xếp hạng thông thường không trả lời được:**

1. Rào cản nào **thực sự quan trọng** với hệ thống, không phải cái nào được nói to nhất.
2. Rào cản nào chỉ là **triệu chứng** của rào cản khác, tức là xử lý sẽ không dứt điểm.
3. Nên can thiệp theo **thứ tự nào** để một đồng bỏ ra tạo hiệu ứng lan toả lớn nhất.
        """)

        st.markdown("### Bốn lợi ích cụ thể")
        b1, b2 = st.columns(2)
        benefits = [
            ("🎯", "Đúng thứ tự, không dàn trải",
             "Thay vì chia đều ngân sách cho mười vấn đề, bạn biết vấn đề nào là gốc và "
             "xử lý nó sẽ kéo theo bao nhiêu vấn đề khác tự giảm."),
            ("🧭", "Loại bỏ cảm tính khỏi cuộc họp",
             "Ý kiến chuyên gia được chấm điểm có kiểm định. Người đánh giá thiếu nhất quán "
             "sẽ bị phát hiện ngay, thay vì tranh luận ai đúng ai sai."),
            ("🔁", "Cùng dữ liệu cho cùng kết quả",
             "Không có tham số nào do người phân tích tự chọn. Ai chạy lại cũng ra đúng "
             "kết quả đó, nên kết luận không phụ thuộc vào người làm báo cáo."),
            ("📑", "Bằng chứng để bảo vệ quyết định",
             "Bạn có số liệu, chỉ số kiểm định và sơ đồ trực quan để trình bày trước ban "
             "lãnh đạo hoặc hội đồng, thay vì chỉ nói *theo kinh nghiệm của tôi*."),
        ]
        for col, (icon, title, body) in zip([b1, b2, b1, b2], benefits):
            col.markdown(
                f"<div class='card' style='margin-bottom:.8rem'><h4>{icon} {title}</h4>"
                f"<p>{body}</p></div>", unsafe_allow_html=True)

        st.markdown("### So với cách làm thường gặp")
        st.markdown("""
| Tình huống | Cách làm phổ biến hiện nay | Khi dùng SB-BDI |
|---|---|---|
| Chọn vấn đề ưu tiên | Họp bàn rồi biểu quyết, ai lập luận mạnh hơn thì ý kiến đó thắng | Chấm điểm theo phương pháp có kiểm định nhất quán, phát hiện được đánh giá mâu thuẫn |
| Tìm nguyên nhân gốc | Dựa vào kinh nghiệm và trực giác của vài người | Suy ra từ cấu trúc quan hệ giữa toàn bộ rào cản, không ai áp đặt |
| Vẽ sơ đồ quan hệ | Tự chọn tay mức lọc, mỗi người vẽ ra một sơ đồ khác nhau | Thuật toán tự tìm mức lọc tốt nhất, cùng dữ liệu luôn cho cùng sơ đồ |
| Phân bổ nguồn lực | Chia đều hoặc ưu tiên nơi kêu to nhất | Theo thứ tự tầng, xử lý gốc trước để tạo hiệu ứng lan toả |
| Bảo vệ kết luận | Khó phản biện vì không có căn cứ định lượng | Có chỉ số, bảng số liệu và sơ đồ kiểm chứng được |
        """)

        st.markdown("### Vì sao kết quả đáng tin cậy")
        st.markdown("""
- **Chất lượng đầu vào được kiểm tra, không nhận bừa.** Mỗi chuyên gia được tính một chỉ số
  nhất quán. Nếu người đó chấm mâu thuẫn với chính mình, hệ thống báo ngay để bạn rà lại,
  thay vì để dữ liệu xấu lọt vào kết quả cuối.
- **Không có nút vặn nào để người phân tích "chỉnh" ra kết quả mong muốn.** Ở các phương pháp
  tương tự, người làm phải tự chọn một mức lọc và mức đó quyết định luôn hình dạng kết luận.
  SB-BDI để máy tự tìm mức lọc tối ưu, nên không thể vô tình hay cố ý lái kết quả.
- **Tái lập được.** Cùng bộ dữ liệu, chạy bao nhiêu lần cũng ra cùng một cấu trúc. Đây là điều
  kiện bắt buộc nếu kết quả dùng để trình hội đồng hoặc đưa vào báo cáo chính thức.
- **Ba góc nhìn kiểm tra chéo nhau.** Mức độ quan trọng, chiều tác động, và vị trí trong chuỗi
  nhân quả được tính bằng ba phương pháp riêng biệt. Khi cả ba cùng chỉ về một rào cản, độ tin
  cậy của kết luận cao hơn hẳn so với chỉ dùng một bảng xếp hạng.
- **Nền tảng đã được công bố khoa học.** Ba phương pháp thành phần đều là chuẩn mực học thuật
  được dùng rộng rãi hàng chục năm, không phải công thức tự nghĩ ra.
        """)

        st.markdown("### Bạn nhận được gì sau khi chạy")
        st.markdown("""
1. **Bảng xếp hạng** mức độ quan trọng của từng rào cản, kèm vai trò là nguyên nhân hay hệ quả.
2. **Sơ đồ phân tầng** chỉ rõ đâu là nguyên nhân gốc, đâu là triệu chứng bề mặt.
3. **Thứ tự can thiệp** theo từng giai đoạn, kèm nhận định về độ tin cậy của kết luận.
4. **File Excel** đầy đủ số liệu để đưa vào báo cáo hoặc phụ lục.
        """)

        with st.expander("🔬 Dành cho người làm phân tích: chi tiết phương pháp"):
            st.markdown(r"""
**SB-BDI** (*Secretary Bird, Barrier Dependency Interpretation*) tích hợp ba phương pháp:

1. **BWM** (*Best-Worst Method*, Rezaei 2015/2016) xác định trọng số tầm quan trọng, cần ít
   phép so sánh hơn AHP và cho độ nhất quán cao hơn.
2. **DEMATEL** chuyển ma trận ảnh hưởng trực tiếp thành ma trận quan hệ tổng
   $T = N(I-N)^{-1}$, tách nhóm nguyên nhân và nhóm hệ quả.
3. **ISM** phân rã hệ thống thành cấu trúc phân tầng có hướng.

**Vấn đề kỹ thuật mà framework giải quyết.** Khi ghép DEMATEL với ISM, ngưỡng cắt $\alpha$
quyết định quan hệ nào trong $T$ được giữ lại. Thực tiễn vẫn chọn $\alpha$ ngoại sinh
(trung bình, trung bình cộng độ lệch chuẩn, phân vị 75%, hoặc do chuyên gia ấn định).
Hệ quả: $\alpha$ lệch một chút thì số tầng thay đổi, nguyên nhân gốc có thể biến mất, và
kết luận không tái lập được giữa các nghiên cứu.

**Đóng góp cốt lõi.** Biến $\alpha$ thành biến quyết định nội sinh, xác định bằng cách cực
đại một hàm mục tiêu không có tham số tự do:
            """)
            st.latex(r"\mathrm{CSI}(\alpha)=\sqrt{\widetilde{CS}(\alpha)\;\cdot\;\widetilde{L}(\alpha)}")
            st.markdown(r"""
- $CS(\alpha) = S / N_{\text{arrows}}(\alpha)$ là cường độ nhân quả đạt được trên mỗi mũi tên
  giữ lại. Tử số $S = \left|\overline{(R-C)}_{>0}\right| + \left|\overline{(R-C)}_{<0}\right|$
  được cố định từ ma trận $T$ chưa cắt ngưỡng, tránh phụ thuộc vòng tròn vào $\alpha$.
- $L(\alpha)$ là số tầng của cấu trúc ISM, đại diện cho chiều sâu diễn giải.
- Trung bình nhân bảo đảm tính không thay thế: một thành phần bằng 0 thì CSI bằng 0, nên không
  thể đánh đổi chiều sâu lấy số lượng mũi tên hay ngược lại.
- Ràng buộc khả thi: $N_{\text{arrows}}(\alpha) \ge n$ để đồ thị không rời rạc.

Vì CSI là hàm bậc thang, đa cực trị và không khả vi theo $\alpha$, thuật toán
**Secretary Bird Optimization** (Fu và cộng sự, 2024) với ba pha săn mồi và hai chiến lược
trốn thoát được dùng để dò tìm $\alpha^*$.
            """)
            st.graphviz_chart(FRAMEWORK_DOT, use_container_width=True)

    with c2:
        st.markdown("### Quy trình 5 bước")
        st.graphviz_chart(SIMPLE_FLOW_DOT, use_container_width=True)

        st.markdown("""
<div class="card" style="margin-top:1rem">
<h4>⏱️ Cần chuẩn bị những gì</h4>
<p><b>Danh sách rào cản:</b> thường 8 tới 15 mục, do nhóm chuyên môn thống nhất.<br><br>
<b>Ý kiến chuyên gia:</b> từ 3 người trở lên. Mỗi người mất khoảng 20 phút để chấm hai bảng.<br><br>
<b>Thời gian chạy:</b> dưới một phút. Phần lâu nhất là thu thập ý kiến, không phải tính toán.</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="card" style="margin-top:.9rem">
<h4>👤 Tác giả framework</h4>
<p><b>Tôn Nguyễn Trọng Hiển</b><br>
Người đề xuất và phát triển framework SB-BDI.<br><br>
ORCID: <a href="https://orcid.org/0000-0002-6970-0799" target="_blank">
0000-0002-6970-0799</a></p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="card" style="margin-top:.9rem">
<h4>📌 Kết quả trả về</h4>
<p>• Xếp hạng mức độ quan trọng từng rào cản<br>
• Bản đồ nguyên nhân và hệ quả<br>
• Sơ đồ phân tầng, chỉ rõ nguyên nhân gốc<br>
• Phân loại rào cản theo bốn nhóm hành động<br>
• Nhận định tự động về độ tin cậy kết quả<br>
• Thứ tự can thiệp theo giai đoạn<br>
• File Excel đầy đủ để đưa vào báo cáo</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Bắt đầu như thế nào?")
    g1, g2, g3 = st.columns(3)
    for col, (icon, title, body) in zip(
        [g1, g2, g3],
        [("①", "Xem thử trước khi nhập liệu",
          "Bấm <b>🎲 Chạy thử tình huống ngẫu nhiên</b> ở thanh bên. Hệ thống tự dựng một "
          "tình huống doanh nghiệp và chạy trọn bộ phân tích để bạn xem đầu ra trông ra sao."),
         ("②", "Khai báo rào cản của bạn",
          "Sang tab <b>Cấu hình rào cản</b>, chọn số lượng và đặt tên cho từng rào cản, "
          "hoặc nhập sẵn từ file Excel."),
         ("③", "Nhập ý kiến rồi chạy",
          "Tab <b>BWM</b> để chấm mức độ quan trọng, tab <b>DEMATEL</b> để chấm quan hệ ảnh "
          "hưởng, rồi bấm chạy. Kết quả hiện ở tab cuối cùng.")],
    ):
        col.markdown(
            f"<div class='card'><h4>{icon} {title}</h4><p>{body}</p></div>",
            unsafe_allow_html=True)

    st.info("💡 Muốn xem ngay đầu ra mà chưa có dữ liệu? Bấm **🎲 Chạy thử tình huống ngẫu nhiên** "
            "ở thanh bên, hoặc **📥 Nạp bộ dữ liệu mẫu** để dùng case study 12 rào cản có thật.")


# ==============================================================================
# TAB 1. CAU HINH RAO CAN
# ==============================================================================

with tab1:
    st.markdown("## ⚙️ Khai báo hệ thống rào cản")
    st.caption("Chọn số lượng rào cản tuỳ ý, sau đó đặt mã, tên đầy đủ và nhóm phân loại cho từng rào cản.")

    c1, c2 = st.columns([1, 2.4])
    with c1:
        new_n = st.number_input("Số lượng rào cản (n)", min_value=3, max_value=30,
                                value=int(S("n")), step=1,
                                help="Có thể phân tích từ 3 đến 30 rào cản.")
        if st.button("✅ Áp dụng số lượng", type="primary", use_container_width=True):
            n_old, n_new = S("n"), int(new_n)
            if n_new != n_old:
                def fit(lst, filler):
                    lst = list(lst)[:n_new]
                    return lst + [filler(i) for i in range(len(lst), n_new)]

                Z_old = S("Z")
                Z_new = np.zeros((n_new, n_new))
                k = min(n_old, n_new)
                Z_new[:k, :k] = Z_old[:k, :k]

                st.session_state.update(
                    n=n_new,
                    codes=fit(S("codes"), lambda i: f"B{i+1}"),
                    names=fit(S("names"), lambda i: f"Rào cản {i+1}"),
                    dims=fit(S("dims"), lambda i: "Chưa phân nhóm"),
                    Z=Z_new,
                    BO=np.full((S("BO").shape[0], n_new), 3.0),
                    WO=np.full((S("WO").shape[0], n_new), 3.0),
                    weights=np.ones(n_new) / n_new,
                    best_idx=0, worst_idx=min(1, n_new - 1),
                    results=None, bwm_out=None,
                )
                st.success(f"Đã đặt lại hệ thống thành {n_new} rào cản.")
                st.rerun()
            else:
                st.info("Số lượng không thay đổi.")

        st.markdown("---")
        st.markdown("**Nhập nhanh từ file**")
        up = st.file_uploader("CSV/Excel: cột `Mã`, `Tên`, `Nhóm`", type=["csv", "xlsx"],
                              key="up_barriers")
        if up is not None:
            try:
                df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                df_up.columns = [str(c).strip().lower() for c in df_up.columns]
                cmap = {"mã": "code", "ma": "code", "code": "code",
                        "tên": "name", "ten": "name", "name": "name",
                        "nhóm": "dim", "nhom": "dim", "dim": "dim", "dimension": "dim"}
                df_up = df_up.rename(columns={c: cmap.get(c, c) for c in df_up.columns})
                nn = len(df_up)
                st.session_state.update(
                    n=nn,
                    codes=[str(v) for v in df_up["code"]],
                    names=[str(v) for v in df_up.get("name", df_up["code"])],
                    dims=[str(v) for v in df_up.get("dim", ["Chưa phân nhóm"] * nn)],
                    Z=np.zeros((nn, nn)),
                    BO=np.full((3, nn), 3.0), WO=np.full((3, nn), 3.0),
                    weights=np.ones(nn) / nn, best_idx=0, worst_idx=min(1, nn - 1),
                    results=None, bwm_out=None,
                )
                st.success(f"Đã nạp {nn} rào cản từ file.")
                st.rerun()
            except ImportError:
                st.error("Máy chủ chưa cài gói đọc Excel. Hãy lưu file sang định dạng CSV "
                         "rồi tải lên lại, hoặc thêm dòng `openpyxl>=3.1` vào requirements.txt.")
            except Exception as e:
                st.error(f"Không đọc được file: {e}")

    with c2:
        df_b = pd.DataFrame({
            "Mã": S("codes"),
            "Tên rào cản": S("names"),
            "Nhóm": S("dims"),
        })
        edited = st.data_editor(
            df_b, use_container_width=True, num_rows="fixed", hide_index=True,
            key="ed_barriers",
            column_config={
                "Mã": st.column_config.TextColumn("Mã", width="small",
                                                  help="Mã ngắn, ví dụ EC1, TE2…"),
                "Tên rào cản": st.column_config.TextColumn("Tên rào cản", width="large"),
                "Nhóm": st.column_config.TextColumn("Nhóm / Khía cạnh", width="medium"),
            },
        )
        codes_new = [str(v).strip() or f"B{i+1}" for i, v in enumerate(edited["Mã"])]
        if len(set(codes_new)) != len(codes_new):
            st.warning("⚠️ Có mã rào cản bị trùng, vui lòng đặt mã duy nhất cho mỗi rào cản.")
        st.session_state["codes"] = codes_new
        st.session_state["names"] = [str(v) for v in edited["Tên rào cản"]]
        st.session_state["dims"] = [str(v) for v in edited["Nhóm"]]

        st.caption(f"Hệ thống hiện có **{S('n')} rào cản** thuộc "
                   f"**{len(set(S('dims')))} nhóm**.")


# ==============================================================================
# TAB 2. BWM
# ==============================================================================

with tab2:
    st.markdown("## ⚖️ BWM: Xác định trọng số rào cản")
    codes = S("codes"); n = S("n")

    mode = st.radio(
        "Cách xác định trọng số",
        ["Chạy BWM từ đánh giá chuyên gia", "Nhập trực tiếp trọng số"],
        horizontal=True, key="bwm_mode",
    )

    if mode == "Nhập trực tiếp trọng số":
        st.caption("Dùng khi bạn đã có sẵn trọng số từ nghiên cứu trước. Hệ thống sẽ tự chuẩn hoá về tổng = 1.")
        dfw = pd.DataFrame({"Mã": codes, "Tên": S("names"),
                            "Trọng số": np.round(S("weights"), 4)})
        edw = st.data_editor(dfw, hide_index=True, use_container_width=True,
                             disabled=["Mã", "Tên"], key="ed_w",
                             column_config={"Trọng số": st.column_config.NumberColumn(
                                 min_value=0.0, max_value=1.0, step=0.001, format="%.4f")})
        w = np.array(edw["Trọng số"], dtype=float)
        if w.sum() > 0:
            st.session_state["weights"] = w / w.sum()
        st.plotly_chart(weights_bar(codes, S("names"), S("weights")),
                        use_container_width=True)

    else:
        st.markdown("""
Với mỗi chuyên gia, hãy xác định rào cản **quan trọng nhất (Best)** và **ít quan trọng nhất (Worst)**,
sau đó chấm điểm theo thang **1 tới 9** (Saaty):

- **Best-to-Others (BO)**: mức độ Best quan trọng hơn từng rào cản còn lại. Ô của chính Best = **1**.
- **Others-to-Worst (WO)**: mức độ từng rào cản quan trọng hơn Worst. Ô của chính Worst = **1**.
        """)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            n_exp = st.number_input("Số chuyên gia", 1, 30, int(S("BO").shape[0]), 1)
        with c2:
            b_idx = st.selectbox("Rào cản quan trọng nhất (Best)", range(n),
                                 index=min(S("best_idx"), n - 1),
                                 format_func=lambda i: f"{codes[i]}: {S('names')[i]}")
        with c3:
            w_idx = st.selectbox("Rào cản ít quan trọng nhất (Worst)", range(n),
                                 index=min(S("worst_idx"), n - 1),
                                 format_func=lambda i: f"{codes[i]}: {S('names')[i]}")

        st.session_state["best_idx"], st.session_state["worst_idx"] = int(b_idx), int(w_idx)

        if b_idx == w_idx:
            st.error("Best và Worst phải là hai rào cản khác nhau.")
        else:
            def resize(M, rows, cols, fill=3.0):
                out = np.full((rows, cols), fill)
                r, c = min(rows, M.shape[0]), min(cols, M.shape[1])
                out[:r, :c] = M[:r, :c]
                return out

            BO = resize(S("BO"), int(n_exp), n)
            WO = resize(S("WO"), int(n_exp), n)
            BO[:, b_idx] = 1.0
            WO[:, w_idx] = 1.0

            idx = [f"Chuyên gia {i+1}" for i in range(int(n_exp))]
            colcfg = {c: st.column_config.NumberColumn(c, min_value=1, max_value=9, step=1,
                                                       format="%d", width="small")
                      for c in codes}

            st.markdown(f"##### Bảng BO: so sánh **{codes[b_idx]}** (Best) với các rào cản khác")
            bo_ed = st.data_editor(pd.DataFrame(BO, columns=codes, index=idx),
                                   use_container_width=True, column_config=colcfg, key="ed_bo")

            st.markdown(f"##### Bảng WO: so sánh các rào cản với **{codes[w_idx]}** (Worst)")
            wo_ed = st.data_editor(pd.DataFrame(WO, columns=codes, index=idx),
                                   use_container_width=True, column_config=colcfg, key="ed_wo")

            BO = np.array(bo_ed, dtype=float); BO[:, b_idx] = 1.0
            WO = np.array(wo_ed, dtype=float); WO[:, w_idx] = 1.0
            st.session_state["BO"], st.session_state["WO"] = BO, WO

            agg = st.selectbox("Cách tổng hợp ý kiến chuyên gia",
                               ["Trung bình nhân (AIJ, khuyến nghị)", "Trung bình cộng"])

            if st.button("▶️ Tính trọng số BWM", type="primary"):
                try:
                    out = run_bwm(
                        BO, WO, int(b_idx), int(w_idx),
                        agg="geometric" if agg.startswith("Trung bình nhân") else "arithmetic")
                    st.session_state["bwm_out"] = out
                    st.session_state["weights"] = out["weights"]
                    st.session_state["results"] = None
                except Exception as e:
                    st.error(f"Lỗi khi giải BWM: {e}")

        out = S("bwm_out")
        if out is not None and len(out["weights"]) == n:
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("CR lớn nhất", f"{out['CR'].max():.4f}",
                      "Đạt (< 0,10)" if out["all_consistent"] else "Vượt ngưỡng",
                      delta_color="normal" if out["all_consistent"] else "inverse")
            m2.metric("Rào cản trọng số cao nhất",
                      codes[int(np.argmax(out['weights']))],
                      f"w = {out['weights'].max():.4f}")
            m3.metric("Số chuyên gia", f"{len(out['CR'])}")

            if not out["all_consistent"]:
                st.warning("⚠️ Một số chuyên gia có CR ≥ 0,10, nên rà soát lại các đánh giá "
                           "thiếu nhất quán trước khi dùng kết quả.")

            cc1, cc2 = st.columns([1.3, 1])
            with cc1:
                st.plotly_chart(weights_bar(codes, S("names"), out["weights"]),
                                use_container_width=True)
            with cc2:
                st.markdown("##### Kiểm định nhất quán từng chuyên gia")
                st.dataframe(pd.DataFrame({
                    "Chuyên gia": [f"CG {i+1}" for i in range(len(out["CR"]))],
                    "ξ*": np.round(out["xi"], 4),
                    "a_BW": out["a_BW"],
                    "CR": np.round(out["CR"], 4),
                    "Đánh giá": ["✅ Nhất quán" if c < 0.1 else "⚠️ Cần xem lại"
                                 for c in out["CR"]],
                }), hide_index=True, use_container_width=True, height=360)

            st.markdown("##### Trọng số tổng hợp")
            st.dataframe(pd.DataFrame({
                "Mã": codes, "Tên": S("names"), "Nhóm": S("dims"),
                "Trọng số": np.round(out["weights"], 4),
                "Xếp hạng": (len(out["weights"]) - np.argsort(np.argsort(out["weights"]))),
            }).sort_values("Xếp hạng"), hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 3. DEMATEL
# ==============================================================================

with tab3:
    st.markdown("## 🔗 DEMATEL: Ma trận quan hệ trực tiếp")
    codes = S("codes"); n = S("n")

    st.markdown("""
Nhập mức độ **rào cản ở hàng *i* ảnh hưởng trực tiếp lên rào cản ở cột *j***, theo thang:

| Điểm | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Mức ảnh hưởng | Không | Rất thấp | Thấp | Cao | Rất cao |

Có thể nhập số thập phân nếu đây là giá trị trung bình của nhiều chuyên gia. Đường chéo luôn = 0.
    """)

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        if st.button("🔄 Xoá trắng ma trận", use_container_width=True):
            st.session_state["Z"] = np.zeros((n, n))
            st.rerun()
    with cB:
        upz = st.file_uploader("Nhập ma trận từ CSV/Excel", type=["csv", "xlsx"], key="up_z")
        if upz is not None:
            try:
                dz = pd.read_csv(upz, index_col=0) if upz.name.endswith(".csv") \
                    else pd.read_excel(upz, index_col=0)
                Zu = np.array(dz.values, dtype=float)
                if Zu.shape != (n, n):
                    st.error(f"Ma trận phải có kích thước {n}×{n}, file đang là {Zu.shape}.")
                else:
                    np.fill_diagonal(Zu, 0.0)
                    st.session_state["Z"] = Zu
                    st.success("Đã nạp ma trận.")
                    st.rerun()
            except ImportError:
                st.error("Máy chủ chưa cài gói đọc Excel. Hãy lưu file sang định dạng CSV "
                         "rồi tải lên lại, hoặc thêm dòng `openpyxl>=3.1` vào requirements.txt.")
            except Exception as e:
                st.error(f"Không đọc được file: {e}")
    with cC:
        buf = io.StringIO()
        pd.DataFrame(S("Z"), index=codes, columns=codes).to_csv(buf)
        st.download_button("⬇️ Tải khung ma trận (CSV)", buf.getvalue(),
                           "ma_tran_Z.csv", "text/csv", use_container_width=True)

    Zc = S("Z").copy()
    if Zc.shape != (n, n):
        Zc = np.zeros((n, n))
    np.fill_diagonal(Zc, 0.0)

    dfz = pd.DataFrame(Zc, index=codes, columns=codes)
    z_ed = st.data_editor(
        dfz, use_container_width=True, key="ed_z",
        column_config={c: st.column_config.NumberColumn(
            c, min_value=0.0, max_value=4.0, step=0.1, format="%.1f", width="small")
            for c in codes},
    )
    Znew = np.array(z_ed, dtype=float)
    np.fill_diagonal(Znew, 0.0)
    st.session_state["Z"] = Znew

    if Znew.sum() == 0:
        st.info("Ma trận đang trống. Hãy nhập dữ liệu hoặc nạp bộ dữ liệu mẫu ở thanh bên.")
    else:
        st.markdown("---")
        try:
            dm = dematel_pipeline(Znew, S("weights"))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Hệ số chuẩn hoá s", f"{dm['s']:.2f}")
            k2.metric("max(T)", f"{dm['T'].max():.4f}")
            k3.metric("Nhóm nguyên nhân", f"{int((dm['relation'] > 0).sum())} rào cản")
            k4.metric("Nhóm hệ quả", f"{int((dm['relation'] < 0).sum())} rào cản")

            st.markdown("##### Ma trận quan hệ tổng **T = N(I-N)⁻¹ · (n·wⱼ)**")
            st.caption("Trọng số BWM được nhân theo **cột**, tức ảnh hưởng đi vào một rào cản "
                       "được khuếch đại theo tầm quan trọng của chính rào cản đó.")
            st.plotly_chart(matrix_heatmap(dm["T"], codes, "Ma trận quan hệ tổng T"),
                            use_container_width=True)

            st.markdown("##### Chỉ số DEMATEL")
            st.dataframe(pd.DataFrame({
                "Mã": codes, "Tên": S("names"),
                "R (phát)": np.round(dm["R"], 4),
                "C (nhận)": np.round(dm["C"], 4),
                "R+C (độ nổi bật)": np.round(dm["prominence"], 4),
                "R-C (quan hệ)": np.round(dm["relation"], 4),
                "Nhóm": ["Nguyên nhân" if v > 0 else "Hệ quả" for v in dm["relation"]],
            }).sort_values("R+C (độ nổi bật)", ascending=False),
                hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi tính DEMATEL: {e}")


# ==============================================================================
# TAB 4. TOI UU NGUONG ALPHA
# ==============================================================================

with tab4:
    st.markdown("## 🦅 Secretary Bird: Tối ưu ngưỡng α")
    st.markdown("""
Thuật toán dò tìm giá trị **α\\*** làm cực đại chỉ số **CSI(α)**, tức là điểm cân bằng tốt nhất
giữa *cường độ nhân quả trên mỗi mũi tên* và *chiều sâu phân tầng của cấu trúc ISM*.
    """)

    with st.expander("🔍 Cơ chế thuật toán Secretary Bird (Fu và cộng sự, 2024)"):
        st.markdown(r"""
**Giai đoạn 1, Săn mồi** (3 pha theo tiến độ $t/T$):

| Pha | Điều kiện | Công thức cập nhật |
|---|---|---|
| 1. Tìm mồi | $t < T/3$ | $X_{new} = X_i + (X_{r1} - X_{r2})\cdot R_1$ |
| 2. Tiêu hao mồi | $T/3 \le t < 2T/3$ | $X_{new} = X_{best} + e^{-(t/T)^4}(R_B - 0.5)(X_{best} - X_i)$ |
| 3. Tấn công | $t \ge 2T/3$ | $X_{new} = X_{best} + (1 - t/T)^{2t/T} X_i \cdot \text{Lévy}$ |

**Giai đoạn 2, Trốn kẻ thù** (chọn ngẫu nhiên 1 trong 2):

- **C₁ Nguỵ trang:** $X_{new} = X_{best} + (2R - 1)(1 - t/T)X_i$
- **C₂ Bỏ chạy:** $X_{new} = X_i + R_2 (X_{rand} - K X_i)$, với $K \in \{1, 2\}$

Chấp nhận tham lam theo từng cá thể. Do CSI là hàm bậc thang, nghiệm cuối được **chiếu về
giá trị ứng viên chính xác** (các phần tử phân biệt ngoài đường chéo của T) để đảm bảo tái lập.
        """)

    ready = S("Z").sum() > 0 and len(S("weights")) == S("n")
    if not ready:
        st.warning("Cần nhập ma trận quan hệ trực tiếp ở tab **DEMATEL** trước khi chạy tối ưu.")
    else:
        if st.button("🚀 Chạy Secretary Bird Optimization", type="primary"):
            bar = st.progress(0.0, text="Đang khởi tạo quần thể…")
            try:
                res = run_full_pipeline(
                    S("Z"), S("weights"), iterations=int(iters), pop_size=int(pop),
                    seed=int(seed),
                    progress_cb=lambda p: bar.progress(
                        min(p, 1.0), text=f"Đang tối ưu… {p*100:.0f}%"),
                )
                bar.progress(1.0, text="Hoàn tất.")
                st.session_state["results"] = res
                st.success("✅ Đã tìm được ngưỡng tối ưu. Xem chi tiết ở tab **Kết quả & Diễn giải**.")
            except Exception as e:
                bar.empty()
                st.error(f"Lỗi khi chạy tối ưu: {e}")

        res = S("results")
        if res:
            sb, ism_r, ctx = res["sbo"], res["ism"], res["ctx"]
            st.markdown("---")
            m = st.columns(5)
            m[0].metric("Ngưỡng tối ưu α*", f"{sb['alpha']:.4f}")
            m[1].metric("CSI(α*)", f"{sb['CSI']:.4f}")
            m[2].metric("Số mũi tên giữ lại", f"{sb['n_arrows']}")
            m[3].metric("Số tầng ISM", f"{ism_r['n_levels']}")
            m[4].metric("Cường độ nhân quả CS", f"{sb['CS']:.4f}")

            g1, g2 = st.columns(2)
            with g1:
                grid, vals = csi_profile(ctx)
                st.plotly_chart(csi_landscape(grid, vals, sb["alpha"], sb["CSI"]),
                                use_container_width=True)
            with g2:
                if sb["history"]:
                    st.plotly_chart(convergence_plot(sb["history"]),
                                    use_container_width=True)

            st.caption(f"Miền tìm kiếm α ∈ [{ctx.alpha_low:.4f}, {ctx.alpha_high:.4f}] · "
                       f"Tử số cố định S = {ctx.S:.4f} · "
                       f"CS ∈ [{ctx.cs_min:.4f}, {ctx.cs_max:.4f}] · "
                       f"L ∈ [{ctx.l_min:.0f}, {ctx.l_max:.0f}]")


# ==============================================================================
# TAB 5. KET QUA
# ==============================================================================

with tab5:
    res = S("results")
    if not res:
        st.info("Chưa có kết quả. Hãy hoàn tất các bước và chạy tối ưu ở tab **Tối ưu ngưỡng α**.")
    else:
        codes, names, n = S("codes"), S("names"), S("n")
        dm, sb, ism_r = res["dematel"], res["sbo"], res["ism"]
        prom, rel = dm["prominence"], dm["relation"]
        levels, part, mm = ism_r["levels"], ism_r["partition"], ism_r["micmac"]

        top_imp = int(np.argmax(prom))
        top_cause = int(np.argmax(rel))
        top_effect = int(np.argmin(rel))
        root_ids = levels[max(levels)]
        surface_ids = levels[1]

        st.markdown("## 📊 Kết quả phân tích")
        if S_get("random_title"):
            st.info(f"🎲 **Tình huống demo tự sinh:** {S_get('random_title')} "
                    f"(mã tình huống #{S_get('random_seed')}). Dữ liệu này do hệ thống dựng "
                    "ngẫu nhiên để minh hoạ đầu ra, không phải số liệu khảo sát thật. "
                    "Bấm lại nút ở thanh bên để xem một tình huống khác.")

        # ---------- Phát hiện chính ----------
        st.markdown("### 🎯 Những phát hiện chính")
        f1, f2 = st.columns(2)
        with f1:
            st.markdown(f"""
<div class="keyfind">
<b>🔴 Nguyên nhân gốc (Tầng {max(levels)}, đáy hệ thống)</b><br>
{", ".join(f"<b>{codes[i]}</b>: {names[i]}" for i in root_ids)}<br>
<span style="color:#6b7280;font-size:.9rem">Đây là điểm khởi phát của chuỗi nhân quả.
Mọi can thiệp nên bắt đầu từ đây; xử lý các tầng trên chỉ giải quyết triệu chứng.</span>
</div>
<div class="keyfind blue">
<b>⭐ Rào cản quan trọng nhất (R+C lớn nhất)</b><br>
<b>{codes[top_imp]}</b>: {names[top_imp]} &nbsp;·&nbsp; R+C = {prom[top_imp]:.3f}<br>
<span style="color:#6b7280;font-size:.9rem">Có mức độ tham gia vào hệ thống cao nhất
(vừa gây vừa chịu ảnh hưởng), nhưng không đồng nghĩa là nguyên nhân gốc.</span>
</div>
""", unsafe_allow_html=True)
        with f2:
            st.markdown(f"""
<div class="keyfind green">
<b>🔵 Tác nhân gây ảnh hưởng mạnh nhất (R-C lớn nhất)</b><br>
<b>{codes[top_cause]}</b>: {names[top_cause]} &nbsp;·&nbsp; R-C = {rel[top_cause]:+.3f}
</div>
<div class="keyfind">
<b>🎈 Rào cản mang tính triệu chứng nhất (R-C nhỏ nhất)</b><br>
<b>{codes[top_effect]}</b>: {names[top_effect]} &nbsp;·&nbsp; R-C = {rel[top_effect]:+.3f}<br>
<span style="color:#6b7280;font-size:.9rem">Chủ yếu là kết quả của các rào cản khác,
tác động trực tiếp vào đây thường kém hiệu quả.</span>
</div>
""", unsafe_allow_html=True)

        k = st.columns(5)
        k[0].metric("α* tối ưu", f"{sb['alpha']:.4f}")
        k[1].metric("CSI(α*)", f"{sb['CSI']:.4f}")
        k[2].metric("Số tầng ISM", f"{ism_r['n_levels']}")
        k[3].metric("Quan hệ giữ lại", f"{ism_r['n_arrows']}")
        k[4].metric("Nguyên nhân gốc", ", ".join(codes[i] for i in root_ids))

        st.markdown("---")

        r0, r1, r2, r3, r4, r5 = st.tabs([
            "🔎 Nhận định", "🗺️ Bản đồ nhân quả", "🏗️ Cấu trúc phân tầng ISM",
            "🎛️ MICMAC", "📋 Bảng chi tiết", "💡 Khuyến nghị"])

        # ---------- Nhận định tự động ----------
        with r0:
            st.markdown("### Đọc kết quả này như thế nào")
            st.caption("Các nhận định dưới đây được sinh tự động từ cấu trúc vừa tính ra. "
                       "Chúng nêu cả điểm mạnh lẫn điểm cần thận trọng của kết quả.")

            insights = interpret_results(res, codes, names, S("weights"))
            tone_style = {
                "good": ("#0EAD69", "#f0fbf6", "✅"),
                "key":  ("#2E86AB", "#f1f8fc", "🎯"),
                "warn": ("#E8730C", "#fff7ee", "⚠️"),
                "info": ("#6b7280", "#f7f8fa", "ℹ️"),
            }
            i1, i2 = st.columns(2)
            for k, item in enumerate(insights):
                color, bg, icon = tone_style.get(item["tone"], tone_style["info"])
                (i1 if k % 2 == 0 else i2).markdown(f"""
<div style="border-left:4px solid {color}; background:{bg}; padding:.9rem 1.15rem;
            border-radius:0 10px 10px 0; margin-bottom:.75rem;">
  <div style="font-weight:600; color:#12304a; margin-bottom:.3rem;">{icon} {item['title']}</div>
  <div style="color:#4b5563; font-size:.93rem; line-height:1.6;">{item['text']}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Tóm tắt một đoạn để đưa vào báo cáo")
            n_cause = int((rel > 0).sum())
            density = ism_r["n_arrows"] / max(n * (n - 1), 1)
            summary_txt = (
                f"Phân tích {n} rào cản bằng framework SB-BDI cho thấy hệ thống có cấu trúc "
                f"{ism_r['n_levels']} tầng với {ism_r['n_arrows']} quan hệ ảnh hưởng đáng kể "
                f"(chiếm {density*100:.0f}% số quan hệ có thể có). Trong đó "
                f"{n_cause} rào cản thuộc nhóm nguyên nhân và {n - n_cause} rào cản thuộc nhóm "
                f"hệ quả. Nguyên nhân gốc của toàn hệ thống là "
                f"{', '.join(f'{codes[i]} ({names[i]})' for i in root_ids)}, nằm ở tầng đáy. "
                f"Rào cản có mức độ liên quan cao nhất là {codes[top_imp]} ({names[top_imp]}) "
                f"với chỉ số nổi bật {prom[top_imp]:.3f}, thuộc tầng {part[top_imp]}. "
                f"Chỉ số chất lượng cấu trúc đạt {sb['CSI']:.3f} trên thang 1, cho thấy điểm "
                f"cắt được lựa chọn là "
                + ("rõ ràng và ổn định" if sb['CSI'] >= 0.85 else
                   "chấp nhận được" if sb['CSI'] >= 0.6 else "chưa thật nổi trội") +
                f". Khuyến nghị tập trung nguồn lực vào tầng {max(levels)} trước, "
                f"sử dụng {', '.join(codes[i] for i in surface_ids)} làm chỉ báo đo lường "
                f"hiệu quả can thiệp."
            )
            st.text_area("Chọn toàn bộ rồi sao chép", summary_txt, height=190,
                         label_visibility="collapsed")

        # ---------- Bản đồ nhân quả ----------
        with r1:
            show_arr = st.checkbox("Hiện mũi tên quan hệ vượt ngưỡng α*", value=True)
            st.plotly_chart(
                causal_map(codes, names, prom, rel, dm["T"], sb["alpha"], show_arr),
                use_container_width=True)
            st.markdown("""
**Cách đọc biểu đồ**

- **Trục ngang (R+C)**, độ nổi bật: rào cản càng nằm bên phải càng gắn kết chặt với toàn hệ thống.
- **Trục dọc (R-C)**, vai trò nhân quả: **trên trục 0** là nhóm *nguyên nhân* (gây ảnh hưởng nhiều hơn chịu),
  **dưới trục 0** là nhóm *hệ quả*.
- **Góc phải trên, Tác nhân cốt lõi:** ưu tiên can thiệp cao nhất (vừa quan trọng vừa dẫn dắt).
- **Góc phải dưới, Hệ quả cốt lõi:** chỉ số đo lường tốt để theo dõi tiến triển, không nên can thiệp trực tiếp.
            """)
            c1, c2 = st.columns(2)
            c1.plotly_chart(prominence_bar(codes, prom, rel), use_container_width=True)
            with c2:
                st.plotly_chart(matrix_heatmap(
                    ism_r["binary"].astype(float), codes,
                    f"Ma trận nhị phân tại α* = {sb['alpha']:.4f}",
                    colorscale="Greys", zmin=0, zmax=1, text_fmt=".0f"),
                    use_container_width=True)

        # ---------- ISM ----------
        with r2:
            cc1, cc2 = st.columns([1.5, 1])
            with cc1:
                st.markdown("#### Sơ đồ cấu trúc phân tầng ISM")
                st.graphviz_chart(
                    ism_dot(codes, names, levels, ism_r["binary"], part),
                    use_container_width=True)
                st.caption("Mũi tên hướng từ dưới lên: nguyên nhân ở tầng thấp thúc đẩy hệ quả ở tầng cao.")
            with cc2:
                st.markdown("#### Phân bố theo tầng")
                for lvl in sorted(levels, reverse=True):
                    role = ("🔴 **Nguyên nhân gốc**" if lvl == max(levels)
                            else "🎈 **Hệ quả / triệu chứng**" if lvl == 1
                            else "🔗 **Trung gian**")
                    st.markdown(
                        f"**Tầng {lvl}**: {role}\n\n"
                        + "\n".join(f"- `{codes[i]}` {names[i]}" for i in levels[lvl]))
                    st.markdown("")

            st.markdown("#### Ma trận khả đạt (sau bao đóng bắc cầu Warshall)")
            st.caption("Ô = 1 nghĩa là rào cản ở hàng có thể tác động tới rào cản ở cột "
                       "qua một hoặc nhiều bước trung gian.")
            st.plotly_chart(matrix_heatmap(
                ism_r["reach"].astype(float), codes, "Ma trận khả đạt cuối cùng",
                colorscale="Purples", zmin=0, zmax=1, text_fmt=".0f"),
                use_container_width=True)

        # ---------- MICMAC ----------
        with r3:
            split_lbl = st.radio(
                "Cách xác định đường chia 4 góc phần tư",
                ["Thích ứng, lấy trung điểm dải giá trị quan sát (khuyến nghị)",
                 "Cổ điển, chia đôi tại n/2"],
                horizontal=True, key="micmac_split")
            mode = "classic" if split_lbl.startswith("Cổ điển") else "adaptive"
            mm = micmac(ism_r["reach"], mode)
            if mode == "classic" and len(set(mm["classification"])) == 1:
                st.warning("⚠️ Với ngưỡng α\\* tối ưu, đồ thị khá thưa nên mốc n/2 dồn toàn bộ "
                           "rào cản vào một nhóm. Hãy dùng cách chia **thích ứng** để đọc được cấu trúc.")

            cc1, cc2 = st.columns([1.4, 1])
            with cc1:
                st.plotly_chart(micmac_plot(codes, names, mm, n), use_container_width=True)
            with cc2:
                st.markdown("""
#### Ý nghĩa 4 nhóm

**I. Tự trị (Autonomous)**: sức dẫn dắt thấp, phụ thuộc thấp.
Gần như tách rời hệ thống, ưu tiên xử lý thấp.

**II. Độc lập / Dẫn dắt (Independent)**: dẫn dắt mạnh, ít bị phụ thuộc.
**Đòn bẩy chính sách mạnh nhất**, nên tập trung nguồn lực vào nhóm này.

**III. Liên kết (Linkage)**: vừa dẫn dắt mạnh vừa phụ thuộc mạnh.
Bất ổn: mọi tác động vào nhóm này đều dội ngược lại. Cần theo dõi sát.

**IV. Phụ thuộc (Dependent)**: chủ yếu chịu ảnh hưởng.
Là **chỉ báo kết quả** để đo hiệu quả can thiệp.
                """)
            st.dataframe(pd.DataFrame({
                "Mã": codes, "Tên": names,
                "Sức dẫn dắt": mm["driving_power"],
                "Mức phụ thuộc": mm["dependence"],
                "Nhóm MICMAC": mm["classification"],
                "Tầng ISM": part,
            }).sort_values(["Sức dẫn dắt", "Mức phụ thuộc"], ascending=[False, True]),
                hide_index=True, use_container_width=True)

        # ---------- Bảng chi tiết ----------
        with r4:
            summary = pd.DataFrame({
                "Mã": codes, "Tên rào cản": names, "Nhóm": S("dims"),
                "Trọng số BWM": np.round(S("weights"), 4),
                "R (phát)": np.round(dm["R"], 4),
                "C (nhận)": np.round(dm["C"], 4),
                "R+C": np.round(prom, 4),
                "R-C": np.round(rel, 4),
                "Vai trò": ["Nguyên nhân" if v > 0 else "Hệ quả" for v in rel],
                "Tầng ISM": part,
                "Sức dẫn dắt": mm["driving_power"],
                "Mức phụ thuộc": mm["dependence"],
                "Nhóm MICMAC": mm["classification"],
            })
            st.dataframe(summary.sort_values("R+C", ascending=False),
                         hide_index=True, use_container_width=True, height=460)

            st.markdown("##### Các quan hệ ảnh hưởng được giữ lại (T ≥ α*)")
            pairs = [(codes[i], codes[j], dm["T"][i, j])
                     for i in range(n) for j in range(n)
                     if i != j and dm["T"][i, j] >= sb["alpha"]]
            pairs.sort(key=lambda x: -x[2])
            st.dataframe(pd.DataFrame(pairs, columns=["Từ (nguyên nhân)", "Đến (chịu tác động)",
                                                      "Cường độ T"]).round(4),
                         hide_index=True, use_container_width=True, height=300)

            # ---------- Xuất kết quả ----------
            # Dùng CSV nén trong ZIP thay vì .xlsx: chỉ cần thư viện chuẩn của Python,
            # không phụ thuộc openpyxl nên không bao giờ lỗi khi triển khai lên máy chủ.
            def _csv(df, index=False):
                return df.to_csv(index=index).encode("utf-8-sig")   # BOM để Excel đọc đúng tiếng Việt

            df_params = pd.DataFrame({
                "Chi so": ["Nguong alpha*", "CSI", "So mui ten giu lai",
                           "So tang ISM", "Tu so co dinh S", "So rao can"],
                "Gia tri": [round(sb["alpha"], 6), round(sb["CSI"], 6), ism_r["n_arrows"],
                            ism_r["n_levels"], round(res["ctx"].S, 6), n],
            })
            df_pairs = pd.DataFrame(pairs, columns=["Tu (nguyen nhan)",
                                                    "Den (chiu tac dong)", "Cuong do T"]).round(4)

            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("1_tong_hop.csv", _csv(summary))
                zf.writestr("2_tham_so.csv", _csv(df_params))
                zf.writestr("3_quan_he_giu_lai.csv", _csv(df_pairs))
                zf.writestr("4_ma_tran_quan_he_tong_T.csv",
                            _csv(pd.DataFrame(dm["T"].round(4), index=codes, columns=codes), True))
                zf.writestr("5_ma_tran_kha_dat.csv",
                            _csv(pd.DataFrame(ism_r["reach"], index=codes, columns=codes), True))
                zf.writestr("6_ma_tran_anh_huong_truc_tiep_Z.csv",
                            _csv(pd.DataFrame(dm["Z"], index=codes, columns=codes), True))

            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Tải toàn bộ kết quả (ZIP nhiều file CSV)",
                               zbuf.getvalue(), "ket_qua_SB-BDI.zip", "application/zip",
                               type="primary", use_container_width=True)
            d2.download_button("⬇️ Chỉ tải bảng tổng hợp (CSV)",
                               _csv(summary), "tong_hop_SB-BDI.csv", "text/csv",
                               use_container_width=True)
            st.caption("File CSV có sẵn dấu BOM nên mở bằng Excel hiển thị đúng tiếng Việt. "
                       "Trong Excel có thể gộp các file lại thành nhiều sheet nếu cần.")

        # ---------- Khuyến nghị ----------
        with r5:
            st.markdown("### 💡 Thứ tự can thiệp đề xuất")
            st.caption("Suy ra trực tiếp từ cấu trúc phân tầng ISM: xử lý từ đáy lên đỉnh.")

            for order, lvl in enumerate(sorted(levels, reverse=True), start=1):
                ids = levels[lvl]
                if lvl == max(levels):
                    tag, color = "Ưu tiên 1: Can thiệp gốc", "#C73E1D"
                    note = ("Đây là điểm khởi phát của toàn bộ chuỗi rào cản. Nguồn lực đầu tư "
                            "vào tầng này tạo hiệu ứng lan toả xuống mọi tầng phía trên.")
                elif lvl == 1:
                    tag, color = f"Ưu tiên {order}: Theo dõi kết quả", "#3A86FF"
                    note = ("Chủ yếu là hệ quả. Dùng làm chỉ báo đo lường hiệu quả của các can thiệp "
                            "ở tầng dưới, thay vì can thiệp trực tiếp.")
                else:
                    tag, color = f"Ưu tiên {order}: Xử lý trung gian", "#E8730C"
                    note = ("Cầu nối truyền dẫn ảnh hưởng. Xử lý sau khi tầng gốc đã chuyển biến "
                            "để tránh lãng phí nguồn lực.")

                items = "".join(
                    f"<li><b>{codes[i]}</b>: {names[i]} "
                    f"<span style='color:#8a93a0'>(w = {S('weights')[i]:.3f}, "
                    f"R+C = {prom[i]:.2f}, {mm['classification'][i]})</span></li>"
                    for i in ids)
                st.markdown(f"""
<div style="border-left:4px solid {color}; background:#fbfcfd; padding:.9rem 1.2rem;
            border-radius:0 10px 10px 0; margin-bottom:.8rem;">
  <div style="color:{color}; font-weight:600; font-size:.86rem; letter-spacing:.02em;
              text-transform:uppercase;">{tag} · Tầng {lvl}</div>
  <ul style="margin:.5rem 0 .5rem 1.1rem; color:#2b3440;">{items}</ul>
  <div style="color:#6b7280; font-size:.9rem;">{note}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📝 Diễn giải tổng hợp")
            leverage = [codes[i] for i in range(n)
                        if mm["classification"][i].startswith("Độc lập")]
            linkage = [codes[i] for i in range(n)
                       if mm["classification"][i].startswith("Liên kết")]
            st.markdown(f"""
Với **{n} rào cản** được phân tích, thuật toán Secretary Bird xác định ngưỡng nội sinh
**α\\* = {sb['alpha']:.4f}** (CSI = {sb['CSI']:.4f}), giữ lại **{ism_r['n_arrows']} quan hệ ảnh hưởng**
và cho ra cấu trúc **{ism_r['n_levels']} tầng**.

- **{", ".join(codes[i] for i in root_ids)}** nằm ở tầng đáy, chính là **nguyên nhân gốc** của hệ thống.
- **{codes[top_imp]}** có độ nổi bật cao nhất (R+C = {prom[top_imp]:.3f}) nhưng thuộc tầng
  {part[top_imp]}; điều này cho thấy *mức độ quan trọng không đồng nghĩa với vị trí căn nguyên*
  Đây là kết luận mà nếu chỉ dùng BWM hoặc chỉ dùng DEMATEL sẽ không phát hiện được.
- Nhóm đòn bẩy chính sách (MICMAC, nhóm Độc lập/Dẫn dắt): **{", ".join(leverage) if leverage else "không có"}**.
- Nhóm cần theo dõi sát vì tính bất ổn (MICMAC, nhóm Liên kết): **{", ".join(linkage) if linkage else "không có"}**.
- Tầng 1 gồm **{", ".join(codes[i] for i in surface_ids)}**, nên dùng làm **chỉ báo kết quả**
  để đo hiệu quả của các can thiệp phía dưới.
            """)

st.markdown("---")
st.caption(
    "SB-BDI Framework · Tôn Nguyễn Trọng Hiển · "
    "ORCID [0000-0002-6970-0799](https://orcid.org/0000-0002-6970-0799) · "
    "BWM (Rezaei 2015, 2016) · DEMATEL (Gabus & Fontela 1972) · "
    "ISM (Warfield 1974) · SBOA (Fu và cộng sự, 2024)")
