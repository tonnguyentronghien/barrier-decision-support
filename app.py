import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import graphviz
from scipy.optimize import linprog
import math
import time

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Hệ Hỗ Trợ Ra Quyết Định Đa Tiêu Chí (SB-BDI)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Framework Hỗ Trợ Ra Quyết Định Đa Tiêu Chí (SB-BDI)")
st.markdown("""
Ứng dụng phân tích cấu trúc rào cản (Barrier Analysis) dựa trên sự tích hợp của **BWM (Best-Worst Method)**, **DEMATEL**, và tối ưu hóa phân tầng **ISM** bằng thuật toán bầy đàn **Secretary Bird Optimization (SBOA)**.
""")

# =====================================================================
# CÁC HÀM THUẬT TOÁN CỐT LÕI (CORE ALGORITHMS)
# =====================================================================

def solve_bwm(bo_vector, ow_vector, best_idx, worst_idx, n):
    """Tính trọng số BWM tối ưu bằng Linear Programming"""
    nv = n + 1
    A_l, b_l = [], []
    for j in range(n):
        if j != best_idx:
            r = np.zeros(nv); r[best_idx] = 1.0; r[j] = -float(bo_vector[j]); r[-1] = -1.0; A_l.append(r); b_l.append(0.0)
            r = np.zeros(nv); r[best_idx] = -1.0; r[j] = float(bo_vector[j]); r[-1] = -1.0; A_l.append(r); b_l.append(0.0)
    for j in range(n):
        if j != worst_idx:
            r = np.zeros(nv); r[j] = 1.0; r[worst_idx] = -float(ow_vector[j]); r[-1] = -1.0; A_l.append(r); b_l.append(0.0)
            r = np.zeros(nv); r[j] = -1.0; r[worst_idx] = float(ow_vector[j]); r[-1] = -1.0; A_l.append(r); b_l.append(0.0)
            
    A_eq = np.zeros((1, nv)); A_eq[0, :n] = 1.0
    c = np.zeros(nv); c[-1] = 1.0
    res = linprog(c, A_ub=np.array(A_l), b_ub=np.array(b_l),
                  A_eq=A_eq, b_eq=np.array([1.0]),
                  bounds=[(0, None)] * nv, method='highs')
    if res.success:
        weights = res.x[:n]
        return weights / np.sum(weights)
    else:
        return np.ones(n) / n

def compute_dematel_total_matrix(Z, W):
    """Tích hợp trọng số BWM và tính ma trận ảnh hưởng tổng hợp T"""
    n = Z.shape[0]
    s = max(np.max(Z.sum(axis=1)), np.max(Z.sum(axis=0)))
    if s == 0: s = 1.0
    N = Z / s
    I = np.eye(n)
    inv_matrix = np.linalg.inv(I - N)
    T_unweighted = N @ inv_matrix
    # Column-wise scaling theo trọng số BWM
    T_weighted = T_unweighted * (W[None, :] * n)
    return T_weighted

def ism_partition(T, alpha):
    """Phân tầng ISM sử dụng Warshall transitive closure"""
    n = T.shape[0]
    binary = (T >= alpha).astype(int)
    np.fill_diagonal(binary, 1)
    reach = binary.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if reach[i, k] and reach[k, j]:
                    reach[i, j] = 1
                    
    partition = -np.ones(n, dtype=int)
    remaining = set(range(n))
    level = 0
    while remaining:
        current = []
        for i in remaining:
            Rset = {j for j in remaining if reach[i, j]}
            Aset = {j for j in remaining if reach[j, i]}
            if Rset.issubset(Aset):
                current.append(i)
        if not current:
            break
        for node in current:
            partition[node] = level
            remaining.remove(node)
        level += 1
    return (level if level > 0 else 1), partition

def compute_n_arrows(T, alpha):
    n = T.shape[0]
    binary = (T >= alpha).astype(int)
    np.fill_diagonal(binary, 1)
    return int(np.sum(binary) - n)

def csi_setup(T):
    n = T.shape[0]
    R = T.sum(axis=1)
    C = T.sum(axis=0)
    rel = R - C
    cause = np.abs(np.mean(rel[rel > 0])) if np.any(rel > 0) else 0.0
    effect = np.abs(np.mean(rel[rel < 0])) if np.any(rel < 0) else 0.0
    S = cause + effect
    
    off_diag = T[~np.eye(n, dtype=bool)]
    pos = off_diag[off_diag > 0]
    if len(pos) == 0:
        return None
    
    alphas_range = np.linspace(pos.min(), pos.max(), 100)
    cs_vals, level_vals = [], []
    for a in alphas_range:
        n_arr = compute_n_arrows(T, a)
        if n_arr < n:
            continue
        cs_vals.append(S / n_arr)
        lvl, _ = ism_partition(T, a)
        level_vals.append(lvl)
        
    if not cs_vals:
        return None
    return S, min(cs_vals), max(cs_vals), min(level_vals), max(level_vals), float(pos.min()), float(pos.max())

def evaluate_csi(alpha, T, cfg):
    S, cs_min, cs_max, l_min, l_max, _, _ = cfg
    n = T.shape[0]
    n_arr = compute_n_arrows(T, alpha)
    if n_arr < n:
        return -1e9
    cs = S / n_arr
    lvl, _ = ism_partition(T, alpha)
    if cs_max == cs_min or l_max == l_min:
        return 0.0
    cs_tilde = max(0.0, min(1.0, (cs - cs_min) / (cs_max - cs_min + 1e-12)))
    l_tilde = max(0.0, min(1.0, (lvl - l_min) / (l_max - l_min + 1e-12)))
    return math.sqrt(cs_tilde * l_tilde)

def levy_flight(beta=1.5):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    return u / (abs(v) ** (1 / beta) + 1e-12)

def run_sboa(T, cfg, iterations=100, pop_size=30):
    _, _, _, _, _, low, high = cfg
    population = np.random.uniform(low, high, pop_size)
    fitness = np.array([evaluate_csi(a, T, cfg) for a in population])
    best_idx = int(np.argmax(fitness))
    best_alpha, best_csi = population[best_idx], fitness[best_idx]
    
    for t in range(1, iterations + 1):
        for i in range(pop_size):
            if t < iterations / 3.0:
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                new_alpha = population[i] + (population[r1] - population[r2]) * np.random.rand()
            elif t < 2 * iterations / 3.0:
                RB = np.random.normal(0, 1)
                new_alpha = best_alpha + np.exp(-((t / iterations) ** 4)) * (RB - 0.5) * (best_alpha - population[i])
            else:
                RL = levy_flight()
                new_alpha = best_alpha + ((1 - t / iterations) ** (2 * t / iterations)) * population[i] * RL
            
            new_alpha = float(np.clip(new_alpha, low, high))
            new_fit = evaluate_csi(new_alpha, T, cfg)
            if new_fit > fitness[i]:
                population[i], fitness[i] = new_alpha, new_fit
                if new_fit > best_csi:
                    best_alpha, best_csi = new_alpha, new_fit

        for i in range(pop_size):
            if np.random.rand() < 0.5:
                Rscalar = np.random.rand()
                new_alpha = best_alpha + (2 * Rscalar - 1) * (1 - t / iterations) * population[i]
            else:
                K = np.round(1 + np.random.rand())
                rr = np.random.choice(pop_size)
                new_alpha = population[i] + np.random.normal(0, 1) * (population[rr] - K * population[i])
                
            new_alpha = float(np.clip(new_alpha, low, high))
            new_fit = evaluate_csi(new_alpha, T, cfg)
            if new_fit > fitness[i]:
                population[i], fitness[i] = new_alpha, new_fit
                if new_fit > best_csi:
                    best_alpha, best_csi = new_alpha, new_fit

    return best_alpha, best_csi

# =====================================================================
# GIAO DIỆN NGƯỜI DÙNG (USER INTERFACE)
# =====================================================================

# Khởi tạo trạng thái danh sách rào cản mặc định
if "barriers" not in st.session_state:
    st.session_state.barriers = ["IN1", "IN2", "IN3", "IN4", "IN5", "EC1", "SO1", "SO2", "SO3", "TE1", "TE2", "TE3"]

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Cấu hình Rào cản", 
    "2️⃣ Đánh giá BWM", 
    "3️⃣ Ma trận DEMATEL", 
    "4️⃣ Kết quả & Phân tầng ISM"
])

# --- TAB 1: CẤU HÌNH SỐ LƯỢNG RÀO CẢN ---
with tab1:
    st.subheader("Cấu hình danh mục Rào cản (Barriers)")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_barriers = st.number_input("Số lượng rào cản cần phân tích:", min_value=3, max_value=30, value=len(st.session_state.barriers), step=1)
    
    current_list = st.session_state.barriers
    if n_barriers > len(current_list):
        current_list += [f"B{i+1}" for i in range(len(current_list), n_barriers)]
    elif n_barriers < len(current_list):
        current_list = current_list[:n_barriers]
        
    barriers_text = st.text_area("Danh sách mã rào cản (phân cách bằng dấu phẩy):", value=", ".join(current_list))
    if st.button("Cập nhật danh sách", type="primary"):
        updated_list = [b.strip() for b in barriers_text.split(",") if b.strip()]
        if len(updated_list) == n_barriers:
            st.session_state.barriers = updated_list
            st.success(f"Đã cập nhật danh sách thành công với {len(updated_list)} rào cản!")
        else:
            st.error(f"Số lượng rào cản nhập ({len(updated_list)}) không khớp với số lượng cấu hình ({n_barriers})!")

barriers = st.session_state.barriers
n = len(barriers)

# --- TAB 2: ĐÁNH GIÁ TRỌNG SỐ BWM ---
with tab2:
    st.subheader("Đánh giá Trọng số bằng Phương pháp Best-Worst (BWM)")
    c1, c2 = st.columns(2)
    with c1:
        best_b = st.selectbox("Chọn rào cản quan trọng nhất (Best - B):", barriers, index=2 if "IN3" in barriers else 0)
    with c2:
        worst_b = st.selectbox("Chọn rào cản ít quan trọng nhất (Worst - W):", barriers, index=7 if "SO2" in barriers else min(1, n-1))
        
    st.markdown("##### 1. Đánh giá Best-to-Others (BO)")
    st.caption("Mức độ quan trọng của Best so với các tiêu chí khác (Thang điểm 1 -> 9, với Best so với Best = 1):")
    bo_init = {b: [1 if b == best_b else 3] for b in barriers}
    df_bo = st.data_editor(pd.DataFrame(bo_init, index=["Mức độ (1-9)"]), key="editor_bo")
    
    st.markdown("##### 2. Đánh giá Others-to-Worst (OW)")
    st.caption("Mức độ quan trọng của các tiêu chí khác so với Worst (Thang điểm 1 -> 9, với Worst so với Worst = 1):")
    ow_init = {b: [1 if b == worst_b else 3] for b in barriers}
    df_ow = st.data_editor(pd.DataFrame(ow_init, index=["Mức độ (1-9)"]), key="editor_ow")
    
    best_idx = barriers.index(best_b)
    worst_idx = barriers.index(worst_b)
    bo_vec = df_bo.iloc[0].values.astype(float)
    ow_vec = df_ow.iloc[0].values.astype(float)
    
    bwm_weights = solve_bwm(bo_vec, ow_vec, best_idx, worst_idx, n)
    df_weights = pd.DataFrame({"Rào cản": barriers, "Trọng số tối ưu (W)": np.round(bwm_weights, 4)})
    st.dataframe(df_weights.T, use_container_width=True)

# --- TAB 3: NHẬP LIỆU MA TRẬN DEMATEL ---
with tab3:
    st.subheader("Ma trận ảnh hưởng trực tiếp DEMATEL (Z)")
    st.caption("Thang đo ảnh hưởng: 0: Không ảnh hưởng, 1: Ảnh hưởng thấp, 2: Vừa, 3: Cao, 4: Rất cao.")
    
    # Ma trận mẫu mặc định
    default_z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: default_z[i, j] = 1.0
            
    df_z = st.data_editor(pd.DataFrame(default_z, index=barriers, columns=barriers), key="editor_z")
    Z_matrix = df_z.values.astype(float)

# --- TAB 4: CHẠY THUẬT TOÁN & SHOW KẾT QUẢ ---
with tab4:
    st.subheader("Tối ưu hóa Phân tầng & Hiển thị Kết quả")
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        n_pop = st.number_input("Kích thước quần thể (Population Size):", value=30, min_value=10, max_value=100, step=5)
    with c_p2:
        max_iter = st.number_input("Số vòng lặp tối đa (Max Iterations):", value=100, min_value=20, max_value=300, step=10)
        
    if st.button("🚀 Bắt đầu Phân Tích Hệ Thống", type="primary"):
        with st.spinner("Đang tính toán DEMATEL và tối ưu hóa ngưỡng bằng SBOA..."):
            start_calc = time.perf_counter()
            
            # 1. Tính toán Total-Relation Matrix T
            T_matrix = compute_dematel_total_matrix(Z_matrix, bwm_weights)
            
            # 2. Thiết lập cấu hình hàm mục tiêu CSI và chạy SBOA
            cfg = csi_setup(T_matrix)
            if cfg is None:
                st.error("Ma trận DEMATEL chưa có đủ tương tác có trọng số dương để tối ưu.")
            else:
                opt_alpha, opt_csi = run_sboa(T_matrix, cfg, iterations=max_iter, pop_size=n_pop)
                elapsed = time.perf_counter() - start_calc
                
                levels, partition = ism_partition(T_matrix, opt_alpha)
                n_arrows = compute_n_arrows(T_matrix, opt_alpha)
                
                # 3. Hiển thị thông số tổng hợp
                st.success(f"Phân tích hoàn tất sau {elapsed:.2f} giây!")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ngưỡng tối ưu (Alpha*)", f"{opt_alpha:.4f}")
                m2.metric("Chỉ số CSI đạt được", f"{opt_csi:.4f}")
                m3.metric("Số liên kết giữ lại (Arrows)", f"{n_arrows}")
                m4.metric("Số tầng phân cấp (ISM)", f"{levels} Tầng")
                
                # 4. Bảng tính R, C, R+C, R-C
                R = T_matrix.sum(axis=1)
                C = T_matrix.sum(axis=0)
                df_rel = pd.DataFrame({
                    "Rào cản": barriers,
                    "Influence (R)": np.round(R, 4),
                    "Received (C)": np.round(C, 4),
                    "Prominence (R+C)": np.round(R + C, 4),
                    "Relation (R-C)": np.round(R - C, 4),
                    "Phân loại": ["Nguyên nhân (Cause)" if (r - c) > 0 else "Kết quả (Effect)" for r, c in zip(R, C)],
                    "Tầng ISM": [f"Level {partition[i]+1}" for i in range(n)]
                })
                
                st.markdown("#### 1. Bảng Chỉ số Quan hệ Causal DEMATEL & Phân tầng ISM")
                st.dataframe(df_rel, use_container_width=True)
                
                # 5. Biểu đồ Causal DEMATEL 4 góc phần tư
                st.markdown("#### 2. Biểu đồ Causal Mapping (Prominence vs Relation)")
                fig = px.scatter(
                    df_rel, x="Prominence (R+C)", y="Relation (R-C)", 
                    text="Rào cản", color="Phân loại",
                    color_discrete_map={"Nguyên nhân (Cause)": "#1f77b4", "Kết quả (Effect)": "#d62728"},
                    title="Bản đồ Nguyên nhân - Kết quả (Causal Diagram)"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_traces(textposition='top center', marker=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
                
                # 6. Sơ đồ cây phân tầng ISM bằng Graphviz
                st.markdown("#### 3. Sơ đồ Cấu trúc Phân tầng ISM (Hierarchical Digraph)")
                dot = graphviz.Digraph()
                dot.attr(rankdir="TB", size="8,8")
                
                # Tạo node theo từng tầng
                for lvl in range(levels):
                    with dot.subgraph() as s:
                        s.attr(rank='same')
                        for i in range(n):
                            if partition[i] == lvl:
                                color = "#cfe2f3" if (R[i] - C[i]) > 0 else "#f4cccc"
                                s.node(barriers[i], f"{barriers[i]}\n(Level {lvl+1})", shape="box", style="rounded,filled", fillcolor=color)
                                
                # Thêm liên kết cạnh
                binary_opt = (T_matrix >= opt_alpha).astype(int)
                for i in range(n):
                    for j in range(n):
                        if i != j and binary_opt[i, j] == 1:
                            dot.edge(barriers[i], barriers[j])
                            
                st.graphviz_chart(dot)
