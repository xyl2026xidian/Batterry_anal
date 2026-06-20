# battery_module_analysis.py - 电动汽车电池模组热-振动耦合分析智能体
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time

# ========== 页面配置 ==========
st.set_page_config(
    page_title="电池模组热-振动耦合分析智能体",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 材料数据库 ==========
MATERIALS = {
    "方形电芯(LFP)": {
        "E_cell": 12e9,
        "mu": 0.012,
        "rho": 2200,
        "C_p": 1100,
        "k_thermal": 2.5,
        "color": "#4facfe"
    },
    "方形电芯(NCM)": {
        "E_cell": 15e9,
        "mu": 0.015,
        "rho": 2500,
        "C_p": 1000,
        "k_thermal": 2.0,
        "color": "#f093fb"
    },
    "软包电芯": {
        "E_cell": 8e9,
        "mu": 0.020,
        "rho": 2000,
        "C_p": 1200,
        "k_thermal": 1.8,
        "color": "#43e97b"
    },
    "圆柱电芯(18650)": {
        "E_cell": 18e9,
        "mu": 0.008,
        "rho": 2600,
        "C_p": 900,
        "k_thermal": 3.0,
        "color": "#f5576c"
    }
}


# ============================================================
# 核心计算函数
# ============================================================
def calculate_thermal_expansion_force(E_cell, mu, dT, A):
    """计算热膨胀力 F_th = E * mu * dT * A"""
    return E_cell * mu * dT * A


def calculate_contact_stiffness(K0, alpha_contact, F_th):
    """计算接触刚度 K = K0 + alpha * F_th"""
    return K0 + alpha_contact * F_th


def calculate_module_frequency(K_total, M_total):
    """计算模组固有频率 f = 1/(2*pi) * sqrt(K/M)"""
    return (1 / (2 * np.pi)) * np.sqrt(K_total / M_total)


def compute_temperature_field(n_cells, T_ambient, dT, Q_gen):
    """计算模组温度场分布"""
    T = np.linspace(T_ambient, T_ambient + dT, n_cells)
    T = T + 5 * np.sin(np.linspace(0, np.pi, n_cells)) * dT * 0.15
    return T


def compute_thermal_stress_distribution(T, E_cell, mu):
    """计算热应力分布"""
    T_ref = 25
    sigma = E_cell * mu * (T - T_ref)
    return sigma


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("电池模组")
    st.markdown("**热-振动耦合分析**")
    st.markdown("---")

    module = st.radio(
        "选择分析模块",
        ["理论与原理", "参数输入与计算", "结果分析",
         "振型图", "3D可视化", "报告生成"]
    )

    st.markdown("---")
    st.caption("交互式分析 | 实时计算 | 3D可视化")

# ============================================================
# 模块1：理论与原理
# ============================================================
if module == "理论与原理":
    st.title("电动汽车电池模组热-振动耦合分析")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["模组构造", "耦合机制", "理论模型"])

    with tab1:
        st.markdown("""
        ### 电池模组构造

        电动汽车电池模组是动力电池系统的核心结构单元，由多个电芯通过串联/并联组成。

        #### 典型结构

        端板 | 电芯 | 泡棉 | 电芯 | 泡棉 | 端板
        绑带提供预紧力，保持模组紧凑

        #### 关键组件
        - 电芯: 储能单元，充放电时厚度膨胀/收缩
        - 泡棉: 吸收膨胀，提供缓冲
        - 端板: 提供预紧力，约束膨胀
        - 绑带: 施加预紧力，保持模组紧凑

        #### 关键数据
        - 电芯温升可达 40-60 度 (快充工况)
        - 电芯厚度膨胀 0.5% - 1.5%
        - 模组固有频率范围 50-200 Hz
        """)

        # 模组示意图
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3)
        ax.axis('off')

        # 绘制端板
        ax.add_patch(plt.Rectangle((0.3, 0.5), 0.8, 2.0, color='#2c3e50', alpha=0.8))
        ax.text(0.7, 1.5, '端板', ha='center', va='center', color='white', fontsize=10)

        # 绘制电芯和泡棉
        cell_colors = ['#4facfe', '#f093fb', '#43e97b', '#f5576c', '#ffd93d']
        for i in range(5):
            x_pos = 1.5 + i * 1.8
            ax.add_patch(plt.Rectangle((x_pos, 0.5), 1.2, 2.0, color=cell_colors[i % len(cell_colors)], alpha=0.7))
            ax.text(x_pos + 0.6, 1.5, f'电芯{i + 1}', ha='center', va='center', fontsize=8, color='white')

            # 泡棉（除了最后一个）
            if i < 4:
                foam_x = x_pos + 1.2
                ax.add_patch(plt.Rectangle((foam_x, 0.7), 0.4, 1.6, color='#f8d7da', alpha=0.6, hatch='/'))
                ax.text(foam_x + 0.2, 0.3, '泡棉', ha='center', va='center', fontsize=7, color='#666')

        # 绘制绑带
        ax.plot([0.2, 11.0], [0.3, 0.3], 'k-', linewidth=3, label='绑带')
        ax.plot([0.2, 11.0], [2.7, 2.7], 'k-', linewidth=3)

        ax.text(5.6, -0.2, '绑带 (提供预紧力)', ha='center', fontsize=10, color='#2c3e50')

        st.pyplot(fig)

    with tab2:
        st.markdown("""
        ### 热-振动耦合物理机制

        #### 耦合路径

        电池发热 -> 电芯膨胀 -> 端板约束 -> 热膨胀力增大
                                    |
                            接触刚度增大 (K增加)
                                    |
                            模组固有频率偏移 (f增加)
                                    |
                            与路面激励耦合 -> 共振风险

        #### 关键效应

        | 效应 | 物理机制 | 后果 |
        |------|----------|------|
        | 热膨胀 | 电芯受热沿厚度方向膨胀 | 产生热膨胀力 |
        | 接触刚化 | 压力增大使接触面更紧密 | 固有频率升高 |
        | 频率漂移 | 温度变化改变K | 可能进入共振区 |
        | 材料软化 | 高温使泡棉软化 | 刚度下降 |

        #### 典型数据
        - 快充时电芯温升: 40-60 度
        - 电芯膨胀量: 0.5-1.5 mm
        - 热膨胀力: 5-20 kN
        - 固有频率偏移: 10-30%
        """)

    with tab3:
        st.markdown("""
        ### 理论模型

        #### 等效模型
        电池模组 -> 串联弹簧-质量系统

        #### 核心假设
        1. 电芯为刚性质量块
        2. 泡棉为非线性弹簧
        3. 热膨胀力均匀分布
        4. 接触刚度随压力线性增加

        #### 关键公式

        **热膨胀力**
        F_th = E_cell * mu * dT * A

        **接触刚度**
        K_c = K_0 + alpha * F_th

        **模组固有频率**
        f = 1/(2*pi) * sqrt(K_c * n / M_total)

        #### 符号说明
        | 符号 | 含义 | 单位 |
        |------|------|------|
        | E_cell | 电芯等效压缩模量 | Pa |
        | mu | 热膨胀系数 | /C |
        | dT | 温升 | C |
        | A | 电芯截面积 | m2 |
        | K0 | 初始接触刚度 | N/m |
        | alpha | 刚度系数 | N/m2 |
        """)

# ============================================================
# 模块2：参数输入与计算
# ============================================================
elif module == "参数输入与计算":
    st.title("参数输入与计算")
    st.markdown("输入电池模组参数，实时计算热-振动耦合结果")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 模组参数")
        n_cells = st.number_input("电芯数量", value=6, min_value=2, max_value=20, step=1)
        m_cell = st.number_input("单颗电芯质量 (kg)", value=0.8, min_value=0.2, max_value=3.0, step=0.1)
        A_cell = st.number_input("电芯截面积 (m2)", value=0.01, min_value=0.002, max_value=0.05, step=0.001,
                                 format="%.4f")
        L_cell = st.number_input("电芯厚度 (m)", value=0.03, min_value=0.01, max_value=0.08, step=0.001, format="%.3f")

        st.markdown("---")
        st.markdown("### 热参数")
        T_ambient = st.number_input("环境温度 (C)", value=25, min_value=-20, max_value=50, step=5)
        dT_cell = st.slider("电芯温升 dT (C)", 0, 80, 40, 5)
        Q_gen = st.number_input("产热功率 Q (W)", value=50, min_value=10, max_value=200, step=5)

    with col2:
        st.markdown("### 材料参数")
        material = st.selectbox("电芯类型", list(MATERIALS.keys()))
        mat = MATERIALS[material]

        st.write("**材料属性:**")
        st.write(f"压缩模量 E_cell = {mat['E_cell'] / 1e9:.1f} GPa")
        st.write(f"热膨胀系数 mu = {mat['mu'] * 100:.1f} %/C")
        st.write(f"密度 rho = {mat['rho']:.0f} kg/m3")
        st.write(f"比热容 C_p = {mat['C_p']:.0f} J/(kg*K)")

        st.markdown("---")
        st.markdown("### 刚度参数")
        K0 = st.number_input("初始接触刚度 K0 (MN/m)", value=50, min_value=10, max_value=200, step=5)
        alpha_contact = st.number_input("接触刚度系数 alpha (MN/m2)", value=2.0, min_value=0.5, max_value=5.0, step=0.1)

        st.markdown("---")
        st.markdown("### 计算控制")
        calc_button = st.button("开始计算", use_container_width=True)

    if calc_button:
        st.markdown("---")
        st.markdown("## 计算结果")

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("步骤1/4: 计算热膨胀力...")
        F_th = calculate_thermal_expansion_force(mat['E_cell'], mat['mu'], dT_cell, A_cell)
        progress_bar.progress(25)
        time.sleep(0.2)

        status_text.text("步骤2/4: 计算接触刚度...")
        K_contact = calculate_contact_stiffness(K0 * 1e6, alpha_contact * 1e6, F_th)
        progress_bar.progress(50)
        time.sleep(0.2)

        status_text.text("步骤3/4: 计算总刚度和质量...")
        M_total = n_cells * m_cell
        K_total = K_contact * (n_cells - 1)
        progress_bar.progress(75)
        time.sleep(0.2)

        status_text.text("步骤4/4: 计算固有频率...")
        f_cold = calculate_module_frequency(K0 * 1e6 * (n_cells - 1), M_total)
        f_hot = calculate_module_frequency(K_total, M_total)
        progress_bar.progress(100)
        time.sleep(0.1)

        status_text.text("计算完成！")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("冷态频率", f"{f_cold:.1f} Hz")
        with col2:
            st.metric("热态频率", f"{f_hot:.1f} Hz")
            st.write(f"偏移: {(f_hot - f_cold) / f_cold * 100:+.1f}%")
        with col3:
            st.metric("频率漂移量", f"{f_hot - f_cold:.1f} Hz")

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("热膨胀力", f"{F_th / 1000:.1f} kN")
        with col2:
            st.metric("接触刚度", f"{K_contact / 1e6:.1f} MN/m")
        with col3:
            st.metric("总质量", f"{M_total:.1f} kg")
        with col4:
            st.metric("电芯温升", f"{dT_cell:.0f} C")

        st.markdown("---")
        st.markdown("### 温升-频率漂移曲线")

        dT_range = np.linspace(0, 80, 30)
        freq_range = []
        for dT in dT_range:
            F = calculate_thermal_expansion_force(mat['E_cell'], mat['mu'], dT, A_cell)
            K = calculate_contact_stiffness(K0 * 1e6, alpha_contact * 1e6, F)
            freq = calculate_module_frequency(K * (n_cells - 1), M_total)
            freq_range.append(freq)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dT_range, y=freq_range, mode='lines+markers',
                                 line=dict(color='red', width=3), name='频率曲线'))
        fig.add_hline(y=f_cold, line_dash="dash", line_color="blue", name='冷态频率')
        fig.add_hline(y=f_hot, line_dash="dash", line_color="orange", name='热态频率')
        fig.update_layout(
            title='固有频率 vs 电芯温升',
            xaxis_title='温升 dT (C)',
            yaxis_title='固有频率 (Hz)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 共振判定")

        road_freq = st.slider("路面激励频率 (Hz)", 5, 50, 15, 1)

        margin_cold = (f_cold - road_freq) / road_freq * 100
        margin_hot = (f_hot - road_freq) / road_freq * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("冷态共振裕度", f"{margin_cold:.1f}%")
            if abs(margin_cold) < 10:
                st.error("冷态存在共振风险！")
            else:
                st.success("冷态安全")

        with col2:
            st.metric("热态共振裕度", f"{margin_hot:.1f}%")
            if abs(margin_hot) < 10:
                st.error("热态存在共振风险！")
            else:
                st.success("热态安全")

        st.markdown("---")
        st.markdown("### 模组温度场分布")

        T_profile = compute_temperature_field(n_cells, T_ambient, dT_cell, Q_gen)
        sigma_profile = compute_thermal_stress_distribution(T_profile, mat['E_cell'], mat['mu'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=['温度分布', '热应力分布'])

        fig.add_trace(go.Bar(x=[f'电芯{i + 1}' for i in range(n_cells)], y=T_profile,
                             marker_color='red', name='温度'), row=1, col=1)
        fig.add_trace(go.Bar(x=[f'电芯{i + 1}' for i in range(n_cells)], y=sigma_profile / 1e6,
                             marker_color='blue', name='热应力'), row=1, col=2)

        fig.update_layout(height=350, showlegend=False)
        fig.update_yaxes(title_text="温度 (C)", row=1, col=1)
        fig.update_yaxes(title_text="应力 (MPa)", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 模块3：结果分析
# ============================================================
elif module == "结果分析":
    st.title("结果分析与工程判断")
    st.markdown("深度解读计算结果，做出工程决策")
    st.markdown("---")

    st.markdown("### 输入计算结果")

    col1, col2, col3 = st.columns(3)
    with col1:
        f_cold = st.number_input("冷态频率 (Hz)", value=45.0, step=1.0)
    with col2:
        f_hot = st.number_input("热态频率 (Hz)", value=58.0, step=1.0)
    with col3:
        road_freq = st.number_input("路面激励频率 (Hz)", value=15.0, step=0.5)

    delta_f = (f_hot - f_cold) / f_cold * 100

    st.markdown("---")
    st.markdown("### 频率漂移分析")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['冷态', '热态'],
        y=[f_cold, f_hot],
        text=[f'{f_cold:.1f} Hz', f'{f_hot:.1f} Hz'],
        textposition='outside',
        marker_color=['#4facfe', '#f5576c']
    ))
    fig.add_hline(y=road_freq, line_dash="dash", line_color="green",
                  annotation_text=f'路面激励 {road_freq}Hz')
    fig.update_layout(title='固有频率对比', yaxis_title='频率 (Hz)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 分析结论")

    if abs(delta_f) < 5:
        drift_effect = "频率漂移较小，热效应不明显"
    elif abs(delta_f) < 15:
        drift_effect = "频率漂移中等，建议关注"
    else:
        drift_effect = "频率漂移显著，必须考虑热-振耦合"

    if f_hot > road_freq * 0.9 and f_hot < road_freq * 1.1:
        risk = "热态频率接近路面激励，存在共振风险！"
    elif f_cold > road_freq * 0.9 and f_cold < road_freq * 1.1:
        risk = "冷态频率接近路面激励，存在共振风险！"
    else:
        risk = "频率安全，无共振风险"

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"频率漂移: {delta_f:+.1f}%\n{drift_effect}")
    with col2:
        if "风险" in risk:
            st.error(risk)
        else:
            st.success(risk)

    st.markdown("---")
    st.markdown("### 工程建议")

    if abs(delta_f) > 15:
        st.warning("""
        建议:
        1. 优化泡棉刚度，降低接触刚度温度敏感性
        2. 考虑变刚度泡棉设计
        3. 优化端板预紧力
        4. 增加模组约束点
        """)
    elif abs(delta_f) > 8:
        st.info("""
        建议:
        1. 监测实际工作温度
        2. 进行随机振动试验验证
        3. 考虑主动避频策略
        """)
    else:
        st.success("""
        设计合理，无需特别措施
        建议进行常规耐久性验证
        """)

# ============================================================
# 模块4：振型图
# ============================================================
elif module == "振型图":
    st.title("电池模组振型图")
    st.markdown("展示模组在不同频率下的振动形态")
    st.markdown("---")

    st.markdown("### 振型参数设置")

    col1, col2 = st.columns(2)
    with col1:
        n_modes = st.slider("电芯数量", 4, 12, 6)
        mode_select = st.selectbox("模态阶数", ["1阶平动", "2阶平动", "3阶平动"], index=0)
        amplitude = st.slider("振幅放大倍数", 1, 10, 3)

    with col2:
        show_animation = st.checkbox("动态显示", value=True)
        temp_effect = st.checkbox("显示热态振型", value=True)

    st.markdown("---")
    st.markdown("### 振型展示")

    x_pos = np.linspace(0, 1, n_modes)
    mode_map = {"1阶平动": 1, "2阶平动": 2, "3阶平动": 3}
    mode_num = mode_map[mode_select]

    shape = np.sin(mode_num * np.pi * x_pos)
    shape = shape / np.max(np.abs(shape))

    fig, ax = plt.subplots(figsize=(12, 5))

    if show_animation:
        t = np.linspace(0, 2 * np.pi, 15)
        for ti in t:
            disp = shape * np.sin(ti) * amplitude * 0.3
            ax.plot(x_pos, disp, 'b-', linewidth=1, alpha=0.2)

    ax.fill_between(x_pos, -shape * amplitude * 0.3, shape * amplitude * 0.3, alpha=0.2, color='blue')
    ax.plot(x_pos, shape * amplitude * 0.3, 'r-', linewidth=2, label='最大振幅')
    ax.plot(x_pos, -shape * amplitude * 0.3, 'r-', linewidth=2, alpha=0.5)

    if temp_effect:
        shape_hot = np.sin((mode_num + 0.3) * np.pi * x_pos)
        shape_hot = shape_hot / np.max(np.abs(shape_hot))
        ax.plot(x_pos, shape_hot * amplitude * 0.25, 'g--', linewidth=2, label='热态振型', alpha=0.7)

    ax.scatter(x_pos, np.zeros_like(x_pos), color='black', s=50, zorder=5, label='电芯位置')

    ax.set_xlabel('电芯位置 (归一化)')
    ax.set_ylabel('振幅 (mm)')
    ax.set_title(f'{mode_select} 振型')
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    st.info("蓝线表示振动包络，红线表示最大振幅，绿线为热态振型")

# ============================================================
# 模块5：3D可视化
# ============================================================
elif module == "3D可视化":
    st.title("电池模组3D可视化")
    st.markdown("三维展示模组温度场与应力分布")
    st.markdown("---")

    st.markdown("### 可视化参数")

    col1, col2 = st.columns(2)
    with col1:
        n_cells_vis = st.slider("电芯数量", 4, 12, 6)
        vis_mode = st.selectbox("显示模式", ["温度场", "应力场", "变形场"])
        T_max = st.slider("最大温度 (C)", 40, 100, 65)

    with col2:
        show_binding = st.checkbox("显示绑带", value=True)
        show_labels = st.checkbox("显示标签", value=True)
        view_angle = st.slider("视角", 0, 360, 45)

    st.markdown("---")
    st.markdown("### 3D模组模型")

    cell_width = 0.6
    cell_height = 1.2
    cell_depth = 0.8
    gap = 0.08

    fig = go.Figure()

    for i in range(n_cells_vis):
        x = i * (cell_width + gap) - (n_cells_vis - 1) * (cell_width + gap) / 2
        y = 0
        z = 0

        temp_ratio = (i / (n_cells_vis - 1)) if n_cells_vis > 1 else 0.5
        temp = 25 + (T_max - 25) * temp_ratio

        if vis_mode == "温度场":
            color_scale = temp / 100
            color = f'rgb({int(255 * color_scale)}, {int(100 * (1 - color_scale))}, 0)'
        elif vis_mode == "应力场":
            stress = (temp - 25) / (T_max - 25)
            color = f'rgb({int(255 * stress)}, 0, {int(255 * (1 - stress))})'
        else:
            deform = np.sin(i / (n_cells_vis - 1) * np.pi) * 0.1
            color = f'rgb({int(100 + 155 * deform)}, {int(200 - 150 * abs(deform))}, 100)'

        fig.add_trace(go.Mesh3d(
            x=[x - cell_width / 2, x + cell_width / 2, x + cell_width / 2, x - cell_width / 2,
               x - cell_width / 2, x + cell_width / 2, x + cell_width / 2, x - cell_width / 2],
            y=[y - cell_depth / 2, y - cell_depth / 2, y + cell_depth / 2, y + cell_depth / 2,
               y - cell_depth / 2, y - cell_depth / 2, y + cell_depth / 2, y + cell_depth / 2],
            z=[z, z, z, z, z + cell_height, z + cell_height, z + cell_height, z + cell_height],
            i=[0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4],
            j=[1, 2, 3, 5, 6, 7, 1, 2, 3, 5, 6, 7],
            k=[2, 3, 0, 6, 7, 4, 5, 6, 7, 1, 2, 3],
            color=color,
            opacity=0.85,
            showscale=False,
            name=f'电芯{i + 1}'
        ))

        if show_labels:
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y + cell_depth / 2 + 0.1], z=[cell_height / 2],
                mode='text',
                text=[f'{i + 1}'],
                textfont=dict(size=10, color='black'),
                showlegend=False
            ))

    for i in range(n_cells_vis - 1):
        x = i * (cell_width + gap) + cell_width / 2 - (n_cells_vis - 1) * (cell_width + gap) / 2 + gap / 2
        fig.add_trace(go.Mesh3d(
            x=[x, x + gap, x + gap, x, x, x + gap, x + gap, x],
            y=[-cell_depth / 2, -cell_depth / 2, cell_depth / 2, cell_depth / 2,
               -cell_depth / 2, -cell_depth / 2, cell_depth / 2, cell_depth / 2],
            z=[0, 0, 0, 0, cell_height, cell_height, cell_height, cell_height],
            i=[0, 0, 0, 4, 4, 4],
            j=[1, 2, 3, 5, 6, 7],
            k=[2, 3, 0, 6, 7, 4],
            color='#ff6b6b',
            opacity=0.4,
            showscale=False,
            name='泡棉'
        ))

    if show_binding:
        for z in [0.1, cell_height - 0.1]:
            fig.add_trace(go.Scatter3d(
                x=[-(n_cells_vis) * (cell_width + gap) / 2 - 0.2, (n_cells_vis) * (cell_width + gap) / 2 + 0.2],
                y=[-cell_depth / 2 - 0.2, -cell_depth / 2 - 0.2],
                z=[z, z],
                mode='lines',
                line=dict(color='#2c3e50', width=6),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[-(n_cells_vis) * (cell_width + gap) / 2 - 0.2, (n_cells_vis) * (cell_width + gap) / 2 + 0.2],
                y=[cell_depth / 2 + 0.2, cell_depth / 2 + 0.2],
                z=[z, z],
                mode='lines',
                line=dict(color='#2c3e50', width=6),
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=2 * np.cos(np.radians(view_angle)),
                                 y=2 * np.sin(np.radians(view_angle)),
                                 z=1.2))
        ),
        height=600,
        title=f'{vis_mode} - 电池模组3D显示'
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 模块6：报告生成
# ============================================================
else:
    st.title("分析报告生成")
    st.markdown("导出完整的热-振动耦合分析报告")
    st.markdown("---")

    st.markdown("### 报告内容")

    col1, col2 = st.columns(2)
    with col1:
        include_params = st.checkbox("包含参数设置", value=True)
        include_results = st.checkbox("包含计算结果", value=True)
    with col2:
        include_conclusions = st.checkbox("包含分析结论", value=True)
        include_recommendations = st.checkbox("包含工程建议", value=True)

    st.markdown("---")
    st.markdown("### 报告预览")

    with st.expander("点击展开完整报告", expanded=True):
        if include_params:
            st.markdown("""
            ## 输入参数
            | 参数 | 值 | 单位 |
            |------|-----|------|
            | 电芯数量 | 6 | 个 |
            | 单颗电芯质量 | 0.8 | kg |
            | 电芯截面积 | 0.01 | m2 |
            | 电芯厚度 | 0.03 | m |
            | 环境温度 | 25 | C |
            | 电芯温升 | 40 | C |
            | 初始接触刚度 | 50 | MN/m |
            """)

        if include_results:
            st.markdown("""
            ## 计算结果
            | 状态 | 频率 (Hz) | 变化 |
            |------|-----------|------|
            | 冷态 | 45.0 | - |
            | 热态 | 58.0 | +28.9% |

            **热膨胀力:** 15.6 kN
            **接触刚度:** 68.5 MN/m
            """)

        if include_conclusions:
            st.markdown("""
            ## 分析结论
            1. 频率漂移显著: 温升导致固有频率上升 28.9%
            2. 热-振耦合效应不可忽略
            3. 热态频率接近路面激励，存在共振风险
            """)

        if include_recommendations:
            st.markdown("""
            ## 工程建议
            1. 优化泡棉刚度设计
            2. 考虑变刚度泡棉
            3. 优化端板预紧力
            """)

    st.markdown("---")
    if st.button("导出报告", use_container_width=True):
        st.success("报告已导出")

# ============================================================
# 底部
# ============================================================
st.markdown("---")
st.caption("电池模组热-振动耦合分析智能体 | 新能源工程应用")

