import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, softmax
import time

# 加上这段兼容性判断
if hasattr(np, 'trapezoid'):
    integrate = np.trapezoid
else:
    integrate = np.trapz
# 设置页面
st.set_page_config(page_title="Cubic Model Convergence", layout="wide")
st.title("Cubic Mean-Field Model: $W \\rightarrow Y_n \\rightarrow Y$")
st.markdown("### 实验演示：固定 $\\alpha$，自动演示 $n$ 的收敛过程")

# 侧边栏：只调整 alpha
alpha = st.sidebar.slider("1. 调整参数 α (控制分布偏斜度)", -2.5, -0.1, -0.8)
speed = st.sidebar.slider("2. 动画速度", 1, 10, 5)
start_btn = st.sidebar.button("▶ 开始动画 (自动变动 n)")

# 核心计算函数
def get_densities(n, alpha, w):
    # 极限 Y (红色)
    d_limit = np.exp(alpha * (w**2) / 2 - (w**4) / 12)
    # 把 np.trapz 换成 integrate
    d_limit /= integrate(d_limit, w) 
    
    # 优化后的 Y_n
    c0, c1, c2 = -4/(3*n**1.75), -alpha+1/n**0.5+alpha/n, 2/n**1.25-1/n**0.25
    c3 = 1/3 + alpha/n**0.5 + alpha**2/n + alpha**3/(3*n**1.5)
    c4 = 1/n**0.75 + 2*alpha/n**1.25 + alpha**2/n**1.75
    c5 = 1/n**1.5 + alpha/n**2
    v_w = c0*w + (c1/2)*w**2 + (c2/3)*w**3 + (c3/4)*w**4 + (c4/5)*w**5 + (c5/6)*w**6
    d_stein = np.exp(-v_w)
    # 把 np.trapz 换成 integrate
    d_stein /= integrate(d_stein, w) 
    return d_limit, d_stein

def get_empirical(n, alpha):
    k_vals = np.arange(-n, n + 1, 2)
    m = k_vals / n
    K, J = n**-0.5, 1 + alpha*n**-0.5
    log_p = gammaln(n + 1) - gammaln((n + k_vals) / 2 + 1) - gammaln((n - k_vals) / 2 + 1) + n*((K/3)*m**3 + (J/2)*m**2)
    return k_vals/(n**0.75), softmax(log_p)

# 绘图区域占位符
plot_spot = st.empty()

# 动画逻辑
w_plot = np.linspace(-4, 4, 300)
n_values = [100, 200, 400, 700, 1000, 2000, 4000, 7000, 10000]

if start_btn:
    for n in n_values:
        d_limit, d_stein = get_densities(n, alpha, w_plot)
        w_real, p_real = get_empirical(n, alpha)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(w_real, p_real * (n**0.75 / 2), width=0.1, color='lightgray', alpha=0.6, label='Empirical $W$')
        ax.plot(w_plot, d_limit, 'r-', lw=2, label='Ultimate Limit $Y$ (Static)')
        ax.plot(w_plot, d_stein, 'b--', lw=2, label=f'Optimized $Y_n$ (n={n})')
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.8)
        ax.set_title(f"$\\alpha = {alpha}$, $n = {n}$")
        ax.legend()
        
        plot_spot.pyplot(fig)
        plt.close(fig)
        time.sleep(1 / speed)
    st.success("动画演示结束！你可以调整 α 重新开始。")
else:
    st.info("请在左侧调整 α，然后点击 '开始动画'")
