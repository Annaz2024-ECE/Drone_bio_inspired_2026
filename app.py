import streamlit as st
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# 导入底层代码
from environment_buildup_3D import UAVEnvironment3D
from path_evaluator import PathEvaluator
from algorithm_select_agent import Algorithm_Select_Agent
from coordinator_agent import CoordinatorAgent
from video_animator import UAVVideoAnimator

# 导入所有底层规划算法
from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from woa_planner import WOAPlanner
from ga_planner import GAPlanner

# ==========================================
# 1. 算法装配工厂函数 (从 main 里搬过来的)
# ==========================================
def create_planner_with_params(algo_name, evaluator, algo_params, specific_params, elite_path=None):
    pop_size = algo_params.get('pop_size', 50)
  #  max_iter = algo_params.get('max_iter', 100)
    max_iter = 100
    num_waypoints = algo_params.get('num_waypoints', 30) 
    
    if algo_name == "PSO": planner = PSOPlanner(evaluator=evaluator, num_particles=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "SSA": planner = SSAPlanner(evaluator=evaluator, num_sparrows=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "GWO": planner = GWOPlanner(evaluator=evaluator, num_wolves=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "WOA": planner = WOAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "GA": planner = GAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    else: planner = PSOPlanner(evaluator=evaluator, num_waypoints=num_waypoints)

    for key, value in specific_params.items():
        if hasattr(planner, key) or key in ['num_producers', 'lift_up', 'press_down', 'radar_guidance', 'emergency_escape']:
            setattr(planner, key, value)

    if elite_path is not None:
        try:
            elite_1d = elite_path[1:-1].flatten() 
            if hasattr(planner, 'particles'): planner.particles[0] = elite_1d
            elif hasattr(planner, 'sparrows'): planner.sparrows[0] = elite_1d
            elif hasattr(planner, 'positions'): planner.positions[0] = elite_1d
        except Exception as e:
            pass
            
    return planner

# ==========================================
# 2. 绘制地图预览（2D俯视图）
# ==========================================
def plot_map_preview(evaluator):
    """使用与 BasePlanner 相同的绘制逻辑，仅显示静态地图（障碍物、起点、终点）"""
    env = evaluator.env
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 设置坐标范围（确保 env 有 x_bounds, y_bounds）
    ax.set_xlim(env.x_bounds)
    ax.set_ylim(env.y_bounds)

    # 绘制障碍物（完全照搬 BasePlanner.plot_result 中的代码）
    for obs in env.obstacles:
        z_max = obs.get('z_max', 20.0)
        if z_max <= 1.5:
            color = '#e3f2fd'
        elif z_max <= 3.0:
            color = '#90caf9'
        elif z_max <= 5.0:
            color = '#1e88e5'
        else:
            color = '#0d47a1'

        if obs['type'] == 'circle':
            circle = plt.Circle(obs['center'][:2], obs['radius'],
                                color=color, alpha=0.7, ec='black')
            ax.add_patch(circle)
        elif obs['type'] == 'rect':
            from matplotlib.patches import Rectangle
            rect = Rectangle(obs['bottom_left'][:2], obs['width'], obs['height'],
                             angle=obs.get('angle', 0), color=color, alpha=0.7, ec='black')
            ax.add_patch(rect)
        elif obs['type'] == 'polygon':
            from matplotlib.patches import Polygon as mpl_Polygon
            poly = mpl_Polygon(obs['points'], closed=True, color=color, alpha=0.7, ec='black')
            ax.add_patch(poly)

    # 绘制起点和终点（使用和 BasePlanner 相同的样式）
    ax.scatter(*env.start_point[:2], color='#fbc02d', s=100, marker='*', edgecolors='black', zorder=10, label='StartPoint')
    ax.scatter(*env.end_point[:2], color='#d32f2f', s=80, marker='^', zorder=10, label='EndPoint')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-down 2D Map')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    return fig

# ==========================================
# 2. Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="基于多智能体的无人机积水巡检", page_icon="🚁", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* 整体背景渐变 */
    .stApp {
        background: linear-gradient(145deg, #0b0f1a 0%, #141b2b 100%);
    }
    
    /* 卡片容器 —— 用于放置地图、视频、图表 */
    .css-1r6slb0, .css-1v3fvcr, .css-1rs6os {
        background: rgba(30, 40, 60, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(0, 170, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        padding: 20px;
    }
    
    /* 标题发光文字 */
    h1, h2, h3 {
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00aaff, #aa66ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    }
    h1 {
        font-size: 2.8rem !important;
        text-shadow: 0 0 30px rgba(0,170,255,0.3);
    }
    
    /* 侧边栏美化 */
    .css-1d391kg {
        background: rgba(14, 17, 23, 0.9) !important;
        border-right: 2px solid rgba(0, 170, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* 按钮样式 —— 霓虹发光 */
    .stButton > button {
        background: linear-gradient(135deg, #00aaff, #0066cc) !important;
        border: none !important;
        border-radius: 50px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 170, 255, 0.3) !important;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(0, 170, 255, 0.6) !important;
        border-color: #00ccff !important;
    }
    .stButton > button:active {
        transform: scale(0.95);
    }
    /* 主要按钮（开始规划）特殊颜色 */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ff6b6b, #ee0979) !important;
        box-shadow: 0 0 25px rgba(238, 9, 121, 0.5) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 50px rgba(238, 9, 121, 0.7) !important;
    }
    
    /* 滑块样式 */
    .stSlider > div > div > div {
        background: #00aaff !important;
    }
    
    /* 指标卡片（后面会用） */
    .metric-card {
        background: rgba(0, 170, 255, 0.08);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(0, 170, 255, 0.2);
        backdrop-filter: blur(5px);
        text-align: center;
    }
    .metric-card .label {
        color: #8899bb;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #00ddff;
        font-size: 2.2rem;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 221, 255, 0.3);
    }
    
    /* 视频容器阴影 */
    .stVideo {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 0 40px rgba(0, 170, 255, 0.2);
    }
    
    /* spinner 文字样式（保留，但不再隐藏任何元素） */
    .stSpinner {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.2rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* 页脚小字 */
    .footer {
        text-align: center;
        color: #445566;
        font-size: 0.8rem;
        padding-top: 30px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# 初始化状态
if 'result_video' not in st.session_state:
    st.session_state.result_video = None
if 'result_score' not in st.session_state:
    st.session_state.result_score = None
if 'fig_2d' not in st.session_state:
    st.session_state.fig_2d = None
if 'fig_curve' not in st.session_state:
    st.session_state.fig_curve = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

st.title("基于多智能体的无人机积水巡检")
#st.markdown("集成多智能体协同的 3D 复杂空域巡检路径规划系统")


# ==========================================
# 4. 侧边栏：参数输入
# ==========================================
with st.sidebar:
    st.header("⚙️ 任务环境参数")
    
    map_choice = st.selectbox(
        "🗺️ 选择测试地图",
        ["测试地图-低难度", 
         "测试地图-中等难度", 
         "测试地图-高难度",
         "应用地图-海宁校区",
         "应用地图-紫金港校区"]
    )
    
    map_file_dict = {
        "测试地图-低难度": "maps/easy_map.json5",
        "测试地图-中等难度": "maps/medium_map.json5",
        "测试地图-高难度": "maps/hard_map.json5",
        "应用地图-海宁校区": "maps/haining.json5",
        "应用地图-紫金港校区":"maps/zijingang_2.json5"
    }
    selected_map_path = map_file_dict[map_choice]

    st.markdown("---")
    rainfall_mm = st.slider("🌧️ 累计降雨量 (mm)", min_value=0.0, max_value=100.0, value=20.0, step=5.0)
    duration_hours = st.slider("⏱️ 降雨持续时间 (h)", min_value=0.5, max_value=24.0, value=2.0, step=0.5)
    
    st.markdown("---")
    use_llm = st.toggle("启用云端 LLM  (DeepSeek)", value=True)
    api_key = st.text_input("🔑 输入 DeepSeek API Key", value=os.environ.get("DEEPSEEK_API_KEY", ""), type="password")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 开始规划", type="primary", use_container_width=True, disabled=st.session_state.is_running)
    with col2:
        reset_btn = st.button("🔄 重置结果", use_container_width=True)

    if reset_btn:
        st.session_state.result_video = None
        st.session_state.result_score = None
        st.session_state.fig_2d = None
        st.session_state.fig_curve = None
        st.rerun()

# ==========================================
# 5. 主区域：加载地图并显示预览（始终显示）
# ==========================================
# 创建 evaluator 并加载地图（用于预览和后续计算）
evaluator = PathEvaluator()
if not os.path.exists(selected_map_path):
    st.error(f"找不到地图文件: {selected_map_path}")
    st.stop()
evaluator.env = UAVEnvironment3D(selected_map_path)

# # === 标题后增加 metrics ===
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("🧠 状态", "就绪" if not st.session_state.is_running else "规划中...")
# with col2:
#     st.metric("🏆 当前得分", f"{st.session_state.result_score:,.2f}" if st.session_state.result_score else "--")
# with col3:
#     st.metric("🗺️ 地图", map_choice.split("-")[-1])

# 显示地图预览
st.subheader("🗺️ 当前空域地图")
fig_preview = plot_map_preview(evaluator)
st.pyplot(fig_preview)
plt.close(fig_preview)

# ==========================================
# 6. 结果展示区（初始提示或显示结果）
# ==========================================
st.markdown("---")
if st.session_state.result_video and os.path.exists(st.session_state.result_video):
    st.success(f"🏁 规划完成！最优得分: {st.session_state.result_score:,.2f}")
    
    st.subheader("🎥 3D 飞行仿真")
    st.video(st.session_state.result_video)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🗺️ 2D 速度热力")
        st.pyplot(st.session_state.fig_2d)
        plt.close(st.session_state.fig_2d)
    with col_right:
        st.subheader("📈 收敛与干预")
        st.pyplot(st.session_state.fig_curve)
        plt.close(st.session_state.fig_curve)
else:
    st.info("👆 在左侧设置参数后，点击「开始规划」执行路径搜索")

# ==========================================
# 7. 核心算法（点击开始后执行，隐藏过程）
# ==========================================
if start_btn:
    st.session_state.is_running = True
    
    if use_llm and not api_key:
        st.error("请在左侧输入 API Key！")
        st.session_state.is_running = False
        st.stop()
        
    if use_llm:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    # 使用 spinner 隐藏运算过程
    with st.spinner("无人机正在规划路径，请稍候..."):
        try:
            # 加载先验知识
            prior_knowledge_file = 'None'
            if os.path.exists(prior_knowledge_file):
                with open(prior_knowledge_file, "r", encoding='utf-8') as f:
                    real_prior_knowledge = json.load(f)
            else:
                real_prior_knowledge = {
                    "Easy_Map": {"PSO": "收敛极快，总体最优。"},
                    "Hard_Map_密集障碍": {"PSO": "展现出极强的破壁能力。"}
                }
            
            # LLM 气象感知智能体
            opt_agent = Algorithm_Select_Agent(
                evaluator=evaluator,
                prior_knowledge=real_prior_knowledge,
                rainfall_mm=rainfall_mm, 
                duration_hours=duration_hours, 
                use_llm=use_llm
            )
            initial_planner = opt_agent.make_decision()
            current_algo_name = type(initial_planner).__name__.replace('Planner', '')
            
            # 协调决策智能体
            pop_size = getattr(initial_planner, 'num_particles', getattr(initial_planner, 'num_sparrows', getattr(initial_planner, 'num_wolves', getattr(initial_planner, 'pop_size', 50))))
            coord_agent = CoordinatorAgent()
            coord_agent.algo_params = {'pop_size': pop_size, 'max_iter': initial_planner.max_iter, 'num_waypoints': initial_planner.num_waypoints}
            
            current_algo_params = coord_agent.algo_params
            current_specific_params = {}
            global_best_path = None
            global_best_score = float('inf')
            global_iteration_count = 0  
            event_history = []
            MAX_META_ITERATIONS = 3  # 为演示速度，仅2轮
            final_history = []
            
            # 大循环
            for meta_iter in range(1, MAX_META_ITERATIONS + 1):
                # 实例化算法
                planner = create_planner_with_params(
                    algo_name=current_algo_name, evaluator=evaluator, 
                    algo_params=current_algo_params, specific_params=current_specific_params,
                    elite_path=global_best_path
                )
                best_path, history = planner.optimize()
                final_history.extend(history)
                global_iteration_count += len(history)
                
                # 评价
                total_score, details, env_info = evaluator.evaluate_particle(best_path)
                
                if total_score < global_best_score:
                    global_best_score = total_score
                    global_best_path = best_path

                # 协调智能体分析
                current_algo_params, new_eval_params, current_specific_params, is_finished, param_changes = \
                    coord_agent.analyze_and_act(global_best_score, details, env_info, current_algo_name)

                if param_changes:
                    event_history.append((global_iteration_count, "\n".join(param_changes)))
                    
                evaluator.params.update(new_eval_params)
                
                if is_finished:
                    break
                    
                # 换将
                if coord_agent.stuck_counter >= 2:
                    next_algo, new_pop, new_iter, new_wp = opt_agent.get_fallback_algorithm(current_algo_name, details)
                    current_algo_name = next_algo
                    coord_agent.algo_params.update({'pop_size': new_pop, 'max_iter': new_iter, 'num_waypoints': new_wp})
                    current_algo_params = coord_agent.algo_params
                    coord_agent.stuck_counter = 0 
                    current_specific_params = {}

            # 生成结果
            if global_best_path is not None:
                output_root = "output"
                run_idx = 0
                run_subdir = os.path.join(output_root, f"run_{run_idx:02d}")
                os.makedirs(run_subdir, exist_ok=True)

                video_filename = os.path.join(run_subdir, f"{current_algo_name}_final_flight.mp4")
                animator = UAVVideoAnimator(evaluator)
                animator.create_flight_video(global_best_path, filename=video_filename, duration=10.0)

                # 获取2D图和收敛曲线
                figs = planner.plot_result(
                    best_path=global_best_path,
                    score_history=final_history,
                    algo_name=current_algo_name,
                    event_history=event_history,
                    run_idx=run_idx,
                    save_dir=output_root,
                    return_figs=True
                )
                _, _, fig_2d, fig_curve = figs

                # 存入 session_state
                st.session_state.result_video = video_filename
                st.session_state.result_score = global_best_score
                st.session_state.fig_2d = fig_2d
                st.session_state.fig_curve = fig_curve

        except Exception as e:
            import traceback
            st.error(f"❌ 运行报错: {e}")
            st.code(traceback.format_exc())
        finally:
            st.session_state.is_running = False
            # 刷新页面以显示结果
            if st.session_state.result_video is not None:
                st.rerun()