import streamlit as st
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
# 1. 算法装配工厂函数
# ==========================================
def create_planner_with_params(algo_name, evaluator, algo_params, specific_params, elite_path=None):
    pop_size = algo_params.get('pop_size', 50)
    max_iter = 100
   # num_waypoints = algo_params.get('num_waypoints', 30)
    num_waypoints = 20

    if algo_name == "PSO":
        planner = PSOPlanner(evaluator=evaluator, num_particles=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "SSA":
        planner = SSAPlanner(evaluator=evaluator, num_sparrows=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "GWO":
        planner = GWOPlanner(evaluator=evaluator, num_wolves=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "WOA":
        planner = WOAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "GA":
        planner = GAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    else:
        planner = PSOPlanner(evaluator=evaluator, num_waypoints=num_waypoints)

    for key, value in specific_params.items():
        if hasattr(planner, key) or key in ['num_producers', 'lift_up', 'press_down', 'radar_guidance', 'emergency_escape']:
            setattr(planner, key, value)

    if elite_path is not None:
        try:
            elite_1d = elite_path[1:-1].flatten()
            if hasattr(planner, 'particles'):
                planner.particles[0] = elite_1d
            elif hasattr(planner, 'sparrows'):
                planner.sparrows[0] = elite_1d
            elif hasattr(planner, 'positions'):
                planner.positions[0] = elite_1d
        except Exception:
            pass

    return planner

# ==========================================
# 2. 绘制地图预览（2D俯视图）
# ==========================================
def plot_map_preview(evaluator):
    env = evaluator.env
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(env.x_bounds)
    ax.set_ylim(env.y_bounds)

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
            circle = plt.Circle(obs['center'][:2], obs['radius'], color=color, alpha=0.7, ec='black')
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

    start_scatter = ax.scatter(*env.start_point[:2], color='#fbc02d', s=100,
                               marker='*', edgecolors='black', zorder=10, label='Start')
    end_scatter = ax.scatter(*env.end_point[:2], color='#d32f2f', s=80,
                             marker='^', zorder=10, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-down 2D Map')
    ax.grid(True, linestyle=':', alpha=0.4)

    obs_patches = [
        Patch(facecolor='#e3f2fd', edgecolor='black', label='H ≤ 1.5m'),
        Patch(facecolor='#90caf9', edgecolor='black', label='1.5< H ≤3.0m'),
        Patch(facecolor='#1e88e5', edgecolor='black', label='3.0< H ≤5.0m'),
        Patch(facecolor='#0d47a1', edgecolor='black', label='H > 5.0m')
    ]
    leg_start = ax.legend(handles=[start_scatter, end_scatter], loc='upper left', fontsize=8)
    obs_legend = ax.legend(
        handles=obs_patches,
        title='Obstacle Height',
        bbox_to_anchor=(1.02, 1),     # 放在右侧，与右上角对齐
        loc='upper left',
        fontsize=6,
        title_fontsize=7,
        handlelength=1.0,
        handleheight=0.8
    )
    plt.subplots_adjust(right=0.78)   # 右侧留出约 22% 空白
    # 或者使用 tight_layout 的 rect 参数
    plt.tight_layout(rect=[0, 0, 0.78, 1])

    return fig

# ==========================================
# 3. Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="基于多智能体的无人机积水巡检", page_icon="🚁", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(145deg, #0b0f1a 0%, #141b2b 100%);
    }
    .css-1r6slb0, .css-1v3fvcr, .css-1rs6os {
        background: rgba(30, 40, 60, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(0, 170, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        padding: 20px;
    }
    /* 标题——大幅增加顶部空间 */
    h1, h2, h3 {
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00aaff, #aa66ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
        margin-top: 1.2rem !important;   /* 关键：上外边距 */
        margin-bottom: 0.2rem !important;
        padding-top: 0.5rem !important;
    }
    h1 {
        font-size: 2.2rem !important;
        text-shadow: 0 0 30px rgba(0,170,255,0.3);
    }
    .css-1d391kg {
        background: rgba(14, 17, 23, 0.9) !important;
        border-right: 2px solid rgba(0, 170, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00aaff, #0066cc) !important;
        border: none !important;
        border-radius: 50px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
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
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ff6b6b, #ee0979) !important;
        box-shadow: 0 0 25px rgba(238, 9, 121, 0.5) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 50px rgba(238, 9, 121, 0.7) !important;
    }
    .stSlider > div > div > div {
        background: #00aaff !important;
    }
    .stVideo {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 0 40px rgba(0, 170, 255, 0.2);
    }
    .stSpinner {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.2rem;
    }
    /* 容器顶部留白足够 */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 32px;
        padding: 0 10px;
        font-size: 0.85rem;
    }
    .stAlert, .stCaption, .stSubheader {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
    }
    .stImage {
        width: 100% !important;
        display: flex;
        justify-content: center;
    }
    img {
        max-width: 100% !important;
        height: auto !important;
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
if 'detail_report' not in st.session_state:
    st.session_state.detail_report = None

st.title("基于多智能体的无人机积水巡检")

# ==========================================
# 4. 侧边栏：参数输入
# ==========================================
with st.sidebar:
    st.header("⚙️ 任务环境参数")
    
    map_choice = st.selectbox(
        "🗺️ 选择测试地图",
        ["测试地图-低难度", "测试地图-中等难度", "测试地图-高难度",
         "应用地图-海宁校区", "应用地图-紫金港校区"]
    )
    
    map_file_dict = {
        "测试地图-低难度": "maps/easy_map.json5",
        "测试地图-中等难度": "maps/medium_map.json5",
        "测试地图-高难度": "maps/hard_map.json5",
        "应用地图-海宁校区": "maps/haining.json5",
        "应用地图-紫金港校区": "maps/zijingang_2.json5"
    }
    selected_map_path = map_file_dict[map_choice]

    st.markdown("---")
    rainfall_mm = st.slider("🌧️ 累计降雨量 (mm)", min_value=0.0, max_value=100.0, value=20.0, step=5.0)
    duration_hours = st.slider("⏱️ 降雨持续时间 (h)", min_value=0.5, max_value=24.0, value=2.0, step=0.5)
    
    st.markdown("---")
    use_llm = st.toggle("启用云端 LLM (DeepSeek)", value=True)
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
        st.session_state.detail_report = None
        st.rerun()

    # ===== 侧边栏底部：详细报告折叠 =====
    if hasattr(st.session_state, 'detail_report') and st.session_state.detail_report:
        with st.expander("📊 查看详细报告"):
            report = st.session_state.detail_report
            st.markdown("**⚙️ 算法参数**")
            if report.get("algorithm_parameters"):
                for k, v in report["algorithm_parameters"].items():
                    st.text(f"  {k}: {v}")
            st.markdown("**📉 惩罚明细**")
            if report.get("fitness_breakdown"):
                for k, v in report["fitness_breakdown"].items():
                    st.text(f"  {k}: {v:,.0f}")
            else:
                st.text("  ✅ 完美路径，无惩罚")
            st.markdown("**🎯 干预事件**")
            if report.get("intervention_events"):
                for event in report["intervention_events"]:
                    st.text(f"  Iter {event['iteration']}: {event['action']}")
            st.caption(f"⏱️ {report.get('time_info', '')}")

# ==========================================
# 5. 主区域：加载地图并显示预览 / 结果
# ==========================================
evaluator = PathEvaluator()
if not os.path.exists(selected_map_path):
    st.error(f"找不到地图文件: {selected_map_path}")
    st.stop()
evaluator.env = UAVEnvironment3D(selected_map_path)

if st.session_state.result_video and os.path.exists(st.session_state.result_video):
    st.success(f"🏁 规划完成！最优得分: {st.session_state.result_score:,.2f}")
    
    tab_video, tab_2d, tab_curve = st.tabs(["🎥 3D 飞行", "🗺️ 2D 速度热力", "📈 收敛曲线"])
    
    with tab_video:
        st.video(st.session_state.result_video)
    
    with tab_2d:
        if st.session_state.fig_2d is not None:
            fig_2d = st.session_state.fig_2d
            fig_2d.set_size_inches(6, 4)   
            fig_2d.tight_layout()
            st.pyplot(fig_2d, use_container_width=False)
            plt.close(fig_2d)
        else:
            st.info("2D 图未生成")
    
    with tab_curve:
        if st.session_state.fig_curve is not None:
            fig_cur = st.session_state.fig_curve
            fig_cur.set_size_inches(6, 4)
            fig_cur.tight_layout()
            st.pyplot(fig_cur, use_container_width=False)
            plt.close(fig_cur)
        else:
            st.info("收敛曲线未生成")
else:
    st.subheader("🗺️ 当前空域地图")
    fig_preview = plot_map_preview(evaluator)
    fig_preview.set_size_inches(6, 4)
    fig_preview.tight_layout()
    st.pyplot(fig_preview, use_container_width=False)
    plt.close(fig_preview)
    st.caption("👆 在左侧设置参数后，点击「开始规划」执行路径搜索")

# ==========================================
# 6. 核心算法（点击开始后执行）
# ==========================================
if start_btn:
    st.session_state.is_running = True
    
    if use_llm and not api_key:
        st.error("请在左侧输入 API Key！")
        st.session_state.is_running = False
        st.stop()
        
    if use_llm:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    with st.spinner("无人机正在规划路径，请稍候..."):
        try:
            real_prior_knowledge = {
                "Easy_Map": {"PSO": "收敛极快，总体最优。"},
                "Hard_Map_密集障碍": {"PSO": "展现出极强的破壁能力。"}
            }
            
            opt_agent = Algorithm_Select_Agent(
                evaluator=evaluator,
                prior_knowledge=real_prior_knowledge,
                rainfall_mm=rainfall_mm,
                duration_hours=duration_hours,
                use_llm=use_llm
            )
            initial_planner = opt_agent.make_decision()
            current_algo_name = type(initial_planner).__name__.replace('Planner', '')
            
            pop_size = getattr(initial_planner, 'num_particles',
                               getattr(initial_planner, 'num_sparrows',
                                       getattr(initial_planner, 'num_wolves',
                                               getattr(initial_planner, 'pop_size', 50))))
            coord_agent = CoordinatorAgent()
            coord_agent.algo_params = {
                'pop_size': pop_size,
                'max_iter': initial_planner.max_iter,
                'num_waypoints': initial_planner.num_waypoints
            }
            
            current_algo_params = coord_agent.algo_params
            current_specific_params = {}
            global_best_path = None
            global_best_score = float('inf')
            global_iteration_count = 0
            event_history = []
            MAX_META_ITERATIONS = 3
            final_history = []
            
            for meta_iter in range(1, MAX_META_ITERATIONS + 1):
                planner = create_planner_with_params(
                    algo_name=current_algo_name,
                    evaluator=evaluator,
                    algo_params=current_algo_params,
                    specific_params=current_specific_params,
                    elite_path=global_best_path
                )
                best_path, history = planner.optimize()
                final_history.extend(history)
                global_iteration_count += len(history)
                
                total_score, details, env_info = evaluator.evaluate_particle(best_path)
                
                if total_score < global_best_score:
                    global_best_score = total_score
                    global_best_path = best_path

                current_algo_params, new_eval_params, current_specific_params, is_finished, param_changes = \
                    coord_agent.analyze_and_act(global_best_score, details, env_info, current_algo_name)

                if param_changes:
                    event_history.append((global_iteration_count, "\n".join(param_changes)))
                    
                evaluator.params.update(new_eval_params)
                
                if is_finished:
                    break
                    
                if coord_agent.stuck_counter >= 2:
                    next_algo, new_pop, new_iter, new_wp = opt_agent.get_fallback_algorithm(current_algo_name, details)
                    current_algo_name = next_algo
                    coord_agent.algo_params.update({'pop_size': new_pop, 'max_iter': new_iter, 'num_waypoints': new_wp})
                    current_algo_params = coord_agent.algo_params
                    coord_agent.stuck_counter = 0
                    current_specific_params = {}

            if global_best_path is not None:
                output_root = "output"
                run_idx = 0
                run_subdir = os.path.join(output_root, f"run_{run_idx:02d}")
                os.makedirs(run_subdir, exist_ok=True)

                video_filename = os.path.join(run_subdir, f"{current_algo_name}_final_flight.mp4")
                animator = UAVVideoAnimator(evaluator)
                animator.create_flight_video(global_best_path, filename=video_filename, duration=10.0)

                figs = planner.plot_result(
                    best_path=global_best_path,
                    score_history=final_history,
                    algo_name=current_algo_name,
                    event_history=event_history,
                    run_idx=run_idx,
                    save_dir=output_root,
                    return_figs=True
                )
                _, _, fig_2d, fig_curve, detail_report = figs

                st.session_state.result_video = video_filename
                st.session_state.result_score = global_best_score
                st.session_state.fig_2d = fig_2d
                st.session_state.fig_curve = fig_curve
                st.session_state.detail_report = detail_report

        except Exception as e:
            import traceback
            st.error(f"❌ 运行报错: {e}")
            st.code(traceback.format_exc())
        finally:
            st.session_state.is_running = False
            if st.session_state.result_video is not None:
                st.rerun()