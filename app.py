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
    max_iter = 20
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
# 2. Streamlit 页面配置与状态初始化
# ==========================================
st.set_page_config(page_title="智能体无人机指挥大脑", page_icon="🚁", layout="wide", initial_sidebar_state="expanded")

# 初始化全局状态变量 (解决按钮状态重置问题)
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = False
if 'result_video' not in st.session_state:
    st.session_state.result_video = None
if 'result_score' not in st.session_state:
    st.session_state.result_score = None
if 'result_history' not in st.session_state:
    st.session_state.result_history = None   
if 'fig_2d' not in st.session_state:
    st.session_state.fig_2d = None
if 'fig_curve' not in st.session_state:
    st.session_state.fig_curve = None

st.title("智能体无人机指挥大脑 (LLM-Assisted UAV Planner)")
st.markdown("集成 Coordinator Agent 多智能体协同的 3D 复杂空域巡检路径规划系统")

# ==========================================
# 3. 侧边栏：输入控制区
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
    
    # 【核心】：动态按钮逻辑。如果正在运行，显示停止按钮；否则显示开始按钮。
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 开始规划", type="primary", use_container_width=True, disabled=st.session_state.is_running)
    with col2:
        stop_btn = st.button("⛔ 紧急停止", type="secondary", use_container_width=True, disabled=not st.session_state.is_running)

    # ---------- 新增重置按钮（放在最底部） ----------
    st.markdown("---")
    if st.button("🔄 重置结果", use_container_width=True):
        st.session_state.result_video = None
        st.session_state.result_score = None
        st.session_state.result_history = None
        st.session_state.fig_2d = None
        st.session_state.fig_curve = None
        st.rerun()
    # ----------------------------------------------

# 处理按钮点击事件
if start_btn:
    st.session_state.is_running = True
    st.session_state.stop_flag = False
    st.rerun() # 刷新页面，让按钮状态生效

if stop_btn:
    st.session_state.stop_flag = True
    st.warning("⛔ 已发送停止指令，正在安全中断后台计算...")
    # 不立刻 rerun，等待下方的主逻辑捕获到 stop_flag 后自己清理

# ==========================================
# 4. 主干逻辑 (包含 Coordinator Agent 大循环)
# ==========================================
if st.session_state.is_running:
    
    if use_llm and not api_key:
        st.error("请在左侧输入 API Key！")
        st.session_state.is_running = False
        st.stop()
        
    if use_llm:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    # 准备页面上的输出占位符，实现动态更新
    progress_bar = st.progress(0)
    status_text = st.empty()
    terminal_log = st.empty()
    logs = []

    def log(msg):
        logs.append(msg)
        # 只保留最新的 15 行日志在界面上，防止太长
        terminal_log.code("\n".join(logs[-15:]), language="text")

    try:
        status_text.info("🔄 正在加载地图环境...")
        evaluator = PathEvaluator()
        
        if not os.path.exists(selected_map_path):
            st.error(f"找不到地图文件: {selected_map_path}")
            st.session_state.is_running = False
            st.stop()
            
        evaluator.env = UAVEnvironment3D(selected_map_path) 
        
        prior_knowledge_file = "prior_knowledge.json"
        if os.path.exists(prior_knowledge_file):
            with open(prior_knowledge_file, "r", encoding='utf-8') as f:
                real_prior_knowledge = json.load(f)
        else:
            # Fallback if the file hasn't been generated yet
            real_prior_knowledge = {
                "Easy_Map": {"PSO": "收敛极快，总体最优。"},
                "Hard_Map_密集障碍": {"SSA": "展现出极强的破壁能力。"}
            }
        
        status_text.info("LLM 气象感知智能体评估环境，决断首发阵容...")
        opt_agent = Algorithm_Select_Agent(
            evaluator=evaluator,
            prior_knowledge=real_prior_knowledge,
            rainfall_mm=rainfall_mm, 
            duration_hours=duration_hours, 
            use_llm=use_llm
        )
        initial_planner = opt_agent.make_decision()
        current_algo_name = type(initial_planner).__name__.replace('Planner', '')
        
        log(f"LLM 首发决断: {current_algo_name}")
        
        # 协调决策智能体介入
        pop_size = getattr(initial_planner, 'num_particles', getattr(initial_planner, 'num_sparrows', getattr(initial_planner, 'num_wolves', getattr(initial_planner, 'pop_size', 50))))
        coord_agent = CoordinatorAgent()
        coord_agent.algo_params = {'pop_size': pop_size, 'max_iter': initial_planner.max_iter, 'num_waypoints': initial_planner.num_waypoints}
        
        current_algo_params = coord_agent.algo_params
        current_specific_params = {}
        
        global_best_path = None
        global_best_score = float('inf')
        global_iteration_count = 0  
        event_history = []
        MAX_META_ITERATIONS = 2 # 前端演示，为了速度这里设为 3 个大轮
        
        global_start_time = time.time()
        
        final_history = [] # 用于最终画图的扁平化得分历史

        # ==========================================
        # 🚀 Coordinator Agent 大循环
        # ==========================================
        for meta_iter in range(1, MAX_META_ITERATIONS + 1):
            
            # 【核心】：检查网页前端是否按下了停止按钮！
            if st.session_state.stop_flag:
                log(f"⛔ 接收到人工终止指令！")
                status_text.warning("⛔ 寻优已人工终止！")
                break
                
            progress_bar.progress(meta_iter / MAX_META_ITERATIONS)
            status_text.info(f"⚙️ 第 {meta_iter}/{MAX_META_ITERATIONS} 轮寻优 | 算法: {current_algo_name}")
            log(f"--- 开启第 {meta_iter} 轮协同寻优 [{current_algo_name}] ---")
            
            # 实例化底层仿生学算法
            planner = create_planner_with_params(
                algo_name=current_algo_name, evaluator=evaluator, 
                algo_params=current_algo_params, specific_params=current_specific_params,
                elite_path=global_best_path
            )
            
            # 我们需要让底层算法的 optimize 也能响应停止按钮！
            # (这需要修改底层代码，或者在这里我们只能等当前大轮跑完。为了不改你底层代码，我们在大轮之间停止即可)
            best_path, history = planner.optimize()
            final_history.extend(history)
            
            global_iteration_count += len(history)
            
            # 评价与记录
            total_score, details, env_info = evaluator.evaluate_particle(best_path)
            log(f"结算得分: {total_score:,.2f}")
            
            if total_score < global_best_score:
                global_best_score = total_score
                global_best_path = best_path

            # 协调智能体看病
            current_algo_params, new_eval_params, current_specific_params, is_finished, param_changes = \
                coord_agent.analyze_and_act(global_best_score, details, env_info, current_algo_name)

            if param_changes:
                exact_tag = "\n".join(param_changes)
                event_history.append((global_iteration_count, exact_tag))
          #      log(f"[特工干预] 触发硬核物理切变")
                
            evaluator.params.update(new_eval_params)
            
            if is_finished:
                log("✅ 协调智能体审核通过：路线绝对安全！")
                break
                
            # 换将机制
            if coord_agent.stuck_counter >= 2:
                log(f"{current_algo_name} 抢救无效！呼叫 LLM 换将...")
                next_algo, new_pop, new_iter, new_wp = opt_agent.get_fallback_algorithm(current_algo_name, details)
                current_algo_name = next_algo
                coord_agent.algo_params.update({'pop_size': new_pop, 'max_iter': new_iter, 'num_waypoints': new_wp})
                current_algo_params = coord_agent.algo_params
                coord_agent.stuck_counter = 0 
                current_specific_params = {}

        # ==========================================
        # 5. 循环结束，结果渲染区
        # ==========================================
        if global_best_path is not None:
            status_text.success(f"🏁 终极规划完成！最优得分: {global_best_score:,.2f}。正在渲染 3D 视频...")
            progress_bar.progress(1.0)
            
            output_root = "output"
            run_idx = 0   # 固定编号，每次运行覆盖
            run_subdir = os.path.join(output_root, f"run_{run_idx:02d}")
            os.makedirs(run_subdir, exist_ok=True)

            video_filename = os.path.join(run_subdir, f"{current_algo_name}_final_flight.mp4")
            
            # 渲染视频 (这里使用 10 秒加速演示)
            animator = UAVVideoAnimator(evaluator)
            animator.create_flight_video(global_best_path, filename=video_filename, duration=10.0)

            # ---------- 新增：调用 BasePlanner 的绘图方法，获取 2D 图和收敛曲线 ----------
            figs = planner.plot_result(
                best_path=global_best_path,
                score_history=final_history,
                algo_name=current_algo_name,
                event_history=event_history,
                run_idx=run_idx,                 # 保存到 output/run_00/
                save_dir=output_root,
                return_figs=True                 # 返回图形供显示
            )
            # 解包：顺序为 speed, 3d, 2d, curve
            _, _, fig_2d, fig_curve = figs

            # 存入 session_state
            st.session_state.fig_2d = fig_2d
            st.session_state.fig_curve = fig_curve
            # ---------------------------------------------------------------------

            # 将结果存入 session_state
            st.session_state.result_video = video_filename
            st.session_state.result_score = global_best_score
            st.session_state.result_history = final_history   # 如有需要

    except Exception as e:
        import traceback
        st.error(f"❌ 运行报错: {e}")
        st.code(traceback.format_exc())
    finally:
        # 重置运行状态，恢复按钮
        st.session_state.is_running = False
        st.session_state.stop_flag = False

# ---------- 结果展示区（独立于运行状态） ----------
if st.session_state.result_video and os.path.exists(st.session_state.result_video):
    st.success(f"🏁 规划完成！最优得分: {st.session_state.result_score:,.2f}")
    
    # ---------- 第一行：视频 ----------
    st.subheader("🎥 3D 飞行仿真")
    st.markdown("<br>", unsafe_allow_html=True)   # 增加一个空行，让标题更舒展
    st.video(st.session_state.result_video)       # 无任何参数，自动填满宽度
    
    # ---------- 第二行：2D图和收敛曲线并排 ----------
    col_left, col_right = st.columns(2)   # 各占一半宽度
    with col_left:
        st.subheader("🗺️ 2D 速度热力")
        st.pyplot(st.session_state.fig_2d)
        plt.close(st.session_state.fig_2d)
    with col_right:
        st.subheader("📈 收敛与干预")
        st.pyplot(st.session_state.fig_curve)
        plt.close(st.session_state.fig_curve)
        
else:
    st.info("等待输入指令")