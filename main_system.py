import numpy as np
import matplotlib.pyplot as plt

# 导入底层规划算法
from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from woa_planner_fix import WOAPlanner
#from dsaco_planner import DSACOPlanner
from hybrid_pso_gwo import HybridPSOGWO

# 导入三大智能体与评价器
from path_evaluator import PathEvaluator
from algorithm_select_agent import Algorithm_Select_Agent
from coordinator_agent import CoordinatorAgent

def create_planner_with_params(algo_name, evaluator, algo_params, specific_params, elite_path=None):
    """
    【算法装配工厂】：根据协调智能体给出的参数，组装并生成底层规划器
    """
    pop_size = algo_params.get('pop_size', 50)
    max_iter = algo_params.get('max_iter', 100)
    
    # 【核心修复】：必须接收智能体计算出的基因长度，不能使用默认值！
    num_waypoints = algo_params.get('num_waypoints', 30) 
    
    # 1. 根据名字实例化对应的底层算法，并严格传入 num_waypoints
    if algo_name == "PSO":
        planner = PSOPlanner(evaluator=evaluator, num_particles=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "SSA":
        planner = SSAPlanner(evaluator=evaluator, num_sparrows=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "GWO":
        planner = GWOPlanner(evaluator=evaluator, num_wolves=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "WOA":
        planner = WOAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "DSACO":
        planner = DSACOPlanner(evaluator=evaluator, num_ants=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "ACO":
        planner = ACOPlanner(evaluator=evaluator, num_ants=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "HybridPSOGWO":
        planner = HybridPSOGWO(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    else:
        planner = PSOPlanner(evaluator=evaluator, num_waypoints=num_waypoints)

    # 2. 【核心黑科技】：把老中医的“专属药方”(如四大物理指令) 通过 setattr 强行注入算法底层
    for key, value in specific_params.items():
        if hasattr(planner, key) or key in ['num_producers', 'lift_up', 'press_down', 'radar_guidance', 'emergency_escape']:
            setattr(planner, key, value)

    # 3. 【精英传承】：防止多轮迭代丢失进度，把上一轮的最佳路线传给新种群的1号位
    if elite_path is not None:
        try:
            elite_1d = elite_path[1:-1].flatten() # 展平中间航点
            if hasattr(planner, 'particles'): planner.particles[0] = elite_1d
            elif hasattr(planner, 'sparrows'): planner.sparrows[0] = elite_1d
            elif hasattr(planner, 'positions'): planner.positions[0] = elite_1d
            # GWO 使用的是 positions，所以也能兼容
        except Exception as e:
            print(f"  [系统提示] 精英传承跳过 (可能因维度变更): {e}")
            
    return planner

# ==========================================
# 理论框架图：闭环主运行流
# ==========================================
if __name__ == "__main__":
    print("\n" + "★" * 65)
    print(" 多智能体协同无人机 3D 路径规划系统 - 启动 (气象 LLM Ready)")
    print("★" * 65)

    # 1. 实例化评价器（确保默认加载紫金港等大型地图）
    evaluator = PathEvaluator()
    
    # ========================================================
    # 2. 模拟外部气象 API 数据输入 (你可以修改这里测试不同天气)
    # ========================================================
    current_rainfall = 70.0   # 累计降雨量 mm (暴雨)
    current_duration = 20.0    # 持续时间 h
    
    # 3. 【框图节点1】：气象感知智能体评估环境，动态裁剪目标，决定首发阵容
    opt_agent = Algorithm_Select_Agent(
        evaluator=evaluator, 
        rainfall_mm=current_rainfall, 
        duration_hours=current_duration,
        use_llm=True  # 等你 LLM 接口写好，这里改成 True 就行
    )
    initial_planner = opt_agent.make_decision()
    current_algo_name = type(initial_planner).__name__.replace('Planner', '')
    
    # ========================================================
    # 4. 【系统握手】：将气象智能体算出的完美基因长度，移交给协调老中医
    # ========================================================
    # 动态获取不同算法种群数量的变量名
    pop_size = getattr(initial_planner, 'num_particles',
               getattr(initial_planner, 'num_sparrows',
               getattr(initial_planner, 'num_wolves',
               getattr(initial_planner, 'pop_size',
               getattr(initial_planner, 'num_ants', 50)))))
               
    # 协调决策智能体介入
    coord_agent = CoordinatorAgent()
    
    # 【最关键的一步】：同步参数！保证大循环里不截断控制点！
    coord_agent.algo_params = {
        'pop_size': pop_size,
        'max_iter': initial_planner.max_iter,
        #'max_iter': 5,
        'num_waypoints': initial_planner.num_waypoints  
    }
    current_algo_params = coord_agent.algo_params
    current_specific_params = {}
    
    # 记录全局最优
    global_best_path = None
    global_best_score = float('inf')
    full_convergence_history = []
    
    MAX_META_ITERATIONS = 5 # 允许协调智能体最多干预的大轮次

    for meta_iter in range(1, MAX_META_ITERATIONS + 1):
        print(f"\n" + "=" * 50)
        print(f"【第 {meta_iter} 大轮寻优开始】 当前算法: {current_algo_name}")
        print("=" * 50)
        
        # 【框图节点2】：实例化底层仿生学算法
        planner = create_planner_with_params(
            algo_name=current_algo_name, 
            evaluator=evaluator, 
            algo_params=current_algo_params, 
            specific_params=current_specific_params,
            elite_path=global_best_path
        )
        
        # 【框图节点3】：仿生学算法执行并输出路线
        best_path, history = planner.optimize()
        
        # 更新全局记录
        full_convergence_history.extend(history)
        
        # 【框图节点4】：路径评价智能体打分 (获取分数与明细)
        total_score, details, env_info = evaluator.evaluate_pso_particle(best_path)
        
        if total_score < global_best_score:
            global_best_score = total_score
            global_best_path = best_path

        # 【框图节点5 & 6】：协调决策智能体介入分析并开药方
        current_algo_params, new_eval_params, current_specific_params, is_finished = \
            coord_agent.analyze_and_act(global_best_score, details, env_info, current_algo_name)
            
        # 调整评价器物理规则
        evaluator.params.update(new_eval_params)
        
        # 判断是否满足分数要求，提前交卷
        if is_finished:
            print("\n协调决策智能体审核通过：路线绝对安全，提前结束寻优！")
            break
            
        # 【框图节点7】：一定次数后仍不达标，尝试更换算法！
        if coord_agent.stuck_counter >= 2:
     #   if True:
            print(f"\n[系统告警] {current_algo_name} 已连续 2 轮抢救无效！触发智能换将机制！")
            
            # ==========================================
            # 【修改这里】：将本轮失败的具体 details 明细，当做病历本传给指挥大脑
            # ==========================================
            next_algo, new_pop, new_iter, new_wp = opt_agent.get_fallback_algorithm(current_algo_name, details)
            
            current_algo_name = next_algo
            
            # 2. 必须把新算法专属的控制点数量和算力，强行同步给老中医！
            coord_agent.algo_params['pop_size'] = new_pop
            coord_agent.algo_params['max_iter'] = new_iter
            coord_agent.algo_params['num_waypoints'] = new_wp
            current_algo_params = coord_agent.algo_params
            
            # 3. 清空历史包袱，重新开始
            coord_agent.stuck_counter = 0 
            current_specific_params = {}

    # ==========================================
    # 寻优结束，调用最终画图
    # ==========================================
    print(f"\n终极规划完成！最终全局得分: {global_best_score:,.2f}")
    # 借用最后一个 planner 实例的画图功能
    planner.plot_result(global_best_path, full_convergence_history, algo_name=f"Final_{current_algo_name}")