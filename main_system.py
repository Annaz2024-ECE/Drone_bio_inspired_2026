import numpy as np
import matplotlib.pyplot as plt

# 导入底层规划算法
from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from woa_planner import WOAPlanner
from dsaco_planner import DSACOPlanner
from aco_planner import ACOPlanner

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
    
    # 1. 根据名字实例化对应的底层算法
    if algo_name == "PSO":
        planner = PSOPlanner(evaluator=evaluator, num_particles=pop_size, max_iter=max_iter)
    elif algo_name == "SSA":
        planner = SSAPlanner(evaluator=evaluator, num_sparrows=pop_size, max_iter=max_iter)
    elif algo_name == "GWO":
        planner = GWOPlanner(evaluator=evaluator, num_wolves=pop_size, max_iter=max_iter)
    elif algo_name == "WOA":
        planner = WOAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter)
    elif algo_name == "DSACO":
        planner = DSACOPlanner(evaluator=evaluator, num_ants=pop_size, max_iter=max_iter)
    elif algo_name == "ACO":
        planner = ACOPlanner(evaluator=evaluator, num_ants=pop_size, max_iter=max_iter)
    else:
        planner = PSOPlanner(evaluator=evaluator)

    # 2. 【核心黑科技】：把队友的“专属药方”通过 setattr 强行注入算法底层
    for key, value in specific_params.items():
        if hasattr(planner, key) or key == 'num_producers': # 兼容队友写的 num_producers
            setattr(planner, key, value)

    # 3. 【精英传承】：防止多轮迭代丢失进度，把上一轮的最佳路线传给新种群的1号位
    if elite_path is not None:
        try:
            elite_1d = elite_path[1:-1].flatten() # 展平中间航点
            if hasattr(planner, 'particles'): planner.particles[0] = elite_1d
            elif hasattr(planner, 'sparrows'): planner.sparrows[0] = elite_1d
            elif hasattr(planner, 'positions'): planner.positions[0] = elite_1d
        except Exception:
            pass # 维度如果因宏观调控发生改变则跳过传承
            
    return planner

# ==========================================
# 理论框架图：闭环主运行流
# ==========================================
if __name__ == "__main__":
    print("\n" + "★" * 60)
    print("多智能体协同无人机路径规划系统 - 启动")
    print("★" * 60)

    # 1. 实例化评价器（包含地图环境）
    evaluator = PathEvaluator()
    # 模拟任务指令 (你可以修改为 'speed', 'safety', 'smoothness' 等测试)
    current_task = 'speed'
    
    # 2. 实例化三大智能体
    opt_agent = Algorithm_Select_Agent(evaluator=evaluator, task_priority=current_task)
    coord_agent = CoordinatorAgent()
    
    # 3. 【框图节点1】：优化智能体感知环境，决定首发阵容
    initial_planner = opt_agent.make_decision()
    current_algo_name = type(initial_planner).__name__.replace('Planner', '')
    
    # 记录全局最优
    global_best_path = None
    global_best_score = float('inf')
    full_convergence_history = []
    
    # 获取协调智能体的初始预算参数
    current_algo_params = coord_agent.algo_params
    current_specific_params = {}

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
        total_score, details = evaluator.evaluate_pso_particle(best_path)
        
        if total_score < global_best_score:
            global_best_score = total_score
            global_best_path = best_path

        # 【框图节点5 & 6】：协调决策智能体介入分析并开药方
        current_algo_params, new_eval_params, current_specific_params, is_finished = \
            coord_agent.analyze_and_act(global_best_score, details, current_algo_name)
            
        # 调整评价器物理规则 (比如增加 Chaikin 平滑次数)
        evaluator.params.update(new_eval_params)
        
        # 判断是否满足分数要求，提前交卷
        if is_finished:
            print("\n协调决策智能体审核通过：路线绝对安全，提前结束寻优！")
            break
            
        # 【框图节点7】：一定次数后仍不达标，尝试更换算法！
        if coord_agent.stuck_counter >= 3:
            print(f"\n[系统告警] {current_algo_name} 已连续 3 轮抢救无效！强制切换算法！")
            
            # 简单的换将逻辑 (你可以调用 opt_agent.switch_algorithm，这里写个通用的)
            fallback_dict = {"PSO": "SSA", "SSA": "DSACO", "GWO": "WOA", "WOA": "PSO", "DSACO": "ACO", "ACO": "GWO"}
            current_algo_name = fallback_dict.get(current_algo_name, "PSO")
            
            print(f"   -> 放弃当前算法，已更换为: {current_algo_name}")
            coord_agent.stuck_counter = 0 # 换将后重置卡壳计数器
            # 为了防止新算法受旧参数影响，清空专属药方
            current_specific_params = {} 

    # ==========================================
    # 寻优结束，调用最终画图
    # ==========================================
    print(f"\n终极规划完成！最终全局得分: {global_best_score:,.2f}")
    # 借用最后一个 planner 实例的画图功能
    planner.plot_result(global_best_path, full_convergence_history, algo_name=f"Final_{current_algo_name}")