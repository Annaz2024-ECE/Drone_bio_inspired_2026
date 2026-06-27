from wakepy import keep #挂后台跑
from path_evaluator import PathEvaluator
from coordinator_agent import CoordinatorAgent

# 1. 导入你手上所有的仿生智能体兵器库！
from aco_planner import ACOPlanner
from dsaco_planner import DSACOPlanner
from pso_planner import PSOPlanner
from gwo_planner import GWOPlanner
from ssa_planner import SSAPlanner
from woa_planner import WOAPlanner

def run_parameter_tuning_loop():
    print("=" * 60)
    print(" 启动【万能算法专属参数调优】试车场")
    print("=" * 60)

    evaluator = PathEvaluator()
    agent = CoordinatorAgent()
    
    # ==========================================
    # 🌟 核心修改：算法字典映射与一键切换
    # ==========================================
    ALGO_MAP = {
        "ACO": ACOPlanner,
        "DSACO": DSACOPlanner,
        "PSO": PSOPlanner,
        "GWO": GWOPlanner,
        "SSA": SSAPlanner,
        "WOA": WOAPlanner
    }
    
    # 🎯 你只需修改这里！想测谁，就改成谁的名字
    TARGET_ALGO = "ACO"  # 比如今天你想测试一下 PSO 的专属调参
    # ==========================================
    
    print(f"  [系统加载] 正在实例化 {TARGET_ALGO} 施工队...")
    PlannerClass = ALGO_MAP[TARGET_ALGO]
    
    # ==========================================
    # 在实例化前，就打包好正确的参数字典 (kwargs)
    # ==========================================
    kwargs = {
        'evaluator': evaluator,
        'num_waypoints': 10,
        'max_iter': agent.algo_params['max_iter']
    }
    
    # 精准对接各个算法底层所需的变量名
    pop_size = agent.algo_params['pop_size']
    if TARGET_ALGO in ["ACO", "DSACO"]: kwargs['num_ants'] = pop_size
    elif TARGET_ALGO == "PSO": kwargs['num_particles'] = pop_size
    elif TARGET_ALGO == "GWO": kwargs['num_wolves'] = pop_size
    elif TARGET_ALGO == "SSA": kwargs['num_sparrows'] = pop_size
    elif TARGET_ALGO == "WOA": kwargs['pop_size'] = pop_size

    # 带着正确的种群规模出生，底层矩阵直接完美生成 50x20！
    planner = PlannerClass(**kwargs)
    
    meta_rounds = 5  # 调参总轮数
    
    for round_idx in range(1, meta_rounds + 1):
        print(f"\n>>>>>>>>>>>>  第 {round_idx} 轮调优测试 [{TARGET_ALGO}] >>>>>>>>>>>>")
        
        # 1. 跑当前参数下的算法
        best_path, history = planner.optimize()
        
        # 2. 终极体检
        final_score, details = evaluator.evaluate_pso_particle(best_path)
        
        print(f"\n [本轮结算] 得分: {final_score:,.2f}")
        for k, v in details.items():
            if v > 0: print(f"    - {k}: {v:,.2f}")
            
        # 3. 提交给老中医，获取更新后的三个字典，以及是否结束的信号
        if round_idx < meta_rounds:
            # 接收新增的第四个返回值
            algo_params, eval_params, specific_params, is_finished = agent.analyze_and_act(final_score, details, TARGET_ALGO)
            
            # 接收到提前交卷信号，直接跳出循环
            if is_finished:
                print(f"\n {TARGET_ALGO} 调教完毕！在第 {round_idx} 轮提前达成完美收敛。")
                break
                
            # 【A】更新评价器参数
            evaluator.update_params(new_params=eval_params)
            
            # 【B】更新算法共性参数 (迭代次数可以随便改，但实体矩阵的种群规模不能中途改)
            planner.max_iter = algo_params['max_iter']
            
            if TARGET_ALGO in ["ACO", "DSACO"]:
                # 只有 ACO/DSACO 这种离散算法，每代会重新撒蚂蚁，才允许中途无缝加兵力
                planner.num_ants = algo_params['pop_size']
            else:
                # 连续型算法 (PSO/GWO/SSA/WOA) 含有固定矩阵，禁止中途加人！
                # 强制把老中医字典里的 pop_size 改回底层原本的真实人数，防止老中医自己记错账
                current_pop = getattr(planner, 'num_particles', 
                              getattr(planner, 'num_wolves', 
                              getattr(planner, 'num_sparrows', 
                              getattr(planner, 'pop_size', 50))))
                
                algo_params['pop_size'] = current_pop
            
            # 【C】动态注入算法“专属参数”！
            for param_key, param_value in specific_params.items():
                if hasattr(planner, param_key):
                    setattr(planner, param_key, param_value)
                    print(f"  └──  [专属注入] 成功将 {TARGET_ALGO} 的 {param_key} 设为 {param_value}")

    # ==========================================
    #  所有调参轮次彻底结束后，输出终极图表
    # ==========================================
    print(f"\n 全部调优轮次结束！正在生成 {TARGET_ALGO} 的终极路线与连续收敛曲线图...")
    planner.plot_result(best_path, history, algo_name=f"{TARGET_ALGO}_Final_Tuned")

if __name__ == "__main__":
    print("\n 开始底层寻优，已开启防休眠模式...")
    
    # 用 with 语句把整个主程序的大循环包起来
    with keep.running():
        # 注意这一行一定要有缩进！
        run_parameter_tuning_loop()
        
    print("\n 规划全部完成，电脑恢复正常休眠策略。")