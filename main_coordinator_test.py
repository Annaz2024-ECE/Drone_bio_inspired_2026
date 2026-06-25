from path_evaluator import PathEvaluator
from aco_planner import ACOPlanner
from coordinator_agent import CoordinatorAgent

def run_meta_optimization():
    print("=" * 50)
    print("启动多智能体协调决策系统 (Meta-Optimization)")
    print("=" * 50)

    # 1. 初始化独立组件
    evaluator = PathEvaluator()
    agent = CoordinatorAgent()
    
    # 设定元循环次数 (比如智能体调参 5 轮)
    meta_rounds = 5
    
    for round_idx in range(1, meta_rounds + 1):
        print(f"\n>>>>>>>>>>>> 第 {round_idx} 轮算法执行 >>>>>>>>>>>>")
        
        # 2. 从智能体获取当前最新的参数
        algo_params = agent.algo_params
        eval_params = agent.eval_params
        
        # 3. 将评价器参数更新
        evaluator.update_params(new_params=eval_params)
        
        # 4. 实例化算法（注意：要把 generic 的 pop_size 映射给 ACO 的 num_ants）
        planner = ACOPlanner(
            evaluator=evaluator,
            num_ants=algo_params['pop_size'],  # 动态映射！如果是PSO就是 num_particles
            max_iter=algo_params['max_iter'],  # 动态映射！
            num_waypoints=9
        )
        
        # 5. 运行这套参数下的算法
        best_path, history = planner.optimize()
        
        # 6. 对算法找出的“最佳路径”做一次终极体检
        final_score, details = evaluator.evaluate_pso_particle(best_path)
        print(f"\n[本轮结果] 最终得分: {final_score:,.2f}")
        for k, v in details.items():
            if v > 0:
                print(f"    - {k}: {v:,.2f}")
                
        # 7. 绘图 (可选，如果想看每轮的变化可以取消注释)
        # planner.plot_result(best_path, history, algo_name="ACO", run_idx=round_idx)
        
        # 8. 闭环核心：智能体根据体检报告，决定下一轮怎么调参！
        if round_idx < meta_rounds: # 最后一轮就不调了
            agent.analyze_and_act(final_score, details)

if __name__ == "__main__":
    run_meta_optimization()