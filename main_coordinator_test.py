from path_evaluator import PathEvaluator
from aco_planner import ACOPlanner
from coordinator_agent import CoordinatorAgent

def run_meta_optimization():
    print("=" * 50)
    print("启动【记忆继承版】多智能体协调决策系统")
    print("=" * 50)

    # 1. 初始化独立组件
    evaluator = PathEvaluator()
    agent = CoordinatorAgent()
    
    # 2. 在双层循环外部只实例化一次算法对象！
    # 这样它的 self.pheromone 矩阵、self.global_best_path 将在整个生命周期内常驻内存，保留记忆
    planner = ACOPlanner(
        evaluator=evaluator,
        num_ants=agent.algo_params['pop_size'],  # 初始蚂蚁数
        max_iter=agent.algo_params['max_iter'],  # 初始迭代数
        num_waypoints=9
    )
    
    # 大脑干预的总轮数 (元迭代)
    meta_rounds = 5 
    
    for round_idx in range(1, meta_rounds + 1):
        print(f"\n>>>>>>>>>>>> 大脑干预循环：第 {round_idx} 轮算法执行 >>>>>>>>>>>>")
        print(f"  [当前运行配置] 蚂蚁数量: {planner.num_ants} | 迭代步数: {planner.max_iter}")
        
        # 3. 运行算法
        # 第一轮它是白纸；第二轮开始，它将带着上一轮沉淀下来的高浓度信息素和全局最优解继续破局！
        best_path, history = planner.optimize()
        
        # 4. 对本轮最终找出的“最佳路径”做一次终极体检
        final_score, details = evaluator.evaluate_pso_particle(best_path)
        print(f"\n[本轮结算表] 最终得分: {final_score:,.2f}")
        for k, v in details.items():
            if v > 0:
                print(f"    - {k}: {v:,.2f}")
                
        # 5. 画图看每一轮的路径进化（可选，若嫌弹窗烦可以注释掉）
        # planner.plot_result(best_path, history, algo_name="ACO_Inherited", run_idx=round_idx)
        
        # 6. 闭环核心：智能体根据体检报告，开出下一轮的调参处方
        if round_idx < meta_rounds: 
            algo_params, eval_params = agent.analyze_and_act(final_score, details)
            
            # 7. 【核心修改】：绝不销毁算法对象，直接将智能体的新指令“注入”到现有的评价器和算法属性中
            evaluator.update_params(new_params=eval_params)
            
            # 动态刷新现有蚂蚁集群的规模和每轮工期，而保留其核心的信息素记忆
            planner.num_ants = algo_params['pop_size']
            planner.max_iter = algo_params['max_iter']
            
            print(f"  └── 📡 \033[92m【记忆成功继承】\033[0m 上轮信息素已保留。下轮动态调整蚂蚁为 {planner.num_ants} 只，迭代 {planner.max_iter} 步。")

if __name__ == "__main__":
    run_meta_optimization()