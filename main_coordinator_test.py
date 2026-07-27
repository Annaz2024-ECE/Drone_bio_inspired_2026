#from wakepy import keep #挂后台跑
from path_evaluator import PathEvaluator
from coordinator_agent import CoordinatorAgent

# 1. 导入你手上所有的仿生智能体兵器库！
from pso_planner import PSOPlanner
from gwo_planner import GWOPlanner
from ssa_planner import SSAPlanner
from woa_planner import WOAPlanner
from ga_planner import GAPlanner
from woa_planner import WOAPlanner
from ga_planner import GAPlanner

import time
import os
import numpy as np

def run_parameter_tuning_loop(run_idx=None, save_dir=None):
    print("=" * 60)
    print(" 启动 [3D 空间算法专属参数调优 Agent]")
    print("=" * 60)

    evaluator = PathEvaluator()
    agent = CoordinatorAgent()
    
    # ==========================================
    # 算法字典映射与一键切换
    # ==========================================
    ALGO_MAP = {
        "PSO": PSOPlanner,
        "GWO": GWOPlanner,
        "SSA": SSAPlanner,
        "WOA": WOAPlanner,
        "GA": GAPlanner
    }
    
    # 你只需修改这里！想测谁，就改成谁的名字
    TARGET_ALGO = "SSA" 
    
    print(f"  [系统加载] 正在实例化 3D {TARGET_ALGO} 算法矩阵...")
    PlannerClass = ALGO_MAP[TARGET_ALGO]
    
    # ==========================================
    # 在实例化前，打包好正确的参数字典 (kwargs)
    # ==========================================
    kwargs = {
        'evaluator': evaluator,
        # 紫金港地图目标较多，控制点建议调大至 40-50 左右
        # 海宁设置为16 比较合适
        'num_waypoints': 12, 
        'max_iter': agent.algo_params['max_iter']
       # 'max_iter': 10
    }
    
    # 精准对接各个算法底层所需的变量名
    pop_size = agent.algo_params['pop_size']
    if TARGET_ALGO == "PSO": kwargs['num_particles'] = 100
    elif TARGET_ALGO == "GWO": kwargs['num_wolves'] = pop_size
    elif TARGET_ALGO == "SSA": kwargs['num_sparrows'] = 100
    elif TARGET_ALGO in ["WOA"]: kwargs['pop_size'] = pop_size 

    # 带着正确的种群规模出生，底层 3D 矩阵直接完美生成！
    planner = PlannerClass(**kwargs)
    
    meta_rounds = 5  # 调参总轮数
    meta_rounds = 5  # 调参总轮数

    # ==========================================
    # 【新增】：大循环启动前，初始化所有全局追踪器
    # ==========================================
    import time
    global_start_time = time.time()       # 1. 记录系统总秒表
   # full_convergence_history = []         # 2. 拼接所有轮次的分数，画出连续的长折线
    global_iteration_count = 0            # 3. 记录当前跑到第几代了 (作为图表的 X 轴坐标)
    event_history = []                    # 4. 记录特工在哪个坐标点下发了什么药方
    
    for round_idx in range(1, meta_rounds + 1):
        print(f"\n>>>>>>>>>>>>  第 {round_idx} 轮 3D 调优测试 [{TARGET_ALGO}] >>>>>>>>>>>>")
        
        # 在改了规则后，重新核算历史最佳路线的基准分
        if round_idx > 1 and hasattr(planner, 'historical_best_pos'):
            # 拿老路线在新评价器里跑一次，获取当前规则下的真实分数
            true_benchmark, _, _ = evaluator.evaluate_particle(planner._decode_path(planner.historical_best_pos))
            # 刷新算法的记忆
            planner.historical_best_score = true_benchmark
            
            # 如果是 GWO，还要顺便刷新 Alpha 狼的记忆
            if hasattr(planner, 'alpha_score'):
                planner.alpha_score = true_benchmark

        # 1. 跑当前参数下的算法
        best_path, history = planner.optimize()

        # ==========================================
        # 【新增】：立刻把本轮成绩汇入全局记录簿！
        # ==========================================
        global_iteration_count = len(history)
       # full_convergence_history.extend(history)
        
        # 2. 终极体检
        final_score, details, env_info = evaluator.evaluate_particle(best_path)
        
        print(f"\n [本轮结算] 3D 最终得分: {final_score:,.2f}")
        for k, v in details.items():
            if v > 0: 
                color = "\033[91m" if v > 1000 else "\033[0m"
                print(f"    - {k}: {color}{v:,.2f}\033[0m")
            
        # 3. 提交给老中医，获取更新后的三个字典，以及是否结束的信号
        if round_idx < meta_rounds:
            # 接收模块化 Agent 返回的字典 (融合了宏观物理与微观数学)
            algo_params, eval_params, specific_params, is_finished, param_changes = agent.analyze_and_act(final_score, details, env_info, TARGET_ALGO)
            
            # ==========================================
            #【修改】：不再提取字符串，直接使用精准的 Diff 数据
            # ==========================================
            if param_changes:
                # 用换行符把所有改变的参数拼成一个小文本块
                exact_tag = "\n".join(param_changes)
                
                # 记录：在当前全局迭代次数点，打上这个硬核数值标签
                event_history.append((global_iteration_count, exact_tag))

                # 【新增】：在终端用高亮颜色实时打印出调参明细
                print(f"\n  [\033[96m干预档案\033[0m] 在第 {global_iteration_count} 代采取行动:")
                for change in param_changes:
                    print(f"     {change}")

            # 接收到提前交卷信号，直接跳出循环
            if is_finished:
                print(f"\n {TARGET_ALGO} 调参完毕！在第 {round_idx} 轮提前达成完美的 3D 空间收敛。")
                break
                
            # 【A】更新评价器参数
            evaluator.update_params(new_params=eval_params)
            
            # 【B】更新算法共性参数 (强制保护矩阵维度)
            planner.max_iter = algo_params['max_iter']
           # planner.max_iter = 10
            
            if TARGET_ALGO in ["ACO", "DSACO"]:
                planner.num_ants = algo_params['pop_size']
            else:
                current_pop = getattr(planner, 'num_particles', 
                              getattr(planner, 'num_wolves', 
                              getattr(planner, 'num_sparrows', 
                              getattr(planner, 'pop_size', 50))))
                
                algo_params['pop_size'] = current_pop
                #  Agent 的内部记忆，彻底消除 Diff 幻觉！
                agent.algo_params['pop_size'] = current_pop
            
            # ==========================================
            # 【C】核心修复：强制参数注入！
            # 删除了 hasattr 的限制，直接强制写入底层对象！
            # 这样 apply_laplacian 和 apply_repulsion 才能完美下发生效！
            # ==========================================
            for param_key, param_value in specific_params.items():
                setattr(planner, param_key, param_value)
                print(f"  └──  [参数下发] 成功将底层 {param_key} 设为 {param_value}")
            # ==========================================

    # ==========================================
    #  所有调参轮次彻底结束后，输出终极图表
    # ==========================================
    print(f"\n 全部调优轮次结束！正在生成 {TARGET_ALGO} 的终极 3D 路线与连续收敛曲线图...")
    # 【修改】：传入全长历史、全局开始时间、以及特工事件簿！
    planner.plot_result(
        best_path, 
        history, 
        algo_name=f"{TARGET_ALGO}_Final_3D_Tuned",
        run_idx=run_idx,
        save_dir=save_dir,
        global_start_time=global_start_time,
        event_history=event_history
    )

    return final_score

# ==========================================
# 【修改3】主程序：跑 10 次并保存结果
# ==========================================
if __name__ == "__main__":
    print("\n 开始底层 3D 寻优，已开启防休眠模式...")
    
    # 创建保存目录
    save_dir = "PSO_Agent_RandomMap"
    os.makedirs(save_dir, exist_ok=True)
    
    num_runs = 1
    all_scores = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*20} 第 {run_idx+1}/{num_runs} 次运行 {'='*20}")
        final_score = run_parameter_tuning_loop(run_idx=run_idx, save_dir=None)
        all_scores.append(final_score)
    
    # 输出统计汇总
    print("\n" + "="*50)
    print("所有运行完成！结果保存在", save_dir)
    print("各次最终得分:")
    for i, score in enumerate(all_scores):
        print(f"  Run {i+1:02d}: {score:,.2f}")
    if all_scores:
        avg = np.mean(all_scores)
        std = np.std(all_scores)
        print(f"\n平均得分: {avg:,.2f}  (±{std:,.2f})")
    print("="*50)

    