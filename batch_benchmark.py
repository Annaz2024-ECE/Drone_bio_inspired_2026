import numpy as np
import json
import time
import os
import sys
from tqdm import tqdm # You might need to `pip install tqdm` for progress bars

from environment_buildup_3D import UAVEnvironment3D
from path_evaluator import PathEvaluator
from coordinator_agent import CoordinatorAgent

from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from woa_planner import WOAPlanner
from ga_planner import GAPlanner

def run_single_trial(evaluator, algo_name, use_coordinator):
    """
    Runs a single optimization trial.
    """
    num_targets = len(evaluator.env.target_areas)
    # A generic heuristic for num_waypoints based on map complexity
    num_waypoints = int(num_targets * 1.5) 
    if num_targets < 8:
        max_iter = 50
    elif num_targets < 11:
        max_iter = 100
    else:
        max_iter = 150
    pop_size = 80 

    algo_params = {'pop_size': pop_size, 'max_iter': max_iter, 'num_waypoints': num_waypoints}
    
    # Instantiate the planner
    if algo_name == "PSO": planner = PSOPlanner(evaluator=evaluator, num_particles=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "SSA": planner = SSAPlanner(evaluator=evaluator, num_sparrows=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "GWO": planner = GWOPlanner(evaluator=evaluator, num_wolves=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "WOA": planner = WOAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    elif algo_name == "GA": planner = GAPlanner(evaluator=evaluator, pop_size=pop_size, max_iter=max_iter, num_waypoints=num_waypoints)
    
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        start_time = time.time()
        
        if use_coordinator:
            # --- 开启 Coordinator Agent ---
            coord_agent = CoordinatorAgent()
            coord_agent.algo_params = algo_params.copy()
            current_algo_params = coord_agent.algo_params
            current_specific_params = {}
            global_best_score = float('inf')
            
            meta_rounds = 5

            for _ in range(meta_rounds): 
                for key, value in current_specific_params.items():
                    if hasattr(planner, key): setattr(planner, key, value)
                    
                best_path, _ = planner.optimize()
                score, details, env_info = evaluator.evaluate_particle(best_path)
                
                if score < global_best_score:
                    global_best_score = score
                    
                current_algo_params, new_eval_params, current_specific_params, is_finished, _ = \
                    coord_agent.analyze_and_act(global_best_score, details, env_info, algo_name)
                    
                evaluator.params.update(new_eval_params)
                if is_finished: break
                
            final_score = global_best_score
        else:
            # --- 不使用 Coordinator Agent (Baseline) ---
            best_path, _ = planner.optimize()
            final_score, details, _ = evaluator.evaluate_particle(best_path)

        exec_time = time.time() - start_time
        
        # 提取真实物理失败次数 (假设 evaluator 返回的是惩罚分数，我们要根据惩罚系数反推次数)
        # 你可以根据你 path_evaluator 中的系数调整这里的除数，通常一次致命碰撞是 1,000,000 分
        col_pen = details.get('fatal_collision', 0)
        miss_pen = details.get('missed_target', 0)
        
        collision_count = int(col_pen / 1000000) if col_pen >= 1000000 else (1 if col_pen > 0 else 0)
        missed_count = int(miss_pen / 500000) if miss_pen >= 500000 else (1 if miss_pen > 0 else 0)
        
        is_success = (collision_count == 0 and missed_count == 0)

    finally:
        # ========================================================
        # 【解除消音】：无论是否报错，必须恢复控制台的输出流
        # ========================================================
        sys.stdout.close()
        sys.stdout = original_stdout
    
    return final_score, exec_time, is_success, details, collision_count, missed_count, algo_params

def run_batch_tests():
    # Setup
    maps = {
        "Easy": "maps/easy_map.json5",
        "Medium": "maps/medium_map.json5",
        "Hard": "maps/hard_map.json5"
    }
    
    algorithms = ["PSO", "SSA"] # choose from ["PSO", "SSA", "GWO", "WOA", "GA"] 
    num_runs = 10 # Number of independent trials
    
    results_ablation = {}
    prior_knowledge_llm = {}
    
    # 【新增】用来计算总任务量
    total_tasks = len(maps) * len(algorithms) * 2
    current_task = 0
    
    print("\n" + "="*70)
    print(f"启动大批量 Benchmark (共 {total_tasks} 组任务, 每组 {num_runs} 次运行)")
    print("="*70 + "\n")
    
    for map_idx, (map_name, map_path) in enumerate(maps.items(), 1):
        if not os.path.exists(map_path):
            print(f"Skipping {map_name} map - file not found at {map_path}")
            continue
            
        print(f"\nLoading Map: {map_name}...")
        evaluator = PathEvaluator()
        evaluator.env = UAVEnvironment3D(map_path)
        
        results_ablation[map_name] = {}
        prior_knowledge_llm[map_name] = {}
        
        for algo_idx, algo in enumerate(algorithms, 1):
            print(f"\n  测试算法: {algo}")
            results_ablation[map_name][algo] = {}
            prior_knowledge_llm[map_name][algo] = {}
            
            for use_agent, mode_name in [(False, "Baseline"), (True, "With_Coordinator")]:
                current_task += 1
                
                # ========================================================
                # 【新增】：显式的全局横幅
                # ========================================================
                progress_pct = (current_task - 1) / total_tasks * 100
                print("-" * 65)
                print(f"🚀 [总体进度: {current_task}/{total_tasks} ({progress_pct:.1f}%)]")
                print(f"🗺️ 地图: {map_name} ({map_idx}/{len(maps)}) | ⚙️ 算法: {algo} ({algo_idx}/{len(algorithms)}) | 🛠️ 模式: {mode_name}")
                print("-" * 65)
                
                scores, times, all_details = [], [], []
                collisions_history, missed_history = [], []
                success_count = 0
                used_params = {}
                
                for _ in tqdm(range(num_runs), desc="   运行中", leave=False, file=sys.stdout, ncols=80):
                    score, ex_time, success, details, coll_c, miss_c, params = run_single_trial(evaluator, algo, use_agent)
                    
                    scores.append(score)
                    times.append(ex_time)
                    all_details.append(details)
                    collisions_history.append(coll_c)
                    missed_history.append(miss_c)
                    used_params = params
                    if success: success_count += 1
                
                # --- 数据聚合 ---
                mean_score = float(np.mean(scores))
                success_rate = (success_count / num_runs) * 100
                avg_time = float(np.mean(times))
                avg_collisions = float(np.mean(collisions_history))
                avg_missed = float(np.mean(missed_history))
                
                # 计算 Fitness Breakdown 平均值
                avg_details = {}
                for key in all_details[0].keys():
                    avg_details[key] = float(np.mean([d.get(key, 0) for d in all_details]))
                
                # 当这 30 遍跑完时，原来底部的进度条会消失，并在这里打印最终这组的成绩
                print(f"✅ 完成! 得分: {mean_score:,.0f} | 成功率: {success_rate:.1f}% | 均碰撞: {avg_collisions:.1f}次 | 均耗时: {avg_time:.1f}s\n")
                
                # ==========================================
                # 1. 保存给消融实验的完整数据包 (无删减)
                # ==========================================
                results_ablation[map_name][algo][mode_name] = {
                    "used_params": used_params,
                    "mean_score": mean_score,
                    "std_score": float(np.std(scores)),
                    "mean_time": avg_time,
                    "success_rate": success_rate,
                    "avg_collisions_count": avg_collisions,
                    "avg_missed_targets_count": avg_missed,
                    "raw_scores_30_runs": scores,
                    "average_fitness_breakdown": avg_details
                }
                
                # ==========================================
                # 2. 保存给 LLM 的先验知识 (移除时间，保留参数和原始得分)
                # (我们只把加上了智能体后的“完全体表现”发给 LLM，因为这是将来的实战状态)
                # ==========================================
                if mode_name == "With_Coordinator":
                    # 动态生成定性评估
                    qualitative = "表现中等。"
                    if success_rate >= 90 and avg_collisions < 0.2: qualitative = "成功率极高，避障能力完美，极度稳定。"
                    elif success_rate < 40 or avg_collisions > 1.0: qualitative = "极易撞墙或陷入局部死锁，不适合复杂地形。"
                    
                    prior_knowledge_llm[map_name][algo] = {
                        "algorithm_params_used": used_params,
                        "mean_score": mean_score,
                        "std_score": float(np.std(scores)),
                        "success_rate": success_rate,
                        "qualitative_evaluation": qualitative,
                        "raw_scores_sample": [round(s, 1) for s in scores]  # 把所有原始分数发给 LLM
                    }
                
    # --- 写入文件 ---
    os.makedirs("results", exist_ok=True)
    
    with open("full_ablation_results/PSO_SSA.json", "w", encoding='utf-8') as f:
        json.dump(results_ablation, f, indent=4, ensure_ascii=False)
    
    with open("prior_knowledge/PSO_SSA.json", "w", encoding='utf-8') as f:
        json.dump(prior_knowledge_llm, f, indent=4, ensure_ascii=False)
    
    print("="*70)
    print("🎉 Benchmark 全部完成！")
    print("✅ 消融实验全量数据已保存至 'results/full_ablation_results.json'")
    print("✅ LLM 先验知识库已保存至 'prior_knowledge.json'")
    print("="*70)

if __name__ == "__main__":
    run_batch_tests()

