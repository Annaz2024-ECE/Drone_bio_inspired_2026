import numpy as np
import json
import time
import os
import sys
from tqdm import tqdm
import concurrent.futures
from functools import partial

from environment_buildup_3D import UAVEnvironment3D
from path_evaluator import PathEvaluator
from coordinator_agent import CoordinatorAgent
from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from woa_planner import WOAPlanner
from ga_planner import GAPlanner


def run_single_trial_wrapper(evaluator, algo, use_agent, seed=None):
    """包装函数，支持设置随机种子保证可复现"""
    if seed is not None:
        np.random.seed(seed)
    return run_single_trial(evaluator, algo, use_agent)


def run_single_trial(evaluator, algo_name, use_coordinator):
    """
    Runs a single optimization trial.
    """
    num_targets = len(evaluator.env.target_areas)
    num_waypoints = int(num_targets * 1.5)
    if num_targets < 8:
        max_iter = 50
    elif num_targets < 11:
        max_iter = 100
    else:
        max_iter = 150
    pop_size = 80

    algo_params = {'pop_size': pop_size, 'max_iter': max_iter, 'num_waypoints': num_waypoints}

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

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        start_time = time.time()

        if use_coordinator:
            coord_agent = CoordinatorAgent()
            coord_agent.algo_params = algo_params.copy()
            current_algo_params = coord_agent.algo_params
            current_specific_params = {}
            global_best_score = float('inf')
            meta_rounds = 5

            for _ in range(meta_rounds):
                for key, value in current_specific_params.items():
                    if hasattr(planner, key):
                        setattr(planner, key, value)

                best_path, _ = planner.optimize()
                score, details, env_info = evaluator.evaluate_particle(best_path)

                if score < global_best_score:
                    global_best_score = score

                current_algo_params, new_eval_params, current_specific_params, is_finished, _ = \
                    coord_agent.analyze_and_act(global_best_score, details, env_info, algo_name)

                evaluator.params.update(new_eval_params)
                if is_finished:
                    break

            final_score = global_best_score
        else:
            best_path, _ = planner.optimize()
            final_score, details, _ = evaluator.evaluate_particle(best_path)

        exec_time = time.time() - start_time

        col_pen = details.get('fatal_collision', 0)
        miss_pen = details.get('missed_target', 0)

        collision_count = int(col_pen / 1000000) if col_pen >= 1000000 else (1 if col_pen > 0 else 0)
        missed_count = int(miss_pen / 500000) if miss_pen >= 500000 else (1 if miss_pen > 0 else 0)

        is_success = (collision_count == 0 and missed_count == 0)

    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    return final_score, exec_time, is_success, details, collision_count, missed_count, algo_params


def save_results(results_ablation, prior_knowledge_llm, suffix=""):
    """
    增量保存结果到 JSON 文件
    """
    os.makedirs("full_ablation_results", exist_ok=True)
    os.makedirs("prior_knowledge", exist_ok=True)
    
    # 保存消融实验结果
    ablation_file = f"full_ablation_results/all_algorithms{suffix}.json"
    with open(ablation_file, "w", encoding='utf-8') as f:
        json.dump(results_ablation, f, indent=4, ensure_ascii=False)
    
    # 保存先验知识
    prior_file = f"prior_knowledge/all_algorithms{suffix}.json"
    with open(prior_file, "w", encoding='utf-8') as f:
        json.dump(prior_knowledge_llm, f, indent=4, ensure_ascii=False)


def run_batch_tests():
    maps = {
        "Easy": "maps/easy_map.json5",
        "Medium": "maps/medium_map.json5",
        "Hard": "maps/hard_map.json5"
    }

    algorithms = ["PSO", "SSA", "GWO", "WOA", "GA"]
    num_runs = 15

    results_ablation = {}
    prior_knowledge_llm = {}

    total_tasks = len(maps) * len(algorithms) * 2
    current_task = 0

    # ============================================================
    # 【新增】：检查是否有之前的保存文件，支持断点续传
    # ============================================================
    checkpoint_file = "full_ablation_results/checkpoint.json"
    start_from_scratch = True
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_tasks = checkpoint_data.get("completed_tasks", [])
                print(f"\n📂 发现之前的检查点文件，已完成 {len(completed_tasks)} 组任务")
                print(f"   已完成: {completed_tasks}")
                
                # 询问是否继续
                response = input("   是否从检查点继续？(y/n): ").strip().lower()
                if response == 'y':
                    start_from_scratch = False
                    # 加载已有的结果
                    if os.path.exists("full_ablation_results/all_algorithms.json"):
                        with open("full_ablation_results/all_algorithms.json", "r", encoding='utf-8') as f:
                            results_ablation = json.load(f)
                    if os.path.exists("prior_knowledge/all_algorithms.json"):
                        with open("prior_knowledge/all_algorithms.json", "r", encoding='utf-8') as f:
                            prior_knowledge_llm = json.load(f)
                    
                    # 标记已完成的任务
                    completed_set = set(completed_tasks)
                else:
                    print("   从头开始运行...")
        except Exception as e:
            print(f"   ⚠️ 读取检查点失败: {e}，从头开始...")

    print("\n" + "=" * 70)
    print(f"启动大批量 Benchmark (共 {total_tasks} 组任务, 每组 {num_runs} 次运行)")
    print("=" * 70 + "\n")

    # 用于记录已完成任务的列表（用于检查点）
    completed_tasks = []

    for map_idx, (map_name, map_path) in enumerate(maps.items(), 1):
        if not os.path.exists(map_path):
            print(f"Skipping {map_name} map - file not found at {map_path}")
            continue

        print(f"\nLoading Map: {map_name}...")
        evaluator = PathEvaluator()
        evaluator.env = UAVEnvironment3D(map_path)

        # 如果从检查点恢复，需要确保数据结构存在
        if map_name not in results_ablation:
            results_ablation[map_name] = {}
        if map_name not in prior_knowledge_llm:
            prior_knowledge_llm[map_name] = {}

        for algo_idx, algo in enumerate(algorithms, 1):
            print(f"\n  测试算法: {algo}")
            
            if algo not in results_ablation[map_name]:
                results_ablation[map_name][algo] = {}
            if algo not in prior_knowledge_llm[map_name]:
                prior_knowledge_llm[map_name][algo] = {}

            for use_agent, mode_name in [(False, "Baseline"), (True, "With_Coordinator")]:
                current_task += 1
                
                # ============================================================
                # 【新增】：检查这个任务是否已经完成
                # ============================================================
                task_key = f"{map_name}_{algo}_{mode_name}"
                
                if not start_from_scratch and task_key in completed_set:
                    print(f"\n   ⏭️ 跳过已完成任务: {task_key}")
                    continue

                # ============================================================
                # 使用并行运行 num_runs 次试验
                # ============================================================
                num_workers = min(num_runs, 15)  # 改为 8 更合适你的 CPU
                
                trial_func = partial(run_single_trial_wrapper, evaluator, algo, use_agent)
                seeds = list(range(num_runs))

                # 显示当前任务信息
                progress_pct = (current_task - 1) / total_tasks * 100
                print("-" * 65)
                print(f"🚀 [总体进度: {current_task}/{total_tasks} ({progress_pct:.1f}%)]")
                print(f"🗺️ 地图: {map_name} ({map_idx}/{len(maps)}) | ⚙️ 算法: {algo} ({algo_idx}/{len(algorithms)}) | 🛠️ 模式: {mode_name}")
                print(f"   并行启动 {num_runs} 次试验 (使用 {num_workers} 个工作线程)...")
                print("-" * 65)

                # 存储结果
                scores = []
                times = []
                all_details = []
                collisions_history = []
                missed_history = []
                success_count = 0
                used_params = {}

                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    future_to_seed = {
                        executor.submit(trial_func, seed=seed): seed
                        for seed in seeds
                    }

                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_seed):
                        seed = future_to_seed[future]
                        try:
                            score, ex_time, success, details, coll_c, miss_c, params = future.result(timeout=300)
                            
                            scores.append(score)
                            times.append(ex_time)
                            all_details.append(details)
                            collisions_history.append(coll_c)
                            missed_history.append(miss_c)
                            used_params = params
                            if success:
                                success_count += 1
                            
                            completed += 1
                            if completed % max(1, num_runs // 10) == 0 or completed == num_runs:
                                print(f"   [进度] {completed}/{num_runs} 完成", end="\r")
                                
                        except Exception as e:
                            print(f"\n   ⚠️ 试验 (seed={seed}) 失败: {e}")
                            completed += 1
                    
                    print()

                # --- 数据聚合 ---
                if len(scores) == 0:
                    print(f"   ⚠️ 所有试验都失败了，跳过此组")
                    continue
                    
                mean_score = float(np.mean(scores))
                success_rate = (success_count / num_runs) * 100
                avg_time = float(np.mean(times))
                avg_collisions = float(np.mean(collisions_history))
                avg_missed = float(np.mean(missed_history))

                avg_details = {}
                if all_details:
                    for key in all_details[0].keys():
                        avg_details[key] = float(np.mean([d.get(key, 0) for d in all_details]))

                print(f"✅ 完成! 得分: {mean_score:,.0f} | 成功率: {success_rate:.1f}% | 均碰撞: {avg_collisions:.1f}次 | 均漏检: {avg_missed:.1f}次 | 均耗时: {avg_time:.1f}s\n")

                # --- 保存结果到内存 ---
                results_ablation[map_name][algo][mode_name] = {
                    "used_params": used_params,
                    "mean_score": mean_score,
                    "std_score": float(np.std(scores)) if scores else 0,
                    "mean_time": avg_time,
                    "success_rate": success_rate,
                    "avg_collisions_count": avg_collisions,
                    "avg_missed_targets_count": avg_missed,
                    "raw_scores": scores,
                    "average_fitness_breakdown": avg_details
                }

                if mode_name == "With_Coordinator":
                    qualitative = "表现中等。"
                    if success_rate >= 90 and avg_collisions < 0.2:
                        qualitative = "成功率极高，避障能力完美，极度稳定。"
                    elif success_rate < 40 or avg_collisions > 1.0:
                        qualitative = "极易撞墙或陷入局部死锁，不适合复杂地形。"

                    prior_knowledge_llm[map_name][algo] = {
                        "algorithm_params_used": used_params,
                        "mean_score": mean_score,
                        "std_score": float(np.std(scores)) if scores else 0,
                        "success_rate": success_rate,
                        "qualitative_evaluation": qualitative,
                        "raw_scores_sample": [round(s, 1) for s in scores] if scores else []
                    }

                # ============================================================
                # 【关键修改】：每完成一组就立即保存文件
                # ============================================================
                # 记录已完成的任务
                completed_tasks.append(task_key)
                
                # 保存检查点（记录已完成的任务列表）
                checkpoint_data = {
                    "completed_tasks": completed_tasks,
                    "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_tasks": total_tasks
                }
                os.makedirs("full_ablation_results", exist_ok=True)
                with open("full_ablation_results/checkpoint.json", "w", encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)
                
                # 保存完整结果
                save_results(results_ablation, prior_knowledge_llm)
                print(f"   💾 数据已保存 (已完成 {len(completed_tasks)}/{total_tasks} 组)")

    # --- 最终保存 ---
    save_results(results_ablation, prior_knowledge_llm)
    
    # 删除检查点文件（所有任务已完成）
    if os.path.exists("full_ablation_results/checkpoint.json"):
        os.remove("full_ablation_results/checkpoint.json")

    print("=" * 70)
    print("🎉 Benchmark 全部完成！")
    print("✅ 消融实验全量数据已保存至 'full_ablation_results/all_algorithms.json'")
    print("✅ LLM 先验知识库已保存至 'prior_knowledge/all_algorithms.json'")
    print("=" * 70)


if __name__ == "__main__":
    run_batch_tests()