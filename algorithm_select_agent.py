import numpy as np
import matplotlib.pyplot as plt

# 导入所有底层规划器
from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from dsaco_planner import DSACOPlanner
from woa_planner import WOAPlanner

from path_evaluator import PathEvaluator

class Algorithm_Select_Agent:
    def __init__(self, evaluator, task_priority='speed'):
        """
        仿生优化智能体 (Algorithm Selector Agent)
        核心能力：【情景目标驱动】 + 【环境感知调参】
        
        :param evaluator: 统一实例化的路径评价器
        :param task_priority: 情景目标 ('speed', 'safety', 'smoothness', 'global_optimal', 'close_inspection')
        """
        self.evaluator = evaluator
        self.env = evaluator.env
        self.task_priority = task_priority

    def analyze_environment(self):
        """
        前期环境考量：从底层地图中提取特征信息，进行雷达扫描分析
        """
        num_obstacles = len(self.env.obstacles)
        num_targets = len(self.env.target_areas)
        
        print("=" * 65)
        print("[仿生优化智能体] 正在对当前地图进行扫描与特征提取...")
        print(f"   -> 发现障碍物数量: {num_obstacles} 个")
        print(f"   -> 发现巡检目标区: {num_targets} 个")
        
        # 障碍物密度/复杂度评级
        if num_obstacles >= 20:
            complexity = "High"
            print("   -> 环境危险评估: 【极高】！地图障碍物非常密集，极易发生碰撞或陷入死胡同。")
        elif num_obstacles >= 10:
            complexity = "Medium"
            print("   -> 环境危险评估: 【中等】。存在一定数量的建筑群。")
        else:
            complexity = "Low"
            print("   -> 环境危险评估: 【较低】。环境较为开阔。")
            
        return complexity, num_targets, num_obstacles

    def _get_target_algorithm(self):
        """
        【模块2】算法分配: 仅根据任务情景，决定派遣哪个算法兵种
        返回: 算法简称, 算法类, 场景描述
        """
        if self.task_priority == 'speed':
            return "PSO", PSOPlanner, "紧急医疗救援，需要最快收敛拉出直线"
        elif self.task_priority == 'safety':
            return "SSA", SSAPlanner, "台风灾后巡检，需要强全局探索能力避开死角"
        elif self.task_priority == 'smoothness':
            return "GWO", GWOPlanner, "夜间校园静默巡逻，不能有急转弯导致电机啸叫"
        elif self.task_priority == 'global_optimal':
            return "DSACO", DSACOPlanner, "校园复杂地形测绘，追求全局最短路"
        elif self.task_priority == 'close_inspection':
            return "WOA", WOAPlanner, "建筑物贴面缺陷检测，需近距离精细微调"
        else:
            return "PSO", PSOPlanner, "未知任务标签，执行默认兜底调度"

    def _fine_tune_parameters(self, algo_name, complexity, num_targets, num_obstacles):
        """
        【模块3】参数微调专家: 独立封装的参数计算函数
        按顺序：先设通用保底 -> 根据算法微调 -> 根据地图难度加算力
        """
        # 1. General 初始通用保底算力
        base_pop = 50
        base_iter = 100
        base_waypoints = num_targets + 3 

        # 2. 根据最终所选的算法，做第一轮性格微调
        if algo_name == "PSO":
            base_pop += 30; base_iter -= 30; base_waypoints -= 1 # 要快
        elif algo_name == "SSA":
            base_iter += 80; base_waypoints += 2 # 要稳
        elif algo_name == "GWO":
            base_pop += 20; base_iter += 50; base_waypoints += 4 # 要丝滑
        elif algo_name == "DSACO":
            base_pop = max(40, base_pop - 10); base_iter += 150 # 要深搜
        elif algo_name == "WOA":
            base_pop += 50; base_iter += 50; base_waypoints += 2 # 要包围

        # 3. 核心：根据地图难度分类 (High/Medium) 进行算力二次扩容
        if complexity == "High":
            base_pop += int(num_obstacles * 1.5 + num_targets * 2)
            base_iter += int(num_obstacles * 3 + num_targets * 5)
            base_waypoints += 2
        elif complexity == "Medium":
            base_pop += int(num_obstacles * 1.0)
            base_iter += int(num_obstacles * 1.5)
            base_waypoints += 1

        # 4. 兜底校验（保证不越过系统最低安全线）
        final_pop = max(30, int(base_pop))
        final_iter = max(50, int(base_iter))
        final_waypoints = max(num_targets + 1, int(base_waypoints))

        return final_pop, final_iter, final_waypoints


    def make_decision(self):
        """
        主控枢纽：依次调用分析、选型、微调，最后干净利落地返回算法实例
        """
        # 1. 获取地图难度
        complexity, num_targets, num_obstacles = self.analyze_environment()
        
        # 2. 获取目标算法
        algo_name, PlannerClass, scene_desc = self._get_target_algorithm()
        
        print("-" * 65)
        print(f"[仿生优化智能体] 场景感知: {scene_desc}")
        print(f"   -> 首选算法: {algo_name}")
        
        # 3. 传入独立函数，计算最终算力
        final_pop, final_iter, final_waypoints = self._fine_tune_parameters(
            algo_name, complexity, num_targets, num_obstacles
        )
        
        print(f"   -> 参数自适应微调: 结合环境难度[{complexity}]与算法特性，最终敲定:")
        print(f"      \033[96m种群={final_pop}, 迭代={final_iter}, 控制点={final_waypoints}\033[0m")
        
        # 4. 统一打包实例化 (解决冗余代码)
        kwargs = {
            'evaluator': self.evaluator,
            'num_waypoints': final_waypoints,
            'max_iter': final_iter
        }
        
        # 适配底层算法形参命名差异
        if algo_name in ["ACO", "DSACO"]: kwargs['num_ants'] = final_pop
        elif algo_name == "PSO": kwargs['num_particles'] = final_pop
        elif algo_name == "GWO": kwargs['num_wolves'] = final_pop
        elif algo_name == "SSA": kwargs['num_sparrows'] = final_pop
        elif algo_name == "WOA": kwargs['pop_size'] = final_pop

        return PlannerClass(**kwargs)


# ==========================================
# 完整系统运行流 
# ==========================================
if __name__ == "__main__":
    shared_evaluator = PathEvaluator()
    
    # 模拟任务指令
    current_task = 'safety'
    
    agent = Algorithm_Select_Agent(evaluator=shared_evaluator, task_priority=current_task)
    active_planner = agent.make_decision()
    
    print("\n" + "=" * 65)
    print(f"智能体指派完毕，[{type(active_planner).__name__}] 开始执行底层寻优...")
    print("=" * 65)
    
    best_path, score_history = active_planner.optimize()
    
    print("\n规划完成，正在生成最终的分析报告图表...")
    algo_name = type(active_planner).__name__.replace('Planner', '')
    active_planner.plot_result(best_path, score_history, algo_name=algo_name, run_idx=f"Scenario_{current_task.upper()}")