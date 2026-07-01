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
            
        return complexity, num_targets

    def make_decision(self):
        """
        核心二维决策树：情景(决定算法) + 环境(决定参数)
        """
        complexity, num_targets = self.analyze_environment()
        
        print("-" * 65)
        print(f"[仿生优化智能体] 结合任务情景【{self.task_priority.upper()}】与环境特征【{complexity}】，开始调度...")
        
        # ==========================================
        # 1. 极速抢救情景 (Speed) -> 偏好 PSO
        # ==========================================
        if self.task_priority == 'speed':
            print("   -> 场景感知: 紧急医疗救援，需要最快收敛拉出直线。")
            print("   -> 首选算法: PSO (粒子群优化算法)")
            if complexity == "High":
                print("   -> 环境自适应调参: 障碍极度密集，传统 PSO 易撞墙。增加种群至 120，航点设为 10（保证灵活度又不至于拖慢速度）。")
                return PSOPlanner(evaluator=self.evaluator, num_particles=120, num_waypoints=10, max_iter=200)
            else:
                print("   -> 环境自适应调参: 环境较为空旷。种群 80，航点 8，追求极致收敛速度。")
                return PSOPlanner(evaluator=self.evaluator, num_particles=80, num_waypoints=8, max_iter=150)

        # ==========================================
        # 2. 台风灾后安全情景 (Safety) -> 偏好 SSA
        # ==========================================
        elif self.task_priority == 'safety':
            print("   -> 场景感知: 台风灾后巡检，需要强全局探索能力避开死角。")
            print("   -> 首选算法: SSA (麻雀搜索算法)")
            if complexity == "High":
                print("   -> 决策执行: 航点 12 个，种群扩大至 150 以增强算力。")
                return SSAPlanner(evaluator=self.evaluator, num_sparrows=150, num_waypoints=12, max_iter=250)
            else:
                print("   -> 环境自适应调参: 障碍较少。种群 100，航点 10 即可安全覆盖。")
                return SSAPlanner(evaluator=self.evaluator, num_sparrows=100, num_waypoints=10, max_iter=200)

        # ==========================================
        # 3. 夜间静默情景 (Smoothness) -> 偏好 GWO
        # ==========================================
        elif self.task_priority == 'smoothness':
            print("   -> 场景感知: 夜间校园静默巡逻，不能有急转弯导致电机啸叫。")
            print("   -> 首选算法: GWO (灰狼优化算法)")
            if complexity == "High":
                print("   -> 环境自适应调参: 障碍密集，必须增加航点以保证绕弯圆润。种群 120，航点 14 个。")
                return GWOPlanner(evaluator=self.evaluator, num_wolves=120, num_waypoints=14, max_iter=250)
            else:
                print("   -> 环境自适应调参: 常规平滑模式。种群 80，航点 10 个。")
                return GWOPlanner(evaluator=self.evaluator, num_wolves=80, num_waypoints=10, max_iter=200)

        # ==========================================
        # 4. 全局最优测绘情景 (Global Optimal) -> 偏好 DSACO
        # ==========================================
        elif self.task_priority == 'global_optimal':
            print("   -> 场景感知: 校园复杂地形测绘，追求全局最短路。")
            print("   -> 首选算法: DSACO (蚁群优化算法)")
            if complexity == "High":
                print("   -> 环境自适应调参: 障碍极多，网格需要划分更细。蚂蚁数 60，分段数 11。")
                return DSACOPlanner(evaluator=self.evaluator, num_ants=60, num_waypoints=11, max_iter=250)
            else:
                print("   -> 环境自适应调参: 常规网格划分。蚂蚁数 40，分段数 8。")
                return DSACOPlanner(evaluator=self.evaluator, num_ants=40, num_waypoints=8, max_iter=200)

        # ==========================================
        # 5. 贴面精细检测情景 (Close Inspection) -> 偏好 WOA
        # ==========================================
        elif self.task_priority == 'close_inspection':
            print("   -> 场景感知: 建筑物贴面缺陷检测，需近距离精细微调。")
            print("   -> 首选算法: WOA (鲸鱼优化算法)")
            if complexity == "High":
                print("   -> 环境自适应调参: 缝隙狭窄，提升种群规模。种群 100，航点 12。")
                return WOAPlanner(evaluator=self.evaluator, pop_size=100, num_waypoints=12, max_iter=250)
            else:
                return WOAPlanner(evaluator=self.evaluator, pop_size=60, num_waypoints=8, max_iter=200)
                
        else:
            print("   -> 遇到未知任务标签，执行默认兜底调度: PSO 算法")
            return PSOPlanner(evaluator=self.evaluator)

# ==========================================
# 完整系统运行流 
# ==========================================
if __name__ == "__main__":
    # 1. 实例化统一地图与评价器
    shared_evaluator = PathEvaluator()
    
    # 2. 模拟任务指令 (你可以修改为 'speed', 'safety', 'smoothness' 等测试)
    current_task = 'safety'
    
    # 3. 实例化仿生优化智能体，传入评价器和当前情景
    agent = Algorithm_Select_Agent(evaluator=shared_evaluator, task_priority=current_task)
    
    # 4. 智能体提取环境信息，并分配配置好的算法实例
    active_planner = agent.make_decision()
    
    print("\n" + "=" * 65)
    print(f"智能体指派完毕，[{type(active_planner).__name__}] 开始执行底层寻优...")
    print("=" * 65)
    
    # 5. 算法执行 (统一接口，返回路径和得分历史)
    best_path, score_history = active_planner.optimize()
    
    print("\n规划完成，正在生成最终的分析报告图表...")
    algo_name = type(active_planner).__name__.replace('Planner', '')
    active_planner.plot_result(best_path, score_history, algo_name=algo_name, run_idx=f"Scenario_{current_task.upper()}")