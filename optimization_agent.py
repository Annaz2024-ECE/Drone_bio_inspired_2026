import numpy as np
import matplotlib.pyplot as plt

# 导入已经实现的所有底层规划器
from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from aco_planner import ACOPlanner
from woa_planner import WOAPlanner 

from path_evaluator import PathEvaluator

class OptimizationAgent:
    def __init__(self, evaluator, task_priority):
        """
        仿生优化智能体 (Algorithm Selector Agent)
        统帅五大算法，根据环境与任务需求，进行“点将”
        
        :param evaluator: 统一实例化的路径评价器
        :param task_priority: 任务的侧重点 ('speed', 'safety', 'smoothness', 'global_optimal', 'close_inspection')
        """
        self.evaluator = evaluator
        self.task_priority = task_priority

    def make_decision(self):
        """
        核心决策大脑：看菜吃饭，量体裁衣
        """
        print("=" * 60)
        print(f"🤖 [仿生优化智能体] 正在进行战前分析...")
        print(f"   -> 接收任务指令: 侧重【{self.task_priority.upper()}】")
        print("-" * 60)
        
        if self.task_priority == 'speed':
            print("   -> 场景感知: 紧急医疗救援，需要最快到达！")
            print("   -> 算法指派: 🏆 PSO (粒子群优化算法)")
            print("   -> 决策理由: PSO 收敛速度最快，易于拉直航线，适合冲刺飞越。")
            print("   -> 参数预设: 种群 90, 航点数 10 (减少航点拉直航线), 迭代 150 次。")
            return PSOPlanner(evaluator=self.evaluator, num_particles=90, num_waypoints=10, max_iter=150)
            
        elif self.task_priority == 'safety':
            print("   -> 场景感知: 台风灾后巡检，障碍物情况复杂，极易陷入死角。")
            print("   -> 算法指派: 🏆 SSA (麻雀搜索算法)")
            print("   -> 决策理由: SSA 的侦察者与加入者逃逸机制提供了极强的全局探索能力，防早熟。")
            print("   -> 参数预设: 种群 120, 航点数 13 (增加航点绕开死胡同), 迭代 200 次。")
            return SSAPlanner(evaluator=self.evaluator, num_sparrows=120, num_waypoints=13, max_iter=200)
            
        elif self.task_priority == 'smoothness':
            print("   -> 场景感知: 夜间校园静默巡逻，不能有急转弯导致电机啸叫。")
            print("   -> 算法指派: 🏆 GWO (灰狼优化算法)")
            print("   -> 决策理由: 灰狼的包围狩猎机制，使得路线演化非常平缓圆润。")
            print("   -> 参数预设: 种群 100, 航点数 20 (高密度航点保障极致丝滑), 迭代 200 次。")
            return GWOPlanner(evaluator=self.evaluator, num_wolves=100, num_waypoints=15, max_iter=200)
            
        elif self.task_priority == 'global_optimal':
            print("   -> 场景感知: 校园复杂地形的全面测绘，追求全局最优通路。")
            print("   -> 算法指派: 🏆 ACO (蚁群优化算法)")
            print("   -> 决策理由: 蚁群通过信息素挥发与累积的正反馈，能极大概率找到全局最短路。")
            print("   -> 参数预设: 蚂蚁数 50, 航点数 9, 迭代 300 次。")
            return ACOPlanner(evaluator=self.evaluator, num_ants=50, num_waypoints=9, max_iter=300)
            
        elif self.task_priority == 'close_inspection':
            print("   -> 场景感知: 建筑物贴面缺陷检测，需要紧贴墙面但不能撞墙。")
            print("   -> 算法指派: 🏆 WOA (鲸鱼优化算法)")
            print("   -> 决策理由: 鲸鱼的‘螺旋气泡网’机制极其适合在极小空间内做局部的精细微调。")
            print("   -> 参数预设: 鲸鱼数 100, 航点数 12, 迭代 250 次。")
            return WOAPlanner(evaluator=self.evaluator, pop_size=100, num_waypoints=12, max_iter=250)
            
        else:
            print("   -> 遇到未知任务标签，执行默认调度: PSO 算法")
            return PSOPlanner(evaluator=self.evaluator)

# ==========================================
# 完整系统运行流
# ==========================================
if __name__ == "__main__":
    # 1. 统一实例化地图和评价器
    shared_evaluator = PathEvaluator()
    
    # 【测试区】：修改这里的值，看看智能体会派谁出战！
    # 可选项: 'speed', 'safety', 'smoothness', 'global_optimal', 'close_inspection'
    current_task_mode = 'smoothness'  
    
    # 2. 仿生优化智能体进行战前分析与调度
    agent = OptimizationAgent(evaluator=shared_evaluator, task_priority=current_task_mode)
    selected_planner = agent.make_decision()
    
    print("\n" + "-" * 60)
    print("🚀 智能体指派完毕，底层仿生学算法开始执行...")
    print("-" * 60)
    
    # 3. 调用底层算法执行 (现在统一只返回 2 个参数！)
    best_path, score_history = selected_planner.optimize()
    
    print("\n✅ 规划完成，正在生成最终的分析报告图表...")
    
    # 动态获取当前被调用的算法名称 (比如 "PSOPlanner" -> "PSO")
    algo_name = type(selected_planner).__name__.replace('Planner', '')
    run_name = f"{current_task_mode.upper()}"
    
    # 4. 调用基类强大的通用画图函数
    selected_planner.plot_result(best_path, score_history, algo_name=algo_name, run_idx=run_name)
    
    print("\n[系统提示] 运行结束。你可以修改 current_task_mode 变量，观察智能体如何自动切换 5 大算法。")