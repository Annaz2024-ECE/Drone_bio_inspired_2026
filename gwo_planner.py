import numpy as np
import matplotlib.pyplot as plt
from environment_buildup import UAVEnvironment2D
from path_evaluator import PathEvaluator

class GWOPlanner:
    def __init__(self, evaluator, num_wolves=50, max_iter=300, num_waypoints=6):
        """
        初始化原始灰狼算法
        :param evaluator: PathEvaluator 对象，用于计算路径适应度
        :param num_wolves: 狼群数量 (种群大小)
        :param max_iter: 最大迭代次数
        :param num_waypoints: 无人机路径中的中间航点数量
        """
        self.evaluator = evaluator
        self.env = evaluator.env
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.num_waypoints = num_waypoints
        
        # 每一个中间航点包含 X 和 Y 坐标，因此搜索空间的维度是 num_waypoints * 2
        self.dim = self.num_waypoints * 2
        self.lb = min(self.env.x_bounds[0], self.env.y_bounds[0])  # 地图坐标下界
        self.ub = max(self.env.x_bounds[1], self.env.y_bounds[1])  # 地图坐标上界
        
        # 初始化 Alpha (最优), Beta (次优), Delta (第三优) 狼的位置和得分
        # 因为我们要求最小值(适应度越低越好)，所以初始得分设为正无穷
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float("inf")
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float("inf")

        # 【新增：历史最佳保险箱】
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = float("inf")
        
        # 【新增：卡死监控器】
        self.stagnation_count = 0
        self.last_alpha_score = float("inf")
        
        # 随机初始化狼群位置矩阵: 维度为 (num_wolves, dim)
        # self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
        # 1. 先让所有狼随机瞎跑
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
        
        # 2. 注入先验知识：强制让前 30% 的精英狼精准踩在打卡点上！
        # top_30_percent = int(self.num_wolves * 0.3)
        # for i in range(top_30_percent):
        #     # 此时总共有 12 个中间航点 (索引 0 到 11)
        #     # 一维数组排列方式为: [x0, y0, x1, y1, x2, y2 ...]
            
        #     # 把第 3 个航点（大概在路线的 1/4 处）死死钉在【西区 West Area】
        #     # 索引 4 是 x, 索引 5 是 y
        #     self.positions[i, 4] = 22.0 
        #     self.positions[i, 5] = 40.0 
            
        #     # 把第 9 个航点（大概在路线的 3/4 处）死死钉在【东区 East Area】
        #     # 索引 16 是 x, 索引 17 是 y
        #     self.positions[i, 16] = 78.0 
        #     self.positions[i, 17] = 70.0

        top_30_percent = int(self.num_wolves * 0.3)
        for i in range(top_30_percent):
            # 现在总共只有 5 个中间航点，索引分别是 0, 1, 2, 3, 4
            # 它们对应的坐标维度索引是 0~9
            
            # 把第 2 个航点（索引 1）钉在【西区 West Area】
            # X坐标是 2, Y坐标是 3
            self.positions[i, 2] = 22.0 
            self.positions[i, 3] = 40.0 
            
            # 把第 4 个航点（索引 3）钉在【东区 East Area】
            # X坐标是 6, Y坐标是 7
            self.positions[i, 6] = 78.0 
            self.positions[i, 7] = 70.0
        
        # 记录每代最佳适应度以便画图
        self.convergence_curve = []

    def _wolf_to_path(self, wolf_pos):
        """
        将一维的狼位置向量还原为无人机的完整二维路径坐标
        返回格式: [Start -> Waypoint1 -> ... -> WaypointN -> End]
        """
        waypoints = wolf_pos.reshape((self.num_waypoints, 2))
        full_path = np.vstack([self.env.start_point, waypoints, self.env.end_point])
        return full_path

    def optimize(self):
        """ 运行灰狼优化主循环 """
        
        # 【核心修改 1：开局前，买好“保险箱”和“监控器”】
        # 为了防止在 __init__ 里漏写，我们直接在优化开始前初始化它们
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = float("inf")
        self.stagnation_count = 0
        self.last_alpha_score = float("inf")

        for l in range(self.max_iter):
            for i in range(self.num_wolves):
                # 1. 边界限制
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                # 2. 计算当前狼的适应度
                current_path = self._wolf_to_path(self.positions[i])
                fitness = self.evaluator.evaluate_pso_particle(current_path)
                
                # 3. 更新 Alpha, Beta, Delta 狼
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                    
                    # 【核心修改 2：只要 Alpha 刷新了，且比保险箱里的好，就立刻复印一份存进保险箱】
                    if self.alpha_score < self.historical_best_score:
                        self.historical_best_score = self.alpha_score
                        self.historical_best_pos = self.alpha_pos.copy()
                        
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                    
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            # 4. 更新收敛控制参数 a 
            a = 2.0 - l * (2.0 / self.max_iter)
            
            # 5. 根据领导者们的位置更新整个狼群的位置
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    # --- Alpha 狼的影响 ---
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # --- Beta 狼的影响 ---
                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # --- Delta 狼的影响 ---
                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    self.positions[i, j] = (X1 + X2 + X3) / 3.0
            
            # 【核心修改 3：卡死监控与大地震机制】
            # 检查这代的分数和上代比有没有明显下降
            if abs(self.last_alpha_score - self.alpha_score) < 1.0:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0  # 分数降了，解除警报
                self.last_alpha_score = self.alpha_score

            # 如果连续 30 代分数都卡住不动，触发大地震！
            if self.stagnation_count > 30:
                print(f"⚠️ 迭代 {l+1}/{self.max_iter}: 卡死在 {self.alpha_score:,.2f} 分！触发大地震，全群重置寻路...")
                
                # 罢免现任头狼（将分数设为无限大）
                self.alpha_score = float("inf")
                self.beta_score = float("inf")
                self.delta_score = float("inf")
                
                # 全体狼群重新随机分布，彻底重新找路
                self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
                self.stagnation_count = 0 # 计数器清零
            
            # 【核心修改 4：画图曲线永远记录“保险箱”里的历史最低分，这样曲线不会因为大地震而反弹】
            self.convergence_curve.append(self.historical_best_score)
            
            # 打印训练进度
            if (l + 1) % 20 == 0 or l == 0:
                print(f"迭代次数 {l+1}/{self.max_iter}, 历史最佳得分: {self.historical_best_score:,.2f}")
                
        # 【核心修改 5：全部跑完后，直接把保险箱里的路线交接给无人机！】
        best_full_path = self._wolf_to_path(self.historical_best_pos)
        return best_full_path, self.historical_best_score, self.convergence_curve

    def plot_result(self, best_path, convergence_curve):
        """ 绘制结果：左图为地图路径，右图为收敛曲线 """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 绘制地图和路径
        self.env.draw_environment(ax=ax1)
        ax1.plot(best_path[:, 0], best_path[:, 1], color='#ff5722', linewidth=2.5, marker='o', label='GWO Best Path')
        # 显式指定图例位置，避免挡住路径
        ax1.legend(loc='upper right', fontsize=10) 
        
        # 2. 绘制收敛曲线
        ax2.plot(convergence_curve, color='#1976d2', linewidth=2)
        ax2.set_title('GWO Convergence Curve', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness Score (Log Scale)')
        
        # 【优化】使用对数坐标轴，完美展现前期跳出惩罚、后期精细优化的全过程
        ax2.set_yscale('log') 
        
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()


# ===================== 测试与可视化运行模块 =====================
if __name__ == "__main__":
    # 实例化您的环境与评估器
    evaluator = PathEvaluator()
    
    # 启动 GWO 规划
    # 参数建议：因为包含了两个检查区域并且要绕开大量建筑，设置6~8个中间航点较为理想
    print("GWO路径规划...")
    # gwo_planner.py 最底部
    gwo_planner = GWOPlanner(evaluator, num_wolves=60, max_iter=300, num_waypoints=8)
    
    best_path, best_score, convergence_curve = gwo_planner.optimize()
    print(f"规划完成！最终路径得分: {best_score:.2f}")

    # 画图前，把原点变成丝滑曲线
    smooth_best_path = evaluator.generate_chaikin_path(best_path, iterations=4)

    # 把丝滑的路线丢进去画图
    gwo_planner.plot_result(smooth_best_path, convergence_curve)
        
    