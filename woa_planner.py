import numpy as np
import math
import random
from base_planner import BasePlanner

class WOAPlanner(BasePlanner):
    def __init__(self, evaluator=None, pop_size=60, max_iter=200, num_waypoints=10):
        """ 继承自 BasePlanner 的 WOA 算法 """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)

class WOAPlanner:
    def __init__(self, num_waypoints=6, pop_size=50, max_iter=150):
        """
        初始化 WOA 路径规划器
        :param num_waypoints: 起点和终点之间的中间控制点数量
        :param pop_size: 鲸鱼种群规模
        :param max_iter: 最大迭代次数
        """
        self.evaluator = PathEvaluator()
        self.env = self.evaluator.env  # 统一使用 evaluator 中的环境实例
        
        self.num_waypoints = num_waypoints
        self.pop_size = pop_size

        # __init__ 中添加
        self.safe_margin = 3.0   # 或更大，如 5.0
        self.effective_lb = self.lb + self.safe_margin
        self.effective_ub = self.ub - self.safe_margin

    def _decode_path(self, position):
        # 重写基类方法
        waypoints = position.reshape((self.num_waypoints, 2))
        # 按照 Y 轴坐标从小到大排序控制点
        waypoints = waypoints[np.argsort(waypoints[:, 1])] 
        path = [self.env.start_point] + waypoints.tolist() + [self.env.end_point]
        return np.array(path)
    def _initialize_population(self):
        positions = np.zeros((self.pop_size, self.dim))
        
        west_center = self.env.target_areas[0]['center']
        east_center = self.env.target_areas[1]['center']
        tier_size = self.pop_size // 3
        
        # 定义安全边距，防止控制点掉入地图边缘的“陷阱区”
        safe_margin = 2.0   # 可以根据实际地图调整
        effective_lb = self.lb + safe_margin
        effective_ub = self.ub - safe_margin
        
        max_retries = 20    # 每个个体最多重试 20 次
        
        for i in range(self.pop_size):
            # 生成基础插值点（与原代码完全相同）
            if i < tier_size:
                x_vals = np.linspace(self.env.start_point[0], self.env.end_point[0], self.num_waypoints + 2)[1:-1].tolist()
                y_vals = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_waypoints + 2)[1:-1].tolist()
            elif i < 2 * tier_size:
                mid_idx = self.num_waypoints // 2
                x_vals1 = np.linspace(self.env.start_point[0], west_center[0], mid_idx + 2)[1:].tolist()
                y_vals1 = np.linspace(self.env.start_point[1], west_center[1], mid_idx + 2)[1:].tolist()
                rem_points = self.num_waypoints - len(x_vals1)
                x_vals2 = np.linspace(west_center[0], self.env.end_point[0], rem_points + 2)[1:-1].tolist()
                y_vals2 = np.linspace(west_center[1], self.env.end_point[1], rem_points + 2)[1:-1].tolist()
                x_vals = x_vals1 + x_vals2
                y_vals = y_vals1 + y_vals2
            else:
                mid_idx = self.num_waypoints // 2
                x_vals1 = np.linspace(self.env.start_point[0], east_center[0], mid_idx + 2)[1:].tolist()
                y_vals1 = np.linspace(self.env.start_point[1], east_center[1], mid_idx + 2)[1:].tolist()
                rem_points = self.num_waypoints - len(x_vals1)
                x_vals2 = np.linspace(east_center[0], self.env.end_point[0], rem_points + 2)[1:-1].tolist()
                y_vals2 = np.linspace(east_center[1], self.env.end_point[1], rem_points + 2)[1:-1].tolist()
                x_vals = x_vals1 + x_vals2
                y_vals = y_vals1 + y_vals2
            
            base = np.column_stack((x_vals, y_vals)).flatten()
            
            # 带安全边界检查的噪声添加
            valid = False
            for retry in range(max_retries):
                # 生成噪声（可使用更保守的幅度，例如 ±5 而不是 ±20）
                if i < tier_size:
                    noise = np.random.uniform(-10, 10, self.dim)   # 降低扰动幅度
                else:
                    noise = np.random.uniform(-5, 5, self.dim)
                
                candidate = base + noise
                # 先裁剪到硬边界内
                candidate = np.clip(candidate, self.lb, self.ub)
                
                # 检查是否所有坐标都在安全区域内
                if np.all(candidate >= effective_lb) and np.all(candidate <= effective_ub):
                    valid = True
                    break
            
            # 如果重试耗尽仍未满足，则强制将所有违规坐标设为对应的安全边界值
            if not valid:
                candidate = np.clip(candidate, effective_lb, effective_ub)
            
            # === 新增：初始化时也进行 Y 轴排序同步 ===
            candidate_pts = candidate.reshape((self.num_waypoints, 2))
            candidate_pts = candidate_pts[np.argsort(candidate_pts[:, 1])]
            
            positions[i, :] = candidate_pts.flatten() # 存入有序的基因
        
        return positions
      
    
    def optimize(self):
        """
        执行鲸鱼优化算法的主循环
        """
        # 1. 初始化种群 (为了加快收敛，在起点和终点之间加入一定梯度的均匀分布再加噪声)
        positions = self._initialize_population() #np.zeros((self.pop_size, self.dim))
    

        # 记录全局最优
        best_position = np.zeros(self.dim)
        best_score = float('inf')
        score_history = []

        # 计算初始种群适应度
        for i in range(self.pop_size):
            path = self._decode_path(positions[i, :])
            score, _ , _= self.evaluator.evaluate_pso_particle(path)
            if score < best_score:
                best_score = score
                best_position = positions[i, :].copy()

        # 2. WOA 主循环
        for t in range(self.max_iter):
            # a 从 2 递减到 0 a = 2.0 - t * (2.0 / self.max_iter)
            k = 0.5  #引入非线性衰减
            a = 2.0 * ((1.0 - t / self.max_iter) ** k)
            
            
            for i in range(self.pop_size):
                r1 = random.random()
                r2 = random.random()
                
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = random.random()
                b = 1.0
                l = random.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) >= 1:
                        # 随机寻找猎物 (探索全局)
                        rand_idx = random.randint(0, self.pop_size - 1)
                        rand_pos = positions[rand_idx, :]
                        D_x_rand = abs(C * rand_pos - positions[i, :])
                        new_pos = rand_pos - A * D_x_rand
                    else:
                        # 包围猎物 (局部寻优)
                        D_Leader = abs(C * best_position - positions[i, :])
                        new_pos = best_position - A * D_Leader
                else:
                    # 气泡网攻击 (螺旋更新)
                    D_Leader = abs(best_position - positions[i, :])
                    new_pos = D_Leader * math.exp(b * l) * math.cos(2 * math.pi * l) + best_position

                # 边界处理 6/25改
                clipped_pos = np.clip(new_pos, self.effective_lb, self.effective_ub)

                # === 新增：位置更新后，立刻同步 Y 轴排序，防止基因错位 ===
                waypoints_temp = clipped_pos.reshape((self.num_waypoints, 2))
                waypoints_temp = waypoints_temp[np.argsort(waypoints_temp[:, 1])] # 按 Y 轴升序
                positions[i, :] = waypoints_temp.flatten() 

                # 评估新位置（此时 _decode_path 内部就不需要再重复 argsort 了）
                path = self._decode_path(positions[i, :])
                score, _, _ = self.evaluator.evaluate_pso_particle(path)

                # 更新全局最优 新解不坍缩
                if score < best_score and np.any(positions[i,:]):
                    best_score = score
                    best_position = positions[i, :].copy()

            self.convergence_curve.append(best_score)
            
            # 控制台输出进度 (每 50 次迭代)
            if (t + 1) % 50 == 0 or t == 0:
                print(f"迭代次数: {t + 1}/{self.max_iter}, 当前最优得分 (适应度): {best_score:.2f}")

        return self._decode_path(best_position), self.convergence_curve

    def plot_result(self, best_path, score_history):
        """
        绘制路径规划结果与适应度收敛曲线 (要求格式)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # ===== 图 1: 最终路径在环境中的可视化 =====
        self.env.draw_environment(ax=ax1)
        
        # 提取 x, y 坐标用于绘图
        path_x = [p[0] for p in best_path]
        path_y = [p[1] for p in best_path]
        
        
        smooth_best_path = self.evaluator.generate_chaikin_path(best_path, iterations=3)
        ax1.plot(smooth_best_path[:, 0], smooth_best_path[:, 1], color='#e65100', linewidth=3, 
                 label='WOA Smooth Path', zorder=6)
        ax1.plot(best_path[:, 0], best_path[:, 1], color='gray', linewidth=1, linestyle='--',
                 marker='o', markersize=5, label='Raw Waypoints', alpha=0.6, zorder=5)
        ax1.legend()
        # ===== 图 2: 适应度收敛曲线 =====
        ax2.plot(score_history, color='#1e88e5', linewidth=2)
        ax2.set_title("Fitness Convergence Curve", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Fitness Score", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_facecolor('#fafafa')
        params_text = (
            f"Algorithm Parameters:\n"
            f"  num_waypoints: {self.num_waypoints}\n"
            f"  pop_size:   {self.pop_size}\n"
            f"  max_iter:  {self.max_iter}\n"
            f"  Best Fitness: {score_history[-1]:.2f}"
        )
    
    # 放置在左上角（坐标轴相对坐标，0~1范围）
        ax2.text(0.5, 0.93, params_text,
             transform=ax2.transAxes,          # 使用相对坐标
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.tight_layout()
        plt.show()

# ================= 运行测试 =================
if __name__ == "__main__":
    # 实例化规划器，可适当增加控制点数量以绕开复杂障碍
    planner = WOAPlanner(num_waypoints=20, pop_size=60, max_iter=200)
    
    print("开始执行 WOA 鲸鱼算法路径规划...")
    best_path, convergence_history = planner.optimize()
    
    print(f"\n规划完成！最终得分: {convergence_history[-1]:.2f}")
    print(best_path)
    # 调用要求的方法进行绘图可视化
    planner.plot_result(best_path, convergence_history)
