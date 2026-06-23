import numpy as np
import matplotlib.pyplot as plt
import math
import random
from environment_buildup import UAVEnvironment2D
from path_evaluator import PathEvaluator

class WOAPathPlanner:
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
        self.max_iter = max_iter
        
        # 优化维度为: 中间航点数 * 2 (每个点有 X 和 Y)
        self.dim = self.num_waypoints * 2 
        
        # 定义搜索空间边界
        self.lb = np.array([self.env.x_bounds[0], self.env.y_bounds[0]] * self.num_waypoints)
        self.ub = np.array([self.env.x_bounds[1], self.env.y_bounds[1]] * self.num_waypoints)

    def _decode_path(self, position):
        waypoints = position.reshape((self.num_waypoints, 2))
        # 按照 Y 轴坐标从小到大排序控制点
        waypoints = waypoints[np.argsort(waypoints[:, 1])] 
        path = [self.env.start_point] + waypoints.tolist() + [self.env.end_point]
        return np.array(path)
    def _initialize_population(self):
        """
        修复后的启发式种群初始化：精准控制切片数量，杜绝维度不匹配报错
        """
        positions = np.zeros((self.pop_size, self.dim))
        
        # 获取目标区的中心坐标
        west_center = self.env.target_areas[0]['center'] # [22.0, 40.0]
        east_center = self.env.target_areas[1]['center'] # [78.0, 70.0]
        
        # 计算每种策略分配的鲸鱼数量
        tier_size = self.pop_size // 3
        
        for i in range(self.pop_size):
            if i < tier_size:
                # ==== 梯队1 (中路)：起点到终点的直线插值 ====
                x_vals = np.linspace(self.env.start_point[0], self.env.end_point[0], self.num_waypoints + 2)[1:-1].tolist()
                y_vals = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_waypoints + 2)[1:-1].tolist()
                
            elif i < 2 * tier_size:
                # ==== 梯队2 (西区先锋)：精准分段插值 ====
                mid_idx = self.num_waypoints // 2
                
                # 1. 从起点到西区中心：生成 mid_idx 个过渡点 + 1 个中心点，切掉起点
                x_vals1 = np.linspace(self.env.start_point[0], west_center[0], mid_idx + 2)[1:].tolist()
                y_vals1 = np.linspace(self.env.start_point[1], west_center[1], mid_idx + 2)[1:].tolist()
                
                # 2. 从西区中心到终点：补齐剩下的点数，切掉西区中心和终点
                rem_points = self.num_waypoints - len(x_vals1)
                x_vals2 = np.linspace(west_center[0], self.env.end_point[0], rem_points + 2)[1:-1].tolist()
                y_vals2 = np.linspace(west_center[1], self.env.end_point[1], rem_points + 2)[1:-1].tolist()
                
                x_vals = x_vals1 + x_vals2
                y_vals = y_vals1 + y_vals2
                
            else:
                # ==== 梯队3 (东区先锋)：精准分段插值 ====
                mid_idx = self.num_waypoints // 2
                
                # 1. 从起点到东区中心
                x_vals1 = np.linspace(self.env.start_point[0], east_center[0], mid_idx + 2)[1:].tolist()
                y_vals1 = np.linspace(self.env.start_point[1], east_center[1], mid_idx + 2)[1:].tolist()
                
                # 2. 从东区中心到终点
                rem_points = self.num_waypoints - len(x_vals1)
                x_vals2 = np.linspace(east_center[0], self.env.end_point[0], rem_points + 2)[1:-1].tolist()
                y_vals2 = np.linspace(east_center[1], self.env.end_point[1], rem_points + 2)[1:-1].tolist()
                
                x_vals = x_vals1 + x_vals2
                y_vals = y_vals1 + y_vals2
                
            # 将 X 和 Y 重新拼回一维数组
            base = np.column_stack((x_vals, y_vals)).flatten()
            
            # 生成与 base 相同维度的噪声 (由于修复了数量，这里必然都是 self.dim)
            noise = np.random.uniform(-5, 5, self.dim) if i >= tier_size else np.random.uniform(-20, 20, self.dim)
            
            # 基础位置加扰动，并限制在边界内
            positions[i, :] = np.clip(base + noise, self.lb, self.ub)
            
        return positions
      
    
    def optimize(self):
        """
        执行鲸鱼优化算法的主循环
        """
        # 1. 初始化种群 (为了加快收敛，在起点和终点之间加入一定梯度的均匀分布再加噪声)
        positions = self._initialize_population() #np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            # 基础线性插值点
            x_vals = np.linspace(self.env.start_point[0], self.env.end_point[0], self.num_waypoints + 2)[1:-1]
            y_vals = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_waypoints + 2)[1:-1]
            
            # 增加全图随机扰动，提升全局探索能力
            base_points = np.column_stack((x_vals, y_vals)).flatten()
            noise = np.random.uniform(-30, 30, self.dim) 
            
            positions[i, :] = np.clip(base_points + noise, self.lb, self.ub)

        # 记录全局最优
        best_position = np.zeros(self.dim)
        best_score = float('inf')
        score_history = []

        # 计算初始种群适应度
        for i in range(self.pop_size):
            path = self._decode_path(positions[i, :])
            score = self.evaluator.evaluate_pso_particle(path)
            if score < best_score:
                best_score = score
                best_position = positions[i, :].copy()

        # 2. WOA 主循环
        for t in range(self.max_iter):
            # a 从 2 线性递减到 0
            a = 2.0 - t * (2.0 / self.max_iter)
            
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

                # 边界处理
                positions[i, :] = np.clip(new_pos, self.lb, self.ub)

                # 评估新位置
                path = self._decode_path(positions[i, :])
                score = self.evaluator.evaluate_pso_particle(path)

                # 更新全局最优
                if score < best_score:
                    best_score = score
                    best_position = positions[i, :].copy()

            score_history.append(best_score)
            
            # 控制台输出进度 (每 50 次迭代)
            if (t + 1) % 50 == 0:
                print(f"迭代次数: {t + 1}/{self.max_iter}, 当前最优得分 (适应度): {best_score:.2f}")

        # 解码最终最优路径
        best_path = self._decode_path(best_position)
        return best_path, score_history

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
            f"  Best Fitness: {convergence_history[-1]:.2f}"
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
    planner = WOAPathPlanner(num_waypoints=10, pop_size=60, max_iter=200)
    
    print("开始执行 WOA 鲸鱼算法路径规划...")
    best_path, convergence_history = planner.optimize()
    
    print(f"\n规划完成！最终得分: {convergence_history[-1]:.2f}")
    print(best_path)
    # 调用要求的方法进行绘图可视化
    planner.plot_result(best_path, convergence_history)