import numpy as np
import math
import time
from base_planner import BasePlanner

class WOAPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_waypoints=25, pop_size=60, max_iter=200):
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.pop_size = pop_size
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.top_30_percent = int(self.pop_size * 0.3)
        
        #动态提取 JSON 中的 3D 巡检目标点
        self.target_anchors = []
        for target in self.env.target_areas:
            center_x, center_y = target['center'][:2]
            z_mid = (target.get('z_min', 4.0) + target.get('z_max', 8.0)) / 2.0
            self.target_anchors.append([center_x, center_y, z_mid])
        # 【优化 1】：参考论文，将纯随机初始化升级为“混合莱维飞行初始化”
        self.positions = self._hybrid_initialization()
    
    def _levy_step(self, dim):
        """
        使用经典 Mantegna 算法生成莱维飞行步长向量
        """
        beta = 1.5  # 论文推荐的特征指数
        
        # 计算 Mantegna 算法中的标准差 sigma_u
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        
        # 生成正态分布随机数
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        
        # 计算莱维步长
        step = u / (np.abs(v) ** (1 / beta))
        return step
    
    def _hybrid_initialization(self):
        """
        混合初始化：50% 传统随机分布，50% 施加莱维飞行大步长扰动
        """
        positions = np.zeros((self.pop_size, self.dim))
        
        # 前 50% 保持传统的均匀随机分布
        m1 = int(self.pop_size * 0.5)
        positions[:m1, :] = np.random.uniform(self.lb, self.ub, (m1, self.dim))
        
        # 后 50% 引入莱维飞行扰动，拓宽开局时航线在 3D 空间中的广度
        for i in range(m1, self.pop_size):
            base_pos = np.random.uniform(self.lb, self.ub, self.dim)
            levy_factor = self._levy_step(self.dim)
            # 扰动强度设为地图跨度的 10%，既能大跳又不会完全失控
            perturbed_pos = base_pos + levy_factor * 0.1 * (self.ub - self.lb)
            positions[i, :] = np.clip(perturbed_pos, self.lb, self.ub)
            
        return positions

    def _decode_path(self, position):
        """
        核心优化：使用最近邻法 (Nearest Neighbor) 动态理顺航点顺序。
        彻底解决同 Y 轴导致的左右横跳问题。
        """
        waypoints = position.reshape((self.num_waypoints, 3))
        
        sorted_waypoints = []
        # 以起点作为寻找下一个最近点的起始基准
        current_point = self.env.start_point 
        remaining_indices = list(range(self.num_waypoints))
        
        while remaining_indices:
            best_idx = -1
            min_dist = float('inf')
            
            # 遍历所有还没被连线的点，找离 current_point 最近的
            for idx in remaining_indices:
                # 计算 3D 欧氏距离
                dist = np.linalg.norm(waypoints[idx] - current_point)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
                    
            # 把找到的最近点加入有序列表，并将当前位置移动到该点
            sorted_waypoints.append(waypoints[best_idx])
            current_point = waypoints[best_idx]
            remaining_indices.remove(best_idx)
            
        sorted_waypoints = np.array(sorted_waypoints)
        full_path = np.vstack([self.env.start_point, sorted_waypoints, self.env.end_point])
        return full_path
      
    def optimize(self):
        print("开始 3D 环境下 WOA 鲸鱼优化算法路径规划(无安全边距+最近邻排序)...")
        
        for i in range(self.pop_size):
            path = self._decode_path(self.positions[i, :])
            score, _, _ = self.evaluator.evaluate_pso_particle(path)
            if score < self.historical_best_score:
                self.historical_best_score = score
                self.historical_best_pos = self.positions[i, :].copy()

        for t in range(self.max_iter):
            a = 2.0 * np.cos((np.pi * t) / (2 * self.max_iter))
            
            for i in range(self.pop_size):
                r1, r2 = np.random.random(), np.random.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = np.random.random()
                b = 1.0
                l = np.random.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) >= 1:
                        rand_idx = np.random.randint(0, self.pop_size)
                        rand_pos = self.positions[rand_idx, :]
                        D_x_rand = abs(C * rand_pos - self.positions[i, :])
                        
                        levy_jump = self._levy_step(self.dim)
                        # 将莱维大跳跃叠加到位置更新公式中（权重控制在 5% 以内保证收敛平稳）
                        new_pos = rand_pos - A * D_x_rand + levy_jump * 0.05 * (self.ub - self.lb)
                    else:
                        D_Leader = abs(C * self.historical_best_pos - self.positions[i, :])
                        new_pos = self.historical_best_pos - A * D_Leader
                else:
                    D_Leader = abs(self.historical_best_pos - self.positions[i, :])
                    new_pos = D_Leader * np.exp(b * l) * np.cos(2 * np.pi * l) + self.historical_best_pos

                
                clipped_pos = np.clip(new_pos, self.lb, self.ub)

                waypoints_temp = clipped_pos.reshape((self.num_waypoints, 3))
                
                # 注入 3D 打卡点
                if i < self.top_30_percent:
                    num_targets_to_inject = min(self.num_waypoints, len(self.target_anchors))
                    for j in range(num_targets_to_inject):
                        waypoints_temp[j] = self.target_anchors[j]

                # 注意：这里我们不再需要用 argsort 按 Y 轴排了！
                # 因为在评估时，_decode_path 会用贪心算法自动帮它们理顺！
                self.positions[i, :] = waypoints_temp.flatten()

                # 评估新位置
                path = self._decode_path(self.positions[i, :])
                score, _, _ = self.evaluator.evaluate_pso_particle(path)

                if score < self.historical_best_score and np.any(self.positions[i,:]):
                    self.historical_best_score = score
                    self.historical_best_pos = self.positions[i, :].copy()

            self.convergence_curve.append(self.historical_best_score)
            
            if (t + 1) % 50 == 0 or t == 0:
                print(f"迭代次数: {t + 1}/{self.max_iter}, 当前最优得分: {self.historical_best_score:,.2f}")

        return self._decode_path(self.historical_best_pos), self.convergence_curve

if __name__ == "__main__":
    start_time = time.time()  # 记录起点时间
    # 建议航点数至少大于等于目标区域数量 (haining.json5 中有 11 个 target)
    planner = WOAPlanner(num_waypoints=20, pop_size=60, max_iter=200)
    
    best_path, convergence_history = planner.optimize()
    
    print(f"\n规划完成！最终得分: {convergence_history[-1]:,.2f}")
    print(best_path)
    planner.plot_result(best_path, convergence_history, algo_name="WOA-3D")

    end_time = time.time()  # 记录起点时间
    elapsed_time = end_time - start_time  # 计算总差值（单位是秒）
    print("-" * 50)
    print(f"本次单线程运行总耗时: {elapsed_time:.2f} 秒 (约 {elapsed_time / 60:.2f} 分钟)")
    print("-" * 50)