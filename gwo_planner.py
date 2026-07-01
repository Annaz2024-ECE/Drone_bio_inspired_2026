import numpy as np
from base_planner import BasePlanner

class GWOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_wolves=120, max_iter=150, num_waypoints=6):
        """ 继承自 BasePlanner，统一输出接口格式为2个返回值 """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_wolves = num_wolves
        self.alpha_pos, self.alpha_score = np.zeros(self.dim), float("inf")
        self.beta_pos,  self.beta_score  = np.zeros(self.dim), float("inf")
        self.delta_pos, self.delta_score = np.zeros(self.dim), float("inf")
        self.stagnation_count, self.last_alpha_score = 0, float("inf")
        
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
        
        # 强制精英狼踩在目标点上
        # 强制精英狼踩在目标点上 (全自动环境读取版)
        top_30_percent = int(self.num_wolves * 0.3)
        
        # 1. 动态从环境读取所有打卡点的坐标
        targets = [t['center'] for t in self.env.target_areas]
        
        # 2. 顺手加个“雷达扫描”排序，防止打卡点连线交叉打结
        center_x = (self.env.x_bounds[0] + self.env.x_bounds[1]) / 2.0
        center_y = (self.env.y_bounds[0] + self.env.y_bounds[1]) / 2.0
        targets.sort(key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))
        
        # 3. 自动把排好序的打卡点，按顺序塞进精英狼的基因(数组)里
        for i in range(top_30_percent):
            for j, target_pt in enumerate(targets):
                idx_x = j * 2
                idx_y = j * 2 + 1
                # 只要控制点维度够装，就一直往里塞
                if idx_y < self.dim:
                    self.positions[i, idx_x] = target_pt[0]
                    self.positions[i, idx_y] = target_pt[1]

    def optimize(self):
        print("开始 GWO 灰狼优化算法路径规划...")
        
        for l in range(self.max_iter):
            for i in range(self.num_wolves):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fitness, _ , _ = self.evaluator.evaluate_pso_particle(self._decode_path(self.positions[i]))
                
                if fitness < self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = fitness, self.positions[i].copy()
                    if self.alpha_score < self.historical_best_score:
                        self.historical_best_score, self.historical_best_pos = self.alpha_score, self.alpha_pos.copy()
                elif fitness < self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = fitness, self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score, self.delta_pos = fitness, self.positions[i].copy()
            
            a = 2.0 * (1.0 - (l / self.max_iter) ** 2)
            epsilon = 1e-8
            w_sum = 1.0/(self.alpha_score+epsilon) + 1.0/(self.beta_score+epsilon) + 1.0/(self.delta_score+epsilon)
            w_alpha = (1.0/(self.alpha_score+epsilon)) / w_sum
            w_beta  = (1.0/(self.beta_score+epsilon))  / w_sum
            w_delta = (1.0/(self.delta_score+epsilon)) / w_sum
            
            # 使用 NumPy 的矩阵向量化技术极大提升计算速度
            r1_a, r2_a = np.random.random((self.num_wolves, self.dim)), np.random.random((self.num_wolves, self.dim))
            r1_b, r2_b = np.random.random((self.num_wolves, self.dim)), np.random.random((self.num_wolves, self.dim))
            r1_d, r2_d = np.random.random((self.num_wolves, self.dim)), np.random.random((self.num_wolves, self.dim))
            
            X1 = self.alpha_pos - (2*a*r1_a - a) * np.abs(2*r2_a * self.alpha_pos - self.positions)
            X2 = self.beta_pos - (2*a*r1_b - a) * np.abs(2*r2_b * self.beta_pos - self.positions)
            X3 = self.delta_pos - (2*a*r1_d - a) * np.abs(2*r2_d * self.delta_pos - self.positions)
            
            self.positions = w_alpha * X1 + w_beta * X2 + w_delta * X3
            
            if abs(self.last_alpha_score - self.alpha_score) < 1.0: self.stagnation_count += 1
            else: self.stagnation_count, self.last_alpha_score = 0, self.alpha_score

            if self.stagnation_count > 30:
                self.alpha_score = self.beta_score = self.delta_score = float("inf")
                self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
                self.stagnation_count = 0 
            
            self.convergence_curve.append(self.historical_best_score)
            if (l + 1) % 50 == 0 or l == 0:
                print(f"  > 迭代 {l+1:03d}/{self.max_iter} | 历史最佳得分: {self.historical_best_score:,.2f}")
                
        # 统一返回两个参数：解码后的2D最优路径，以及历史收敛曲线
        return self._decode_path(self.historical_best_pos), self.convergence_curve

if __name__ == "__main__":
    planner = GWOPlanner()
    best_path, history = planner.optimize()
    planner.plot_result(best_path, history, algo_name="GWO")