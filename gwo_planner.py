import numpy as np
from base_planner import BasePlanner

class GWOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_wolves=120, max_iter=150, num_waypoints=16): 
        """ 
        继承自 BasePlanner
        【重大修正】：默认控制点数 num_waypoints 必须从 6 调大到 16！因为地图有 11 个打卡点，6 个点根本装不下！
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_wolves = num_wolves
        self.alpha_pos, self.alpha_score = np.zeros(self.dim), float("inf")
        self.beta_pos,  self.beta_score  = np.zeros(self.dim), float("inf")
        self.delta_pos, self.delta_score = np.zeros(self.dim), float("inf")
        self.stagnation_count, self.last_alpha_score = 0, float("inf")
        
        # 允许决策大脑（老中医）动态修改的停滞重置阈值
        self.stagnation_max = 30  
        
        # 【核心修改】：不再用纯随机，直接调用闭环雷达扫描初始化函数
        self.positions = self._initialize_wolves()

    def _initialize_wolves(self):
        """ 闭环专属初始化：雷达排序 + 目标注入 """
        positions = np.zeros((self.num_wolves, self.dim))
        targets = [t['center'] for t in self.env.target_areas]
        
        center_x = (self.env.x_bounds[0] + self.env.x_bounds[1]) / 2.0
        center_y = (self.env.y_bounds[0] + self.env.y_bounds[1]) / 2.0
        # 确保目标点本身也是顺着圆圈转的
        targets.sort(key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))

        for i in range(self.num_wolves):
            # 1. 先生成基础的随机点
            rand_x = np.random.uniform(self.env.x_bounds[0] + 5, self.env.x_bounds[1] - 5, self.num_waypoints)
            rand_y = np.random.uniform(self.env.y_bounds[0] + 5, self.env.y_bounds[1] - 5, self.num_waypoints)
            raw_pts = np.column_stack((rand_x, rand_y))

            # 2. 给前 30% 的精英狼注入打卡点
            if i < int(self.num_wolves * 0.3):
                for j, tgt in enumerate(targets):
                    if j < self.num_waypoints:
                        raw_pts[j] = tgt

            # 3. 【核心】对这条狼所有的点（目标点+随机点）进行极坐标雷达排序！
            # 这一步彻底消灭意大利面式的交叉打结！
            angles = np.arctan2(raw_pts[:, 1] - center_y, raw_pts[:, 0] - center_x)
            sorted_pts = raw_pts[np.argsort(angles)]
            
            positions[i] = np.clip(sorted_pts.flatten(), self.lb, self.ub)

        return positions

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

            # 使用变量 stagnation_max，让 CoordinatorAgent 药方能生效！
            if self.stagnation_count > getattr(self, 'stagnation_max', 30):
                self.alpha_score = self.beta_score = self.delta_score = float("inf")
                # 【修改】：核爆重置后，依然要用带目标基因和雷达排序的方法重生！
                self.positions = self._initialize_wolves()
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