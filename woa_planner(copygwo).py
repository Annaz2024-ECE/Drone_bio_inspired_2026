import numpy as np
from base_planner import BasePlanner

class WOAPlanner(BasePlanner): 
    def __init__(self, evaluator=None, num_waypoints=6, pop_size=50, max_iter=150):
        """
        采用自由坐标解码（不按Y轴排序）, 定点在巡检区域内，初始化随机分布，仿照GWO的强制重启
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        self.pop_size = pop_size
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.top_30_percent = int(self.pop_size * 0.3) 
        for i in range(self.top_30_percent):
            if self.dim >= 8:
                self.positions[i, 2], self.positions[i, 3] = 22.0, 40.0 
                self.positions[i, 6], self.positions[i, 7] = 78.0, 70.0
        self.stagnation_count = 0
        self.last_best_score = float('inf')
    

    def optimize(self):
        """
        实现核心 WOA 迭代寻优（自由解码版本）
        """
        print("开始【自由解码版】WOA 鲸鱼优化算法路径规划...")

        # 2. 评估初始最优
        for i in range(self.pop_size):
            path = self._decode_path(self.positions[i])
            score, _ , _ = self.evaluator.evaluate_pso_particle(path)
            if score < self.historical_best_score:
                self.historical_best_score = score
                self.historical_best_pos = self.positions[i].copy()

        # 3. WOA 主循环
        for t in range(self.max_iter):
            k = 0.5 #非线性衰减
            a = 2.0 * ((1.0 - t / self.max_iter) ** k)
            
            for i in range(self.pop_size):
                r1, r2 = np.random.random(), np.random.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = np.random.random()
                b = 1.0
                l = np.random.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) >= 1:
                        # 全局探索
                        rand_idx = np.random.randint(0, self.pop_size)
                        rand_pos = self.positions[rand_idx]
                        D_rand = np.abs(C * rand_pos - self.positions[i])
                        new_pos = rand_pos - A * D_rand
                    else:
                        # 局部寻优
                        D_Leader = np.abs(C * self.historical_best_pos - self.positions[i])
                        new_pos = self.historical_best_pos - A * D_Leader
                else:
                    # 气泡网攻击（螺旋更新）
                    D_Leader = np.abs(self.historical_best_pos - self.positions[i])
                    new_pos = D_Leader * np.exp(b * l) * np.cos(2 * np.pi * l) + self.historical_best_pos

                # 4. 边界裁剪
                new_pos = np.clip(new_pos, self.lb, self.ub)

                # 5. === 像 GWO 一样，在更新后重新对精英个体的指定维度强加紧箍咒 ===
                if i < self.top_30_percent and self.dim >= 8:
                    new_pos[2], new_pos[3] = 22.0, 40.0
                    new_pos[6], new_pos[7] = 78.0, 70.0

                self.positions[i] = new_pos

                # 6. 评估并动态更新基类的全局最优
                path = self._decode_path(self.positions[i])
                score, _ , _ = self.evaluator.evaluate_pso_particle(path)

                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = self.positions[i].copy()
            # 停滞检测与种群重启（仿 GWO）
            if abs(self.last_best_score - self.historical_best_score) < 1.0:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
                self.last_best_score = self.historical_best_score

            if self.stagnation_count > 30:
                print(f"迭代 {t+1}: 陷入停滞，重新初始化种群...")
                self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.positions[0] = self.historical_best_pos.copy()
                # 重新锁定精英锚点
                for j in range(self.top_30_percent):
                    if self.dim >= 8:
                        self.positions[j, 2], self.positions[j, 3] = 22.0, 40.0
                        self.positions[j, 6], self.positions[j, 7] = 78.0, 70.0
                self.stagnation_count = 0
                
            # 注入基类收敛曲线
            self.convergence_curve.append(self.historical_best_score)
            
            if (t + 1) % 50 == 0 or t == 0:
                print(f"  > 迭代 {t+1:03d}/{self.max_iter} | 历史最佳得分: {self.historical_best_score:,.2f}")

        return self._decode_path(self.historical_best_pos), self.convergence_curve
    
if __name__ == "__main__":
    planner = WOAPlanner(num_waypoints=10, pop_size=80, max_iter=150)
    best_path, history = planner.optimize()
    planner.plot_result(best_path, history, algo_name="WOA (copy GWO)")