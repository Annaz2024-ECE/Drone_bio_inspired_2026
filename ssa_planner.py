import numpy as np
from base_planner import BasePlanner

class SSAPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_sparrows=100, max_iter=200, num_waypoints=15):
        """ 继承自 BasePlanner 的 SSA 麻雀搜索算法 """
        # 调用父类初始化，统管环境、维度、边界等
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_sparrows = num_sparrows
        
        # SSA 核心参数
        self.PD = 0.2  # 发现者比例
        self.SD = 0.1  # 侦察者比例
        self.ST = 0.8  # 安全阈值
        
        self.num_producers = int(self.num_sparrows * self.PD)
        self.num_scouts = int(self.num_sparrows * self.SD)
        
        # 初始化麻雀位置与适应度
        self.sparrows = self._initialize_sparrows()
        self.fitness = np.full(num_sparrows, np.inf)
        
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

    def _initialize_sparrows(self):
        """ 初始化麻雀位置，并引入投影排序防打结策略 """
        sparrows = np.zeros((self.num_sparrows, self.dim))
        direction_vec = self.env.end_point - self.env.start_point
        
        for i in range(self.num_sparrows):
            # 在起终点之间均匀取点
            x_vals = np.linspace(self.env.start_point[0], self.env.end_point[0], self.num_waypoints + 2)[1:-1]
            y_vals = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_waypoints + 2)[1:-1]
            
            # 加入全图高强度随机扰动
            noise_x = np.random.uniform(-40, 40, self.num_waypoints)
            noise_y = np.random.uniform(-40, 40, self.num_waypoints)
            raw_waypoints = np.column_stack((x_vals + noise_x, y_vals + noise_y))
            
            # 【防打结神技】将控制点向主方向投影并排序
            projections = np.dot(raw_waypoints - self.env.start_point, direction_vec)
            sorted_waypoints = raw_waypoints[np.argsort(projections)]
            
            # 将 2D 坐标展平为 1D，并限制在边界内
            sparrows[i] = np.clip(sorted_waypoints.flatten(), self.lb, self.ub)
            
        return sparrows

    def optimize(self):
        """ 运行 SSA 主循环 """
        print("🚀 开始 SSA 麻雀搜索算法路径规划...")
        
        # 1. 初始适应度评估
        for i in range(self.num_sparrows):
            full_path = self._decode_path(self.sparrows[i])
            self.fitness[i] = self.evaluator.evaluate_pso_particle(full_path)
            if self.fitness[i] < self.gbest_score:
                self.gbest_score = self.fitness[i]
                self.gbest_pos = np.copy(self.sparrows[i])
                
        # 2. 核心迭代寻优
        for iteration in range(self.max_iter):
            sort_indices = np.argsort(self.fitness)
            best_pos_current = np.copy(self.sparrows[sort_indices[0]])
            worst_pos_current = np.copy(self.sparrows[sort_indices[-1]])
            worst_fit_current = self.fitness[sort_indices[-1]]
            
            new_sparrows = np.copy(self.sparrows)
            
            # (1) 发现者 (Producers) 更新位置
            R2 = np.random.rand()
            for i in range(self.num_producers):
                idx = sort_indices[i]
                if R2 < self.ST:
                    alpha = np.random.rand()
                    step = np.random.randn(self.dim) * 15.0 
                    new_sparrows[idx] = self.sparrows[idx] + step * np.exp(-(iteration + 1) / (alpha * self.max_iter + 1e-8))
                else:
                    new_sparrows[idx] = self.sparrows[idx] + np.random.randn(self.dim) * 2.0
                    
            # (2) 加入者 (Scroungers) 更新位置
            for i in range(self.num_producers, self.num_sparrows):
                idx = sort_indices[i]
                if i > self.num_sparrows / 2:
                    # 极度饥饿的麻雀，全图随机重生，保证种群基因多样性
                    new_sparrows[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    # 向当前最优位置靠拢
                    A = np.random.choice([-1, 1], size=self.dim)
                    new_sparrows[idx] = best_pos_current + np.abs(self.sparrows[idx] - best_pos_current) * (A / 2.0)
                    
            # (3) 侦察者/警戒者 (Scouts) 更新位置
            scout_indices = np.random.choice(self.num_sparrows, self.num_scouts, replace=False)
            for idx in scout_indices:
                if self.fitness[idx] > self.fitness[sort_indices[0]]:
                    # 处于边缘的麻雀向中心靠拢
                    new_sparrows[idx] = best_pos_current + np.random.randn(self.dim) * np.abs(self.sparrows[idx] - best_pos_current)
                else:
                    # 处于中心的麻雀随机逃窜
                    new_sparrows[idx] = self.sparrows[idx] + np.random.uniform(-1, 1) * (np.abs(self.sparrows[idx] - worst_pos_current) / (self.fitness[idx] - worst_fit_current + 1e-8))

            # (4) 越界处理与适应度重新评估
            for i in range(self.num_sparrows):
                new_sparrows[i] = np.clip(new_sparrows[i], self.lb, self.ub)
                score = self.evaluator.evaluate_pso_particle(self._decode_path(new_sparrows[i]))
                
                self.sparrows[i] = new_sparrows[i]
                self.fitness[i] = score
                
                # 更新全局最优
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos = np.copy(new_sparrows[i])
                    
            # 记录历史最优得分
            self.convergence_curve.append(self.gbest_score)
            
            # 控制台输出
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 全局最优得分: {self.gbest_score:.2f}")
                
        # 统一返回：解码后的最优 2D 路径，以及历史收敛曲线
        return self._decode_path(self.gbest_pos), self.convergence_curve

# ===================== 本地单文件测试 =====================
if __name__ == "__main__":
    planner = SSAPlanner()
    best_path, history = planner.optimize()
    
    # 完美复用 BasePlanner 的画图功能，自动抓取参数！
    planner.plot_result(best_path, history, algo_name="SSA")