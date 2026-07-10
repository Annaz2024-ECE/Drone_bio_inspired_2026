import numpy as np
from base_planner import BasePlanner

class HybridPSOGWO(BasePlanner):
    def __init__(self, evaluator=None, pop_size=100, max_iter=150, pso_ratio=0.3, num_waypoints=16): 
        """
        [最强缝合怪] PSO + GWO 混合仿生路径规划算法
        :param pso_ratio: PSO 阶段占总迭代次数的比例 (默认 30% 探路，70% 灰狼精修)
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        # 强制覆盖 3D 维度与边界 (防止父类是 2D 老版本)
        self.dim = self.num_waypoints * 3
        self.lb = np.tile([self.env.x_bounds[0], self.env.y_bounds[0], max(5.0, self.env.z_bounds[0])], self.num_waypoints)
        self.ub = np.tile([self.env.x_bounds[1], self.env.y_bounds[1], self.env.z_bounds[1]], self.num_waypoints)
        
        self.pop_size = pop_size
        self.default_pso_ratio = pso_ratio  # 保存一个初始默认比例
        
        # 初始化种群位置 (使用强大的 Cubic 混沌 + 靶向注入)
        self.positions = self._initialize_population()
        
        # ==========================================
        # PSO 专属参数
        # ==========================================
        self.V = np.zeros((self.pop_size, self.dim))
        self.pbest_pos = self.positions.copy()
        self.pbest_score = np.full(self.pop_size, float("inf"))
        
        # ==========================================
        # GWO 专属参数
        # ==========================================
        self.alpha_pos, self.alpha_score = np.zeros(self.dim), float("inf")
        self.beta_pos,  self.beta_score  = np.zeros(self.dim), float("inf")
        self.delta_pos, self.delta_score = np.zeros(self.dim), float("inf")

    def _decode_path(self, position):
        """ 强制覆盖为 3D 路径还原 """
        waypoints = position.reshape((self.num_waypoints, 3))
        full_path = np.vstack([self.env.start_point, waypoints, self.env.end_point])
        return full_path

    def _cubic_chaotic_map(self, size):
        rho = 2.595  
        chaos_seq = np.zeros(size)
        x = np.random.rand() * 2 - 1 
        if x == 0: x = 0.1 
        for i in range(size):
            x = rho * x * (1 - x**2)
            chaos_seq[i] = x
        return (chaos_seq + 1.0) / 2.0

    def _initialize_population(self):
        """ Cubic 混沌 + 3D目标靶向注入 """
        positions = np.zeros((self.pop_size, self.dim))
        
        targets_3d = []
        for t in self.env.target_areas:
            z_mid = (t.get('z_min', 0.0) + t.get('z_max', 20.0)) / 2.0
            targets_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
        
        center_x = (self.env.x_bounds[0] + self.env.x_bounds[1]) / 2.0
        center_y = (self.env.y_bounds[0] + self.env.y_bounds[1]) / 2.0
        targets_3d.sort(key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))

        for i in range(self.pop_size):
            chaos_x = self._cubic_chaotic_map(self.num_waypoints)
            chaos_y = self._cubic_chaotic_map(self.num_waypoints)
            chaos_z = self._cubic_chaotic_map(self.num_waypoints)
            
            rand_x = self.env.x_bounds[0] + chaos_x * (self.env.x_bounds[1] - self.env.x_bounds[0])
            rand_y = self.env.y_bounds[0] + chaos_y * (self.env.y_bounds[1] - self.env.y_bounds[0])
            rand_z = max(5.0, self.env.z_bounds[0]) + chaos_z * (self.env.z_bounds[1] - max(5.0, self.env.z_bounds[0]))
            
            raw_pts = np.column_stack((rand_x, rand_y, rand_z))

            if i < int(self.pop_size * 0.3):
                for j, tgt in enumerate(targets_3d):
                    if j < self.num_waypoints:
                        raw_pts[j] = tgt

            angles = np.arctan2(raw_pts[:, 1] - center_y, raw_pts[:, 0] - center_x)
            sorted_pts = raw_pts[np.argsort(angles)]
            positions[i] = np.clip(sorted_pts.flatten(), self.lb, self.ub)

        return positions

    def optimize(self):
        # 🔥 核心升级 1：动态读取老中医可能修改过的 pso_ratio 重新分配算力
        current_pso_ratio = getattr(self, 'pso_ratio', self.default_pso_ratio)
        self.pso_iters = int(self.max_iter * current_pso_ratio)
        self.gwo_iters = self.max_iter - self.pso_iters

        print("\n" + "="*50)
        print(" 🚀 开始执行 PSO-GWO 混合算法 (双擎驱动)")
        print(f"    - 上半场 (PSO探路): {self.pso_iters} 代")
        print(f"    - 下半场 (GWO精修): {self.gwo_iters} 代")
        print("="*50 + "\n")
        
        # ==========================================
        # 【上半场】：PSO 粒子群探路阶段
        # ==========================================
        if self.pso_iters > 0:
            print(" 🦅 [Phase 1/2] 启动 PSO 粒子群大范围搜索...")
            
        for l in range(self.pso_iters):
            w = 0.9 - 0.5 * (l / max(1, self.pso_iters))  # 惯性权重递减
            c1, c2 = 2.0, 2.0
            
            for i in range(self.pop_size):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fitness, _, _ = self.evaluator.evaluate_pso_particle(self._decode_path(self.positions[i]))
                
                # 更新个体最优
                if fitness < self.pbest_score[i]:
                    self.pbest_score[i] = fitness
                    self.pbest_pos[i] = self.positions[i].copy()
                    
                # 更新全局最优
                if fitness < self.historical_best_score:
                    self.historical_best_score = fitness
                    self.historical_best_pos = self.positions[i].copy()
                    
            # PSO 位置与速度更新
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            cognitive = c1 * r1 * (self.pbest_pos - self.positions)
            social = c2 * r2 * (self.historical_best_pos - self.positions)
            
            self.V = w * self.V + cognitive + social
            self.V = np.clip(self.V, -0.2*(self.ub-self.lb), 0.2*(self.ub-self.lb)) # 限制最大速度
            self.positions += self.V
            self.positions = np.clip(self.positions, self.lb, self.ub)
            
            # ==========================================
            # 🔥 核心升级 2：上半场 PSO 更新后，启动通用物理引擎！
            # 无论拉普拉斯平滑还是推离墙壁，统统生效！
            self.execute_universal_physics_directives()
            # ==========================================
            
            self.convergence_curve.append(self.historical_best_score)
            if (l + 1) % 10 == 0:
                print(f"   [PSO] 迭代 {l+1:03d}/{self.pso_iters} | 历史最佳得分: {self.historical_best_score:,.2f}")

        # ==========================================
        # 【下半场】：GWO 灰狼精英包围阶段 (无缝接力)
        # ==========================================
        if self.gwo_iters > 0:
            print("\n 🐺 [Phase 2/2] 启动 GWO 灰狼动态加权包围收缩...")
            # 将 PSO 的历史最优作为 GWO 开局的 Alpha 狼，防止丢掉好基因
            self.alpha_score = self.historical_best_score
            self.alpha_pos = self.historical_best_pos.copy()

        for l in range(self.gwo_iters):
            for i in range(self.pop_size):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fitness, _, _ = self.evaluator.evaluate_pso_particle(self._decode_path(self.positions[i]))
                
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

            # GWO 独有的非线性收缩因子
            a = 2.0 * (1.0 - (l / max(1, self.gwo_iters)) ** 2) 
            
            # 动态适应度权重分配
            epsilon = 1e-8
            w_sum = 1.0/(self.alpha_score+epsilon) + 1.0/(self.beta_score+epsilon) + 1.0/(self.delta_score+epsilon)
            w_alpha = (1.0/(self.alpha_score+epsilon)) / w_sum
            w_beta  = (1.0/(self.beta_score+epsilon))  / w_sum
            w_delta = (1.0/(self.delta_score+epsilon)) / w_sum
            
            r1_a, r2_a = np.random.random((self.pop_size, self.dim)), np.random.random((self.pop_size, self.dim))
            r1_b, r2_b = np.random.random((self.pop_size, self.dim)), np.random.random((self.pop_size, self.dim))
            r1_d, r2_d = np.random.random((self.pop_size, self.dim)), np.random.random((self.pop_size, self.dim))
            
            X1 = self.alpha_pos - (2*a*r1_a - a) * np.abs(2*r2_a * self.alpha_pos - self.positions)
            X2 = self.beta_pos - (2*a*r1_b - a) * np.abs(2*r2_b * self.beta_pos - self.positions)
            X3 = self.delta_pos - (2*a*r1_d - a) * np.abs(2*r2_d * self.delta_pos - self.positions)
            
            self.positions = w_alpha * X1 + w_beta * X2 + w_delta * X3
            
            # 【精英保护局部变异】防止 GWO 后期死锁
            mutation_rate = 0.15 * (1.0 - l / max(1, self.gwo_iters))
            do_mutation = np.random.rand(self.pop_size) < 0.2
            do_mutation[0] = False 
            noise = np.random.randn(self.pop_size, self.dim) * (self.ub - self.lb) * mutation_rate
            self.positions[do_mutation] = (self.alpha_pos + noise)[do_mutation]
            self.positions = np.clip(self.positions, self.lb, self.ub)
            
            # ==========================================
            # 🔥 核心升级 3：下半场 GWO 更新后，同样启动通用物理引擎！
            self.execute_universal_physics_directives()
            # ==========================================
            
            self.convergence_curve.append(self.historical_best_score)
            if (l + 1) % 20 == 0 or l == self.gwo_iters - 1:
                print(f"   [GWO] 迭代 {l+1:03d}/{self.gwo_iters} | 历史最佳得分: {self.historical_best_score:,.2f}")

        print("\n 🎉 混合算法优化完成！")
        return self._decode_path(self.historical_best_pos), self.convergence_curve

if __name__ == "__main__":
    planner = HybridPSOGWO(max_iter=150, pso_ratio=0.3)
    best_path, history = planner.optimize()
    planner.plot_result(best_path, history, algo_name="Hybrid_PSO_GWO")