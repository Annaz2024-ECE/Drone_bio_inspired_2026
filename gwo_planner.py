import numpy as np
from base_planner import BasePlanner

class GWOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_wolves=120, max_iter=150, num_waypoints=16): 
        """ 
        继承自 BasePlanner
        默认控制点数 num_waypoints 必须从 6 调大到 16, 因为地图有 11 个打卡点, 6 个点根本装不下！
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_wolves = num_wolves
        self.alpha_pos, self.alpha_score = np.zeros(self.dim), float("inf")
        self.beta_pos,  self.beta_score  = np.zeros(self.dim), float("inf")
        self.delta_pos, self.delta_score = np.zeros(self.dim), float("inf")
        self.stagnation_count, self.last_alpha_score = 0, float("inf")
        
        # 允许决策大脑（老中医）动态修改的停滞重置阈值
        self.stagnation_max = 30  
        
        # 闭环雷达扫描初始化函数
        self.positions = self._initialize_wolves()

    def _cubic_chaotic_map(self, size):
        """ Cubic 立方混沌映射生成器 """
        rho = 2.595  
        chaos_seq = np.zeros(size)
        x = np.random.rand() * 2 - 1 
        if x == 0: x = 0.1 
            
        for i in range(size):
            x = rho * x * (1 - x**2)
            chaos_seq[i] = x
            
        return (chaos_seq + 1.0) / 2.0

    def _initialize_wolves(self):
        """ 闭环专属初始化：雷达排序 + 3D目标注入 """
        positions = np.zeros((self.num_wolves, self.dim))
        
        targets_3d = []
        for t in self.env.target_areas:
            z_mid = (t.get('z_min', 0.0) + t.get('z_max', 20.0)) / 2.0
            targets_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
        
        center_x = (self.env.x_bounds[0] + self.env.x_bounds[1]) / 2.0
        center_y = (self.env.y_bounds[0] + self.env.y_bounds[1]) / 2.0
        
        targets_3d.sort(key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))

        for i in range(self.num_wolves):
            chaos_x = self._cubic_chaotic_map(self.num_waypoints)
            chaos_y = self._cubic_chaotic_map(self.num_waypoints)
            chaos_z = self._cubic_chaotic_map(self.num_waypoints)
            
            rand_x = self.env.x_bounds[0] + chaos_x * (self.env.x_bounds[1] - self.env.x_bounds[0])
            rand_y = self.env.y_bounds[0] + chaos_y * (self.env.y_bounds[1] - self.env.y_bounds[0])
            rand_z = max(5.0, self.env.z_bounds[0]) + chaos_z * (self.env.z_bounds[1] - max(5.0, self.env.z_bounds[0]))
            
            raw_pts = np.column_stack((rand_x, rand_y, rand_z))

            if i < int(self.num_wolves * 0.3):
                for j, tgt in enumerate(targets_3d):
                    if j < self.num_waypoints:
                        raw_pts[j] = tgt

            angles = np.arctan2(raw_pts[:, 1] - center_y, raw_pts[:, 0] - center_x)
            sorted_pts = raw_pts[np.argsort(angles)]
            positions[i] = np.clip(sorted_pts.flatten(), self.lb, self.ub)

        return positions

    def optimize(self):
        print("开始 GWO 灰狼优化算法路径规划 (搭载四大通用物理引擎 + 拉普拉斯平滑)...")
        
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
            
            r1_a, r2_a = np.random.random((self.num_wolves, self.dim)), np.random.random((self.num_wolves, self.dim))
            r1_b, r2_b = np.random.random((self.num_wolves, self.dim)), np.random.random((self.num_wolves, self.dim))
            r1_d, r2_d = np.random.random((self.num_wolves, self.dim)), np.random.random((self.num_wolves, self.dim))
            
            X1 = self.alpha_pos - (2*a*r1_a - a) * np.abs(2*r2_a * self.alpha_pos - self.positions)
            X2 = self.beta_pos - (2*a*r1_b - a) * np.abs(2*r2_b * self.beta_pos - self.positions)
            X3 = self.delta_pos - (2*a*r1_d - a) * np.abs(2*r2_d * self.delta_pos - self.positions)
            
            self.positions = w_alpha * X1 + w_beta * X2 + w_delta * X3
            
            # 🔥 统一物理指令接收器 (Universal API)
            # 这里也加入了接收Agent对变异率的直接修改！
            default_rate = 0.2
            default_scale = 0.2 * (1.0 - l / self.max_iter) 
            
            mutation_rate = getattr(self, 'mutation_rate', default_rate)
            mutation_scale = getattr(self, 'mutation_scale', default_scale)

            # 监听老中医广播的四大信号
            is_emergency = getattr(self, 'emergency_escape', False)
            is_press_down = getattr(self, 'press_down', False) 
            is_lift_up = getattr(self, 'lift_up', False)        
            is_radar = getattr(self, 'radar_guidance', False)   
            
            if is_radar:
                mutation_rate = 0.1  # 空投 10% 敢死队
            elif is_emergency:
                mutation_rate = 0.5  # 大乱窜
                mutation_scale = 0.5 
            elif is_lift_up:
                mutation_rate = 0.4  # 拉升力度较大
                mutation_scale = 0.2
            elif is_press_down:
                mutation_rate = 0.3  # 试探低空
                mutation_scale = 0.1 
            
            do_mutation = np.random.rand(self.num_wolves) < mutation_rate
            do_mutation[0] = False 
            do_mutation[1] = False 
            
            if is_radar:
                # 触发雷达技能：敢死狼瞬间传送到目标
                targets_3d = []
                for t in self.env.target_areas:
                    z_mid = (t.get('z_min', 0.0) + t.get('z_max', 20.0)) / 2.0
                    targets_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
                for i in range(self.num_wolves):
                    if do_mutation[i]:
                        new_pos = np.zeros((self.num_waypoints, 3))
                        if len(targets_3d) > 0:
                            for j in range(self.num_waypoints):
                                new_pos[j] = targets_3d[j % len(targets_3d)]
                        noise = np.random.randn(self.num_waypoints, 3) * 2.0
                        self.positions[i] = np.clip((new_pos + noise).flatten(), self.lb, self.ub)
            else:
                # 触发物理推力技能：Z轴强制位移
                noise = np.random.randn(self.num_wolves, self.dim) * (self.ub - self.lb) * mutation_scale
                
                if is_emergency:
                    for d in range(2, self.dim, 3): noise[:, d] += 15.0 # 原地升天起飞
                elif is_lift_up:
                    for d in range(2, self.dim, 3): noise[:, d] += 8.0  # 遇墙紧急拉升 8 米
                elif is_press_down:
                    for d in range(2, self.dim, 3): noise[:, d] -= 3.0  # 安全时向下试探 3 米
                        
                mutated_pos = self.alpha_pos + noise
                self.positions[do_mutation] = mutated_pos[do_mutation]
                
            self.execute_universal_physics_directives() #直接调用基类的通用物理引擎，进行拉普拉斯平滑

            if abs(self.last_alpha_score - self.alpha_score) < 1.0: self.stagnation_count += 1
            else: self.stagnation_count, self.last_alpha_score = 0, self.alpha_score

            if self.stagnation_count > getattr(self, 'stagnation_max', 30):
                self.alpha_score = self.beta_score = self.delta_score = float("inf")
                self.positions = self._initialize_wolves()
                self.stagnation_count = 0 
            
            self.convergence_curve.append(self.historical_best_score)
            if (l + 1) % 50 == 0 or l == 0:
                print(f"  > 迭代 {l+1:03d}/{self.max_iter} | 历史最佳得分: {self.historical_best_score:,.2f}")
                
        return self._decode_path(self.historical_best_pos), self.convergence_curve

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from path_evaluator import PathEvaluator

    evaluator = PathEvaluator()
    planner = GWOPlanner(evaluator=evaluator)
    best_path, history = planner.optimize()
    
    print("\n 正在绘制狼群最终分布 3D 散点图...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    evaluator.env.draw_environment_3d(ax)
    
    for i in range(planner.num_wolves):
        wolf_pts = planner.positions[i].reshape((planner.num_waypoints, 3))
        ax.scatter(wolf_pts[:, 0], wolf_pts[:, 1], wolf_pts[:, 2], 
                   c='blue', alpha=0.15, s=15, marker='o')
                   
    smooth_path = evaluator.generate_bspline_path(best_path, num_points=100)
    ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2], 
            color='#FF007F', linewidth=4, label='Alpha Wolf (Best Path)')
            
    ax.set_title("GWO Final Population Distribution (Checking Stagnation)", fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    planner.plot_result(best_path, history, algo_name="GWO_Scatter_Check")