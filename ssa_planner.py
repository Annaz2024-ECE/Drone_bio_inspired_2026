import numpy as np
from base_planner import BasePlanner

class SSAPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_sparrows=100, max_iter=200, num_waypoints=30, disturb_ratio=0.15):
        """
        3D 麻雀搜索算法路径规划器 (已深度重构：加入贪婪选择与三点微簇锚固)
        """
        # 【核心修复1】为应对三点锚固，强制提供充足的控制点 (11个目标*3 = 33，45足够冗余)
        num_waypoints = max(num_waypoints, 45) 
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_sparrows = num_sparrows
        self.disturb_ratio = disturb_ratio
        
        # SSA 核心参数
        self.PD = 0.2   # 发现者比例
        self.SD = 0.1   # 侦察者比例
        self.ST = 0.8   # 安全阈值
        
        self.num_producers = int(self.num_sparrows * self.PD)
        self.num_scouts = int(self.num_sparrows * self.SD)
        
        # Z轴噪声缩放面具：缩减Z轴暴走
        self.z_mask = np.tile([1.0, 1.0, 0.2], self.num_waypoints)
        
        # 初始化麻雀位置与适应度
        self.sparrows = self._initialize_sparrows()
        self.fitness = np.full(num_sparrows, np.inf)
        
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = np.inf

    def _decode_path(self, position):
        """
        [子类重写] 植入贪心最近邻排序 (Nearest Neighbor Sort)
        在每次评价前，强制理顺被打乱的麻雀基因，彻底消除 3D 航线打结与绕圈现象。
        """
        # 1. 将 1D 基因还原为 3D 坐标点阵
        waypoints = position.reshape((self.num_waypoints, 3))
        
        # 2. 贪心最近邻排序核心逻辑
        sorted_waypoints = []
        current_point = self.env.start_point 
        remaining_indices = list(range(self.num_waypoints))
        
        while remaining_indices:
            best_idx = -1
            min_dist = float('inf')
            
            # 遍历所有还没被连线的点，找离当前位置最近的
            for idx in remaining_indices:
                dist = np.linalg.norm(waypoints[idx] - current_point)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
                    
            # 把找到的最近点加入有序列表，并将“当前位置”推进到该点
            sorted_waypoints.append(waypoints[best_idx])
            current_point = waypoints[best_idx]
            remaining_indices.remove(best_idx)
            
        # 3. 拼接起终点返回
        sorted_waypoints = np.array(sorted_waypoints)
        full_path = np.vstack([self.env.start_point, sorted_waypoints, self.env.end_point])
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

    def _initialize_sparrows(self):
        sparrows = np.zeros((self.num_sparrows, self.dim))
        
        # ==========================================
        # 1. 获取 TSP + 拱门 的“无敌拓扑骨架” (保证初始不撞墙)
        # ==========================================
        super_skeleton = self._generate_heuristic_skeleton()
        
        # ==========================================
        # 2. 精英部队 (前 20%)：围绕无敌骨架微调，保有指路明灯
        # ==========================================
        elite_count = int(self.num_sparrows * 0.2)
        sparrows[0] = super_skeleton # 第0号直接封神，保底 10 万分起步！
        
        for i in range(1, elite_count):
            noise = np.random.normal(0, 3.0, self.dim) * self.z_mask
            sparrows[i] = np.clip(super_skeleton + noise, self.lb, self.ub)

        # ==========================================
        # 3. 混沌散勇 (后 80%)：全图撒网，制造队友需要的“破壁”奇迹
        # ==========================================
        center_x = (self.env.x_bounds[0] + self.env.x_bounds[1]) / 2.0
        center_y = (self.env.y_bounds[0] + self.env.y_bounds[1]) / 2.0

        for i in range(elite_count, self.num_sparrows):
            chaos_x = self._cubic_chaotic_map(self.num_waypoints)
            chaos_y = self._cubic_chaotic_map(self.num_waypoints)
            chaos_z = self._cubic_chaotic_map(self.num_waypoints)
            
            rand_x = self.env.x_bounds[0] + chaos_x * (self.env.x_bounds[1] - self.env.x_bounds[0])
            rand_y = self.env.y_bounds[0] + chaos_y * (self.env.y_bounds[1] - self.env.y_bounds[0])
            rand_z = max(5.0, self.env.z_bounds[0]) + chaos_z * (self.env.z_bounds[1] - max(5.0, self.env.z_bounds[0]))
            
            raw_pts = np.column_stack((rand_x, rand_y, rand_z))
            
            # 极坐标理顺
            angles = np.arctan2(raw_pts[:, 1] - center_y, raw_pts[:, 0] - center_x)
            sorted_pts = raw_pts[np.argsort(angles)]
            
            sparrows[i] = np.clip(sorted_pts.flatten(), self.lb, self.ub)

        return sparrows

    def optimize(self):
        print("开始 3D SSA 麻雀搜索算法路径规划 (启用贪婪保护与三点锚固)...")
        
        for i in range(self.num_sparrows):
            full_path = self._decode_path(self.sparrows[i])
            self.fitness[i], _, _ = self.evaluator.evaluate_pso_particle(full_path)
            if self.fitness[i] < self.historical_best_score:
                self.historical_best_score = self.fitness[i]
                self.historical_best_pos = np.copy(self.sparrows[i])
        
        for iteration in range(self.max_iter):
            sort_indices = np.argsort(self.fitness)
            best_pos_current = np.copy(self.sparrows[sort_indices[0]])
            worst_pos_current = np.copy(self.sparrows[sort_indices[-1]])
            worst_fit_current = self.fitness[sort_indices[-1]]
            
            new_sparrows = np.copy(self.sparrows)
            
            # (1) 发现者更新
            R2 = np.random.rand()
            for i in range(self.num_producers):
                idx = sort_indices[i]
                if R2 < self.ST:
                    alpha = np.random.rand()
                    step = np.random.randn(self.dim) * 8.0 * self.z_mask   
                    new_sparrows[idx] = self.sparrows[idx] + step * np.exp(-(iteration + 1) / (alpha * self.max_iter + 1e-8))
                else:
                    new_sparrows[idx] = self.sparrows[idx] + np.random.randn(self.dim) * 2.0 * self.z_mask
            
            # (2) 加入者更新
            for i in range(self.num_producers, self.num_sparrows):
                idx = sort_indices[i]
                if i > self.num_sparrows / 2:
                    new_sparrows[idx] = best_pos_current + np.random.randn(self.dim) * 5.0 * self.z_mask
                else:
                    A = np.random.choice([-1, 1], size=self.dim)
                    new_sparrows[idx] = best_pos_current + np.abs(self.sparrows[idx] - best_pos_current) * (A / 2.0)
            
            # (3) 侦察者更新
            scout_indices = np.random.choice(self.num_sparrows, self.num_scouts, replace=False)
            for idx in scout_indices:
                if self.fitness[idx] > self.fitness[sort_indices[0]]:
                    new_sparrows[idx] = best_pos_current + np.random.randn(self.dim) * np.abs(self.sparrows[idx] - best_pos_current)
                else:
                    new_sparrows[idx] = self.sparrows[idx] + np.random.uniform(-1, 1) * (np.abs(self.sparrows[idx] - worst_pos_current) / (self.fitness[idx] - worst_fit_current + 1e-8))
            
            # ==========================================
            # 【新增：破壁者机制】抓取 20% 麻雀强行在最优解附近引爆
            # ==========================================
            mutate_count = int(self.num_sparrows * 0.2)
            # 随机挑选 20% 的倒霉蛋，但绝不碰当前表现最好的那只
            mutate_indices = np.random.choice(self.num_sparrows, mutate_count, replace=False)
            if sort_indices[0] in mutate_indices:
                mutate_indices = np.delete(mutate_indices, np.where(mutate_indices == sort_indices[0]))
                
            for idx in mutate_indices:
                # 以最佳麻雀为中心，产生 3.0 米的高斯噪声球，并压制 Z 轴暴走
                noise = np.random.randn(self.dim) * 3.0 * self.z_mask
                new_sparrows[idx] = best_pos_current + noise
            # ==========================================

            # ==========================================
            # >>> 【完美缝隙】：触发基类的通用物理引擎 (拉普拉斯平滑) <<<
            # 桥接黑科技：把局部的 new_sparrows 伪装成基类认识的 self.positions
            self.positions = new_sparrows
            
            # 执行老中医下发的全局平滑物理指令
            self.execute_universal_physics_directives()
            
            # 技能释放完毕后，把平滑后的结果重新拿回给 new_sparrows
            new_sparrows = self.positions

            # 烧毁临时护照
            del self.positions
            # ==========================================

            # (4) 越界处理与【贪婪适应度评估】
            for i in range(self.num_sparrows):
                new_sparrows[i] = np.clip(new_sparrows[i], self.lb, self.ub)
                full_path = self._decode_path(new_sparrows[i])
                score, _, _ = self.evaluator.evaluate_pso_particle(full_path)
                
                # 【核心修复3：贪婪选择 (Greedy Selection)】
                if score <= self.fitness[i]:
                    self.sparrows[i] = new_sparrows[i]
                    self.fitness[i] = score
                
                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = np.copy(new_sparrows[i])

            # ==========================================
            # 接收老中医的四大通用物理指令 (Universal API)
            # ==========================================
            is_radar = getattr(self, 'radar_guidance', False)
            is_emergency = getattr(self, 'emergency_escape', False)
            is_lift_up = getattr(self, 'lift_up', False)
            is_press_down = getattr(self, 'press_down', False)

            # 只要触发了全局动作，就启动底层基因强行干预（无视贪婪法则）
            if is_radar or is_emergency or is_lift_up or is_press_down:
                
                mutation_rate = 0.1
                if is_emergency: mutation_rate = 0.5
                elif is_lift_up: mutation_rate = 0.4
                elif is_press_down: mutation_rate = 0.3
                
                do_mutation = np.random.rand(self.num_sparrows) < mutation_rate
                
                # 绝对保护：保留历史最优的那只麻雀，留下革命火种
                best_idx = np.argmin(self.fitness)
                do_mutation[best_idx] = False 
                
                # 1. 雷达空投逻辑
                if is_radar:
                    targets_3d = []
                    for t in self.env.target_areas:
                        z_mid = (t.get('z_min', 0.0) + t.get('z_max', 10.0)) / 2.0
                        targets_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
                    
                    for i in range(self.num_sparrows):
                        if do_mutation[i]:
                            new_pos = np.zeros((self.num_waypoints, 3))
                            if len(targets_3d) > 0:
                                for j in range(self.num_waypoints):
                                    new_pos[j] = targets_3d[j % len(targets_3d)]
                            
                            noise = np.random.randn(self.num_waypoints, 3) * 2.0
                            self.sparrows[i] = np.clip((new_pos + noise).flatten(), self.lb, self.ub)
                            
                            # 【核心斩断贪婪陷阱】：无视分数变差，强行更新肉体记忆
                            full_path = self._decode_path(self.sparrows[i])
                            self.fitness[i], _, _ = self.evaluator.evaluate_pso_particle(full_path)
                
                # 2. 物理推力逻辑 (Z轴强制位移)
                else:
                    noise = np.zeros((self.num_sparrows, self.dim))
                    if is_emergency:
                        for d in range(2, self.dim, 3): noise[:, d] = 15.0 
                    elif is_lift_up:
                        for d in range(2, self.dim, 3): noise[:, d] = 8.0  
                    elif is_press_down:
                        for d in range(2, self.dim, 3): noise[:, d] = -3.0 
                    
                    for i in range(self.num_sparrows):
                        if do_mutation[i]:
                            self.sparrows[i] += noise[i]
                            self.sparrows[i] = np.clip(self.sparrows[i], self.lb, self.ub)
                            
                            # 【核心斩断贪婪陷阱】：无视分数变差，强行更新肉体记忆
                            full_path = self._decode_path(self.sparrows[i])
                            self.fitness[i], _, _ = self.evaluator.evaluate_pso_particle(full_path)

                # 3. 动作结束后，二次检查是否诞生了新的历史最优
                for i in range(self.num_sparrows):
                    if self.fitness[i] < self.historical_best_score:
                        self.historical_best_score = self.fitness[i]
                        self.historical_best_pos = np.copy(self.sparrows[i])
                        
                # ==========================================
                # 【极其关键】：阅后即焚！
                # 执行完一次老中医的“冲量”干预后，必须立刻销毁指令
                # 否则后续代数麻雀会被无限次拔高或压低！
                # ==========================================
                self.radar_guidance = False
                self.emergency_escape = False
                self.lift_up = False
                self.press_down = False
            # ==========================================
            
            self.convergence_curve.append(self.historical_best_score)
            
            if (iteration + 1) % 20 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 全局最优得分: {self.historical_best_score:,.2f}")
        
        return self._decode_path(self.historical_best_pos), self.convergence_curve
        
if __name__ == "__main__":
    planner = SSAPlanner(disturb_ratio=0.5, num_sparrows=100, max_iter=150, num_waypoints=65)
    best_path, history = planner.optimize()
    #planner.evaluator.debug_target_coverage(best_path)

    # ==========================================
    # 🔥 【新增】：把 SSA 跑出来的 3D 路线存到本地
    # ==========================================
    np.save('saved_best_path.npy', best_path)
    print("✅ 3D路线坐标已安全存档至 'saved_best_path.npy'！")
    
    planner.plot_result(best_path, history, algo_name="SSA-3D")