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

    def _initialize_sparrows(self):
        sparrows = np.zeros((self.num_sparrows, self.dim))
        
        start = self.env.start_point
        end = self.env.end_point
        
        # 1. 收集所有目标点的 3D 中心坐标
        targets_3d = []
        for t in self.env.target_areas:
            center = t['center']
            z_mid = (t.get('z_min', 0) + t.get('z_max', 10)) / 2.0
            targets_3d.append(np.array([center[0], center[1], z_mid]))
        
        # 2. 最近邻（贪心 TSP）排序
        unvisited = targets_3d.copy()
        sorted_targets = []
        current = start
        while unvisited:
            distances = [np.linalg.norm(p - current) for p in unvisited]
            idx = np.argmin(distances)
            nearest = unvisited.pop(idx)
            sorted_targets.append(nearest)
            current = nearest
        
        # 3. 【核心修复2：三点微簇锚固 (Triple-Cluster Anchoring)】
        # 无论 coordinator 怎么改变 bspline_num_points，0.6m 的贯穿微簇都能确保曲线死死咬住目标中心！
        control_pts = []
        for t in sorted_targets:
            control_pts.append(t + np.array([-0.3, -0.3, -0.1]))
            control_pts.append(t)
            control_pts.append(t + np.array([0.3, 0.3, 0.1]))
            
        # 4. 【拱门跨越】：补全控制点，并强制拉高避免穿模
        safe_arch_z = 8.0 
        
        while len(control_pts) < self.num_waypoints:
            max_gap = -1.0
            max_idx = 0
            temp_path = [start] + control_pts + [end]
            for i in range(len(temp_path) - 1):
                gap = np.linalg.norm(temp_path[i+1] - temp_path[i])
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
            
            mid_point = (temp_path[max_idx] + temp_path[max_idx+1]) / 2.0
            
            # 只有在长距离跨越时才拉起拱门
            if max_gap > 8.0:
                mid_point[2] = max(mid_point[2], safe_arch_z)
            
            if max_idx == 0:
                control_pts.insert(0, mid_point)
            elif max_idx == len(temp_path) - 1:
                control_pts.append(mid_point)
            else:
                control_pts.insert(max_idx, mid_point)
                
        control_pts = control_pts[:self.num_waypoints]
        perfect_control_pts = np.array(control_pts).flatten()
        
        # 第 0 号：完美路径
        sparrows[0] = np.clip(perfect_control_pts, self.lb, self.ub)

        # 5. 基于 TSP 骨架进行不同程度的变异
        for i in range(1, self.num_sparrows):
            if i < int(self.num_sparrows * 0.4):
                noise = np.random.normal(0, 2.0, self.dim) * self.z_mask
            elif i < int(self.num_sparrows * 0.8):
                noise = np.random.normal(0, 6.0, self.dim) * self.z_mask
            else:
                noise = np.random.normal(0, 12.0, self.dim) * self.z_mask
                
            sparrow = perfect_control_pts + noise
            sparrows[i] = np.clip(sparrow, self.lb, self.ub)
            
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
    planner = SSAPlanner(disturb_ratio=0.5, num_sparrows=80, max_iter=200, num_waypoints=30)
    best_path, history = planner.optimize()
    planner.evaluator.debug_target_coverage(best_path)
    planner.plot_result(best_path, history, algo_name="SSA-3D")