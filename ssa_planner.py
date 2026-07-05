import numpy as np
from base_planner import BasePlanner

class SSAPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_sparrows=100, max_iter=200, num_waypoints=15, disturb_ratio=0.15):
        """
        3D 麻雀搜索算法路径规划器
        :param evaluator: 路径评价器实例
        :param num_sparrows: 麻雀种群数量
        :param max_iter: 最大迭代次数
        :param num_waypoints: 中间控制点数量
        :param disturb_ratio: 初始化时随机扰动幅度相对于环境尺寸的比例
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_sparrows = num_sparrows
        self.disturb_ratio = disturb_ratio   # 扰动比例
        
        # SSA 核心参数
        self.PD = 0.2   # 发现者比例
        self.SD = 0.1   # 侦察者比例
        self.ST = 0.8   # 安全阈值
        
        self.num_producers = int(self.num_sparrows * self.PD)
        self.num_scouts = int(self.num_sparrows * self.SD)
        
        # 初始化麻雀位置与适应度
        self.sparrows = self._initialize_sparrows()
        self.fitness = np.full(num_sparrows, np.inf)
        
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

    def _initialize_sparrows(self):
        sparrows = np.zeros((self.num_sparrows, self.dim))
        
        # ============================================
        # 【新增神技】构造一个“完美路径”作为第 0 号麻雀
        # ============================================
        start = self.env.start_point
        end = self.env.end_point
        direction_vec = end - start
        
        # 1. 收集所有目标点的中心坐标（并转成 3D）
        targets_3d = []
        for t in self.env.target_areas:
            center = t['center']
            # 如果 target 只有 2D，补 Z，取 z_min 和 z_max 的中值
            z_mid = (t.get('z_min', 0) + t.get('z_max', 10)) / 2.0
            targets_3d.append(np.array([center[0], center[1], z_mid]))
        
        # ============================================================
        # 【核心改动】放弃投影排序！改用 最近邻（贪心TSP）排序
        # 这样生成的路径是相邻的，没有大幅折返，B样条会完美跟随
        # ============================================================
        unvisited = targets_3d.copy()
        sorted_targets = []
        current = start
        while unvisited:
            # 计算到所有未访问点的距离
            distances = [np.linalg.norm(p - current) for p in unvisited]
            # 找最近的
            idx = np.argmin(distances)
            nearest = unvisited.pop(idx)
            sorted_targets.append(nearest)
            current = nearest  # 更新当前位置为刚找到的点
        
        # 3. 构造控制点列表：起点 + 排序后的目标点 + 终点
        waypoints_list = [start] + sorted_targets + [end]
        
        # ============================================================
        # 4. 【终极修复】补全控制点，但绝不丢弃任何目标！
        #    策略：如果目标点少于 num_waypoints，就在“距离最远”的两个相邻目标之间插入中点，
        #    直到数量达标。这样所有原始目标都会原封不动地保留在列表中。
        # ============================================================
        num_targets = len(sorted_targets)
        
        # 如果目标点数量大于控制点数量，说明 num_waypoints 设小了，直接报错提醒
        if num_targets > self.num_waypoints:
            print(f"警告: 目标点({num_targets})多于控制点({self.num_waypoints})，路径必然漏检！请增加 num_waypoints！")
            # 临时处理：取前 num_waypoints 个（但会导致漏检，强烈建议增大 num_waypoints）
            control_pts = sorted_targets[:self.num_waypoints]
        else:
            # 正常情况：目标点少于等于控制点，开始补点
            control_pts = list(sorted_targets)  # 复制所有目标点
            
            while len(control_pts) < self.num_waypoints:
                # 找到当前列表中距离最远的相邻点对（欧氏距离）
                max_gap = -1.0
                max_idx = 0
                for i in range(len(control_pts) - 1):
                    gap = np.linalg.norm(control_pts[i+1] - control_pts[i])
                    if gap > max_gap:
                        max_gap = gap
                        max_idx = i
                
                # 在最大间隔的正中间插入一个新点
                mid_point = (control_pts[max_idx] + control_pts[max_idx+1]) / 2.0
                control_pts.insert(max_idx + 1, mid_point)
            
            # 循环结束，此时 len(control_pts) 必然等于 num_waypoints
            # 且所有原始目标点 (P1~P11) 都完美保留在列表中！
        
        # 5. 展平作为第 0 号麻雀
        super_sparrow = np.clip(np.array(control_pts).flatten(), self.lb, self.ub)
        sparrows[0] = super_sparrow

        # 计算每个维度可用的扰动范围（环境边界大小的一部分）
        x_range = (self.env.x_bounds[1] - self.env.x_bounds[0]) * self.disturb_ratio
        y_range = (self.env.y_bounds[1] - self.env.y_bounds[0]) * self.disturb_ratio
        z_range = (self.env.z_bounds[1] - self.env.z_bounds[0]) * self.disturb_ratio

        # 在生成随机粒子之前，先准备好完美控制点
        perfect_control_pts = np.array(control_pts).flatten()

        # 6. 剩下的麻雀继续用原来的随机方法（保持多样性）
        for i in range(self.num_sparrows):
            if i < int(self.num_sparrows * 0.2):   # 前 20% 粒子基于完美路径加小扰动
                # 在完美路径上添加小噪声 (幅值 5~10 米)
                noise = np.random.uniform(-5.0, 5.0, self.dim)
                sparrow = perfect_control_pts + noise
                sparrows[i] = np.clip(sparrow, self.lb, self.ub)
            else:
                # 1. 在起终点之间均匀生成原始控制点（3D）
                t = np.linspace(0, 1, self.num_waypoints + 2)[1:-1]  # 不包括端点
                base_x = start[0] + t * direction_vec[0]
                base_y = start[1] + t * direction_vec[1]
                base_z = start[2] + t * direction_vec[2]
                raw_waypoints = np.column_stack((base_x, base_y, base_z))
                
                # 2. 添加各维度独立随机扰动（均匀分布）
                noise_x = np.random.uniform(-x_range, x_range, self.num_waypoints)
                noise_y = np.random.uniform(-y_range, y_range, self.num_waypoints)
                noise_z = np.random.uniform(-z_range, z_range, self.num_waypoints)
                noisy = raw_waypoints + np.column_stack((noise_x, noise_y, noise_z))
                
                # 3. 【防打结】将控制点向主方向投影并排序
                projections = np.dot(noisy - start, direction_vec)
                sorted_waypoints = noisy[np.argsort(projections)]
                
                # 4. 展平并约束在边界内
                sparrows[i] = np.clip(sorted_waypoints.flatten(), self.lb, self.ub)
            
        return sparrows

    def optimize(self):
        """ 运行 SSA 主循环（3D 版本） """
        print("开始 3D SSA 麻雀搜索算法路径规划...")
        
        # 1. 初始适应度评估
        for i in range(self.num_sparrows):
            full_path = self._decode_path(self.sparrows[i])
            self.fitness[i], _, _ = self.evaluator.evaluate_pso_particle(full_path)
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
            
            # (1) 发现者更新
            R2 = np.random.rand()
            for i in range(self.num_producers):
                idx = sort_indices[i]
                if R2 < self.ST:
                    alpha = np.random.rand()
                    step = np.random.randn(self.dim) * 40.0   # 步长可根据环境调整
                    new_sparrows[idx] = self.sparrows[idx] + step * np.exp(-(iteration + 1) / (alpha * self.max_iter + 1e-8))
                else:
                    new_sparrows[idx] = self.sparrows[idx] + np.random.randn(self.dim) * 2.0
            
            # (2) 加入者更新
            for i in range(self.num_producers, self.num_sparrows):
                idx = sort_indices[i]
                if i > self.num_sparrows / 2:
                    # 饥饿麻雀随机重生
                    new_sparrows[idx] = np.random.uniform(self.lb, self.ub, self.dim)
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
            
            # (4) 越界处理与适应度评估
            for i in range(self.num_sparrows):
                new_sparrows[i] = np.clip(new_sparrows[i], self.lb, self.ub)
                full_path = self._decode_path(new_sparrows[i])
                score, _, _ = self.evaluator.evaluate_pso_particle(full_path)
                
                self.sparrows[i] = new_sparrows[i]
                self.fitness[i] = score
                
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos = np.copy(new_sparrows[i])
            
            self.convergence_curve.append(self.gbest_score)
            
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 全局最优得分: {self.gbest_score:.2f}")
        
        return self._decode_path(self.gbest_pos), self.convergence_curve


# ===================== 本地单文件测试 =====================
if __name__ == "__main__":
    # 注意：需要先定义好环境（PathEvaluator 内部已包含3D环境）
    planner = SSAPlanner(disturb_ratio=0.5, num_sparrows=80, max_iter=150, num_waypoints=18)
    best_path, history = planner.optimize()
    planner.evaluator.debug_target_coverage(best_path)
    planner.plot_result(best_path, history, algo_name="SSA-3D")