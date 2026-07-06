import numpy as np
from base_planner import BasePlanner

class GWOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_wolves=120, max_iter=150, num_waypoints=16): 
        """ 
        继承自 BasePlanner
        【重大修正】：默认控制点数 num_waypoints 必须从 6 调大到 16, 因为地图有 11 个打卡点, 6 个点根本装不下！
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

    def _cubic_chaotic_map(self, size):
        """ 
        Cubic 立方混沌映射生成器
        生成均匀分布在 [0, 1] 的混沌序列，供后续映射到地图真实尺寸
        """
        rho = 2.595  # 经典混沌系数
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
        
        # 【修改1：提取 3D 目标点 (取圆柱体的中心高度)】
        targets_3d = []
        for t in self.env.target_areas:
            z_mid = (t.get('z_min', 0.0) + t.get('z_max', 20.0)) / 2.0
            targets_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
        
        center_x = (self.env.x_bounds[0] + self.env.x_bounds[1]) / 2.0
        center_y = (self.env.y_bounds[0] + self.env.y_bounds[1]) / 2.0
        
        # 雷达排序依旧可以只看 XY 平面的极坐标投影
        targets_3d.sort(key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))

        for i in range(self.num_wolves):
            chaos_x = self._cubic_chaotic_map(self.num_waypoints)
            chaos_y = self._cubic_chaotic_map(self.num_waypoints)
            chaos_z = self._cubic_chaotic_map(self.num_waypoints)
            
            # 映射到真实的地图边界，Z 轴起步设为 5 米防止钻地
            rand_x = self.env.x_bounds[0] + chaos_x * (self.env.x_bounds[1] - self.env.x_bounds[0])
            rand_y = self.env.y_bounds[0] + chaos_y * (self.env.y_bounds[1] - self.env.y_bounds[0])
            rand_z = max(5.0, self.env.z_bounds[0]) + chaos_z * (self.env.z_bounds[1] - max(5.0, self.env.z_bounds[0]))
            
            # 拼装成 3D 点云
            raw_pts = np.column_stack((rand_x, rand_y, rand_z))

            # 给前 30% 的精英狼注入 3D 打卡点
            if i < int(self.num_wolves * 0.3):
                for j, tgt in enumerate(targets_3d):
                    if j < self.num_waypoints:
                        raw_pts[j] = tgt

            # 对这条狼所有的点进行 XY 极坐标雷达排序
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
            
            # 基于 Alpha 狼精英引导的局部变异 (Elite-guided Mutation)
            mutation_rate = 0.2  # 选取 20% 的灰狼进行变异，充当敢死队去探路
            # 变异步长随迭代衰减：前期大范围扰动找缺口(20%边界跨度)，后期小范围平滑微调
            mutation_scale = 0.2 * (1.0 - l / self.max_iter) 

            is_emergency = getattr(self, 'emergency_escape', False)
            is_press_down = getattr(self, 'press_down', False) # 新增：读取老中医的下压指令
            
            if is_emergency:
                mutation_rate = 0.5  # 卡死时，扩大敢死队规模到 50%
                mutation_scale = 0.5 # 极度放大水平面的乱窜范围（实现“试着拐弯”）
            elif is_press_down:
                mutation_rate = 0.3  # 抓 30% 的狼去试探低空
                mutation_scale = 0.1 # 水平方向不大动，仅仅试探高度，步长调小
            
            # 随机生成掩码，决定哪些狼变异
            do_mutation = np.random.rand(self.num_wolves) < mutation_rate
            
            # 【保护精英】：由于目前狼群未严格按分数排序，简单保护前2只不参与变异
            do_mutation[0] = False 
            do_mutation[1] = False 
            
            # 生成围绕 Alpha 狼的高斯扰动噪声
            noise = np.random.randn(self.num_wolves, self.dim) * (self.ub - self.lb) * mutation_scale

            # 根据指令注入 Z 轴物理推力
            if is_emergency:
                # 遍历所有控制点的 Z 轴 (数组结构: x1,y1,z1, x2,y2,z2... Z的索引是 2, 5, 8...)
                for d in range(2, self.dim, 3): 
                    noise[:, d] += 15.0 # 强制向上方拉升 15 米！
            elif is_press_down:
                for d in range(2, self.dim, 3): 
                    noise[:, d] -= 3.0  # 强制向下试探 3 米！寻找更省电的低空缝隙！
                    
            mutated_pos = self.alpha_pos + noise
            
            # 应用变异：只更新被选中的那 20% 的狼，其余 80% 依然遵循原本的 GWO 包围机制
            self.positions[do_mutation] = mutated_pos[do_mutation]
            # ==========================================

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

    # 1. 独立实例化评价器和算法
    evaluator = PathEvaluator()
    planner = GWOPlanner(evaluator=evaluator)
    
    # 2. 执行优化
    best_path, history = planner.optimize()
    
    # 【新增功能】：上帝视角 —— 观察所有灰狼的最终死活位置
    print("\n 正在绘制狼群最终分布 3D 散点图...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画出建筑物和目标圈
    evaluator.env.draw_environment_3d(ax)
    
    # 将 120 只狼的最终坐标全部画出来！
    # 设置透明度 alpha=0.15，如果很多狼死死挤在一起，那个地方颜色就会变得极其深（发黑）
    for i in range(planner.num_wolves):
        # 把狼的一维基因数组还原成 (16, 3) 的 3D 控制点
        wolf_pts = planner.positions[i].reshape((planner.num_waypoints, 3))
        ax.scatter(wolf_pts[:, 0], wolf_pts[:, 1], wolf_pts[:, 2], 
                   c='blue', alpha=0.15, s=15, marker='o')
                   
    # 最后，用高亮粉色画出 Alpha 狼（历史最佳）跑出的丝滑曲线
    smooth_path = evaluator.generate_bspline_path(best_path, num_points=100)
    ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2], 
            color='#FF007F', linewidth=4, label='Alpha Wolf (Best Path)')
            
    ax.set_title("GWO Final Population Distribution (Checking Stagnation)", fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 原本的收敛曲线图依然保留
    planner.plot_result(best_path, history, algo_name="GWO_Scatter_Check")