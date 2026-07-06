import numpy as np
from base_planner import BasePlanner
import os

class PSOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_particles=100, max_iter=200, num_waypoints=15, disturb_ratio=0.15):
        """
        3D PSO 路径规划器
        :param evaluator: 路径评价器
        :param num_particles: 粒子数量
        :param max_iter: 最大迭代次数
        :param num_waypoints: 中间控制点数量
        :param disturb_ratio: 初始化扰动幅度相对于环境尺寸的比例
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_particles = num_particles
        self.disturb_ratio = disturb_ratio
        
        # PSO 核心参数
        self.w_max = 0.9  
        self.w_min = 0.4  
        self.c1 = 1.5 
        self.c2 = 1.5 
        self.v_max = 8.0 
        
        # 初始化粒子位置和速度
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((self.num_particles, self.dim))
        
        # 记录个体最优和全局最优
        self.pbest_pos = np.copy(self.particles)
        self.pbest_scores = np.full(self.num_particles, np.inf)
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

    def _initialize_particles(self):
        """ 
        生成 3D 初始粒子群，并注入一个“超级粒子”精确踩过所有目标点，
        其余粒子随机生成（保持多样性）。
        """
        particles = np.zeros((self.num_particles, self.dim))
        
        start = self.env.start_point
        end = self.env.end_point
        direction_vec = end - start
        direction_norm = np.linalg.norm(direction_vec)
        if direction_norm == 0:
            direction_vec = np.array([1.0, 0.0, 0.0])
        
        # ============================================================
        # 1. 收集目标点，并用“最近邻（贪心TSP）”排序（避免大折返）
        # ============================================================
        targets_3d = []
        for t in self.env.target_areas:
            center = t['center']
            z_mid = (t.get('z_min', 0) + t.get('z_max', 10)) / 2.0
            targets_3d.append(np.array([center[0], center[1], z_mid]))
        
        # 贪心最近邻排序（从起点开始）
        unvisited = targets_3d.copy()
        sorted_targets = []
        current = start
        while unvisited:
            distances = [np.linalg.norm(p - current) for p in unvisited]
            idx = np.argmin(distances)
            nearest = unvisited.pop(idx)
            sorted_targets.append(nearest)
            current = nearest
        
        # ============================================================
        # 2. 构造控制点列表：起点 + 排序后的目标 + 终点
        # ============================================================
        waypoints_list = [start] + sorted_targets + [end]
        num_targets = len(sorted_targets)
        
        # 3. 补全控制点以达到 num_waypoints
        if num_targets > self.num_waypoints:
            print(f"⚠️ 警告: 目标点({num_targets})多于控制点({self.num_waypoints})，路径可能漏检！建议增大 num_waypoints。")
            # 只能取前 num_waypoints 个
            control_pts = sorted_targets[:self.num_waypoints]
        else:
            control_pts = list(sorted_targets)  # 复制所有目标点
            while len(control_pts) < self.num_waypoints:
                # 在距离最远的相邻点对之间插入中点
                max_gap = -1.0
                max_idx = 0
                for i in range(len(control_pts) - 1):
                    gap = np.linalg.norm(control_pts[i+1] - control_pts[i])
                    if gap > max_gap:
                        max_gap = gap
                        max_idx = i
                mid_point = (control_pts[max_idx] + control_pts[max_idx+1]) / 2.0
                control_pts.insert(max_idx + 1, mid_point)
            # 确保长度正好等于 num_waypoints（可能因循环多出一个，但不会，我们每次加1）
            # 但保险起见，截断
            control_pts = control_pts[:self.num_waypoints]
        
        # 展平作为“超级粒子”
        super_particle = np.clip(np.array(control_pts).flatten(), self.lb, self.ub)
        
        # ============================================================
        # 4. 填充粒子群：第一个粒子为超级粒子，其余随机生成
        # ============================================================
        particles[0] = super_particle
        
        # 随机生成的范围（disturb_ratio 控制）
        x_range = (self.env.x_bounds[1] - self.env.x_bounds[0]) * self.disturb_ratio
        y_range = (self.env.y_bounds[1] - self.env.y_bounds[0]) * self.disturb_ratio
        z_range = (self.env.z_bounds[1] - self.env.z_bounds[0]) * self.disturb_ratio
        
        for i in range(1, self.num_particles):
            # 在起点-终点连线上均匀取点
            t = np.linspace(0, 1, self.num_waypoints + 2)[1:-1]
            base_x = start[0] + t * direction_vec[0]
            base_y = start[1] + t * direction_vec[1]
            base_z = start[2] + t * direction_vec[2]
            raw_waypoints = np.column_stack((base_x, base_y, base_z))
            
            # 加扰动
            noise_x = np.random.uniform(-x_range, x_range, self.num_waypoints)
            noise_y = np.random.uniform(-y_range, y_range, self.num_waypoints)
            noise_z = np.random.uniform(-z_range, z_range, self.num_waypoints)
            noisy = raw_waypoints + np.column_stack((noise_x, noise_y, noise_z))
            
            # 投影排序（防打结）
            projections = np.dot(noisy - start, direction_vec)
            sorted_waypoints = noisy[np.argsort(projections)]
            particles[i] = np.clip(sorted_waypoints.flatten(), self.lb, self.ub)
        
        return particles

    def optimize(self):
        """ PSO 主循环（与 2D 完全一致，因所有操作基于 self.dim） """
        print("开始 3D PSO 粒子群算法路径规划...")
        
        # 初始评估
        for i in range(self.num_particles):
            full_path = self._decode_path(self.particles[i])
            score, _, _ = self.evaluator.evaluate_pso_particle(full_path)
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_pos = np.copy(self.particles[i])
                
        for iteration in range(self.max_iter):
            w_current = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest_pos - self.particles[i])
                self.velocities[i] = w_current * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)
                
                full_path = self._decode_path(self.particles[i])
                score, _, _ = self.evaluator.evaluate_pso_particle(full_path)
                
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_pos[i] = np.copy(self.particles[i])
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos = np.copy(self.particles[i])
                    
            self.convergence_curve.append(self.gbest_score)
            
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 全局最优得分: {self.gbest_score:.2f}")
                
        return self._decode_path(self.gbest_pos), self.convergence_curve


# ===================== 修改后的主函数（10次循环保存） =====================
if __name__ == "__main__":
    # 创建保存目录
    save_dir = "PSO_3D"
    os.makedirs(save_dir, exist_ok=True)
    
    num_runs = 10
    all_final_scores = []  # 用于记录每次的最终得分
    
    for run_idx in range(num_runs):
        print(f"\n{'='*20} 第 {run_idx+1}/{num_runs} 次运行 {'='*20}")
        planner = PSOPlanner(num_particles=100, max_iter=150, num_waypoints=15)
        best_path, history = planner.optimize()
        
        # 打印覆盖明细（debug_target_coverage 内部会打印）
        planner.evaluator.debug_target_coverage(best_path)
        
        # 保存图片（传入 save_dir 和 run_idx）
        planner.plot_result(best_path, history, algo_name="PSO-3D", run_idx=run_idx, save_dir=save_dir)
        
        # 将最终得分保存到文本文件
        final_score = history[-1] if history else None
        all_final_scores.append(final_score)
        with open(os.path.join(save_dir, f"run_{run_idx:02d}_score.txt"), 'w') as f:
            f.write(f"Final score: {final_score:.2f}\n")
            # 可以顺便记录路径点（可选）
            # np.savetxt(f, best_path, header="best_path (x,y,z)")
    
    # 输出所有运行的得分汇总
    print("\n" + "="*50)
    print("所有运行完成！结果保存在", save_dir)
    print("各次最终得分:")
    for i, score in enumerate(all_final_scores):
        print(f"  Run {i+1:02d}: {score:,.2f}")
    if all_final_scores:
        avg = np.mean(all_final_scores)
        std = np.std(all_final_scores)
        print(f"\n平均得分: {avg:,.2f}  (±{std:,.2f})")
    print("="*50)