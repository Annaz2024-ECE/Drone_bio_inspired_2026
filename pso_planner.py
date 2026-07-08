import numpy as np
from base_planner import BasePlanner
import os

class PSOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_particles=100, max_iter=200, num_waypoints=30, disturb_ratio=0.15):
        """
        3D PSO 路径规划器 (已移植 SSA 的多目标锚固与高度约束技术)
        """
        # 【修复1】强制提升控制点数量以应对 3D 复杂度和双点锚固
        num_waypoints = max(num_waypoints, 30)
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_particles = num_particles
        self.disturb_ratio = disturb_ratio
        
        # PSO 核心参数
        self.w_max = 0.9  
        self.w_min = 0.4  
        self.c1 = 1.5 
        self.c2 = 1.5 
        self.v_max = 8.0 
        
        # 【新增】Z轴维度面具与速度限制
        self.z_mask = np.tile([1.0, 1.0, 0.2], self.num_waypoints)
        self.v_max_arr = self.v_max * self.z_mask  # 限制垂直最大速度，防止上下暴走
        
        # 初始化粒子位置和速度
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((self.num_particles, self.dim))
        
        # 记录个体最优和全局最优
        self.pbest_pos = np.copy(self.particles)
        self.pbest_scores = np.full(self.num_particles, np.inf)
        
        # 【核心修改】：废除原有的 gbest，使用 historical_best 完美对接老中医，实现跨轮次记忆！
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = np.inf

    def _initialize_particles(self):
        """ 
        生成 3D 初始粒子群，引入“双点锚固”和“拱门飞跃”技术。
        """
        particles = np.zeros((self.num_particles, self.dim))
        
        start = self.env.start_point
        end = self.env.end_point
        
        # 1. 收集目标点
        targets_3d = []
        for t in self.env.target_areas:
            center = t['center']
            z_mid = (t.get('z_min', 0) + t.get('z_max', 10)) / 2.0
            targets_3d.append(np.array([center[0], center[1], z_mid]))
        
        # 2. 贪心最近邻 TSP 排序
        unvisited = targets_3d.copy()
        sorted_targets = []
        current = start
        while unvisited:
            distances = [np.linalg.norm(p - current) for p in unvisited]
            idx = np.argmin(distances)
            nearest = unvisited.pop(idx)
            sorted_targets.append(nearest)
            current = nearest
        
        # 3. 【核心修复：双点锚固】对抗 B-Spline 切角漏检
        control_pts = []
        for t in sorted_targets:
            control_pts.append(t)
            control_pts.append(t + np.array([0.0, 0.0, 0.15])) # 微小偏移躲避去重审查
            
        # 4. 【核心修复：拱门跨越】
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
            mid_point[2] = max(mid_point[2], safe_arch_z)  # 强制拉高
            
            if max_idx == 0:
                control_pts.insert(0, mid_point)
            elif max_idx == len(temp_path) - 1:
                control_pts.append(mid_point)
            else:
                control_pts.insert(max_idx, mid_point)
                
        control_pts = control_pts[:self.num_waypoints]
        super_particle = np.clip(np.array(control_pts).flatten(), self.lb, self.ub)
        
        # 5. 填充粒子群：使用 z_mask 限制垂直扰动幅度
        particles[0] = super_particle
        
        for i in range(1, self.num_particles):
            if i < int(self.num_particles * 0.4):
                noise = np.random.normal(0, 2.0, self.dim) * self.z_mask
            elif i < int(self.num_particles * 0.8):
                noise = np.random.normal(0, 6.0, self.dim) * self.z_mask
            else:
                noise = np.random.normal(0, 12.0, self.dim) * self.z_mask
                
            particles[i] = np.clip(super_particle + noise, self.lb, self.ub)
        
        return particles

    def optimize(self):
        print("开始 3D PSO 粒子群算法路径规划 (带多目标锚固、限速与历史最优保护)...")
        
        # 初始评估 (适应调参大脑的多轮重入，强制刷新 pbest 的真实得分)
        for i in range(self.num_particles):
            # 如果不是第一轮，老中医可能修改了规则，过去的 pbest_score 已作废，必须按新规则重算！
            if self.pbest_scores[i] != np.inf:
                true_pbest_score, _, _ = self.evaluator.evaluate_pso_particle(self._decode_path(self.pbest_pos[i]))
                self.pbest_scores[i] = true_pbest_score
                
            full_path = self._decode_path(self.particles[i])
            score, _, _ = self.evaluator.evaluate_pso_particle(full_path)
            
            if score < self.pbest_scores[i]:
                self.pbest_scores[i] = score
                self.pbest_pos[i] = np.copy(self.particles[i])
                
            # 记录历史最优
            if score < self.historical_best_score:
                self.historical_best_score = score
                self.historical_best_pos = np.copy(self.particles[i])
                
        for iteration in range(self.max_iter):
            w_current = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                # 【核心修改】：社会向心力追随 historical_best
                social = self.c2 * r2 * (self.historical_best_pos - self.particles[i])
                
                self.velocities[i] = w_current * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max_arr, self.v_max_arr)
                
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)
                
                full_path = self._decode_path(self.particles[i])
                score, _, _ = self.evaluator.evaluate_pso_particle(full_path)
                
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_pos[i] = np.copy(self.particles[i])
                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = np.copy(self.particles[i])
                    
            self.convergence_curve.append(self.historical_best_score)
            
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 历史最优得分: {self.historical_best_score:,.2f}")
                
        return self._decode_path(self.historical_best_pos), self.convergence_curve

# ===================== 修改后的主函数（10次循环保存） =====================
if __name__ == "__main__":
    save_dir = "PSO_3D"
    os.makedirs(save_dir, exist_ok=True)
    
    num_runs = 1
    all_final_scores = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*20} 第 {run_idx+1}/{num_runs} 次运行 {'='*20}")
        # 注意这里的 num_waypoints 被强制设定成了 30 以上来适应复杂的 3D 拐角
        planner = PSOPlanner(num_particles=100, max_iter=150, num_waypoints=16)
        best_path, history = planner.optimize()
        
        #planner.evaluator.debug_target_coverage(best_path)
        planner.plot_result(best_path, history, algo_name="PSO-3D", run_idx=run_idx, save_dir=save_dir)
        
        final_score = history[-1] if history else None
        all_final_scores.append(final_score)
        with open(os.path.join(save_dir, f"run_{run_idx:02d}_score.txt"), 'w') as f:
            f.write(f"Final score: {final_score:.2f}\n")
            
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