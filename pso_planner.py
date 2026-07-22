import numpy as np
from base_planner import BasePlanner
import os

class PSOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_particles=100, max_iter=200, num_waypoints=30, disturb_ratio=0.15):
        """
        3D PSO 路径规划器 (已彻底剥离最近邻排序与锚固，采用最纯净的维度转化)
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
        
        # 【新增】Z轴维度面具与速度限制
        self.z_mask = np.tile([1.0, 1.0, 0.2], self.num_waypoints)
        self.v_max_arr = self.v_max * self.z_mask  
        
        # 初始化粒子位置和速度
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((self.num_particles, self.dim))
        
        # 记录个体最优和全局最优
        self.pbest_pos = np.copy(self.particles)
        self.pbest_scores = np.full(self.num_particles, np.inf)
        
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = np.inf

    def _initialize_particles(self):
        particles = np.zeros((self.num_particles, self.dim))
        super_skeleton = self._generate_basic_skeleton()
        
        particles[0] = super_skeleton
        
        for i in range(1, self.num_particles):
            if i < int(self.num_particles * 0.4):
                noise = np.random.normal(0, 2.0, self.dim) * self.z_mask
            elif i < int(self.num_particles * 0.8):
                noise = np.random.normal(0, 6.0, self.dim) * self.z_mask
            else:
                noise = np.random.normal(0, 12.0, self.dim) * self.z_mask
                
            particles[i] = np.clip(super_skeleton + noise, self.lb, self.ub)
        
        return particles

    def optimize(self):
        print("开始 3D PSO 粒子群算法路径规划 (带限速、历史保护与动态突变)...")
        
        for i in range(self.num_particles):
            if self.pbest_scores[i] != np.inf:
                # 【修复核心】：这里必须使用 _decode_path 将 pbest_pos[i] 转为 3D
                true_pbest_score, _, _ = self.evaluator.evaluate_particle(self._decode_path(self.pbest_pos[i]))
                self.pbest_scores[i] = true_pbest_score
                
            # 【修复核心】：这里必须使用 _decode_path 将 particles[i] 转为 3D
            full_path = self._decode_path(self.particles[i])
            score, _, _ = self.evaluator.evaluate_particle(full_path)
            
            if score < self.pbest_scores[i]:
                self.pbest_scores[i] = score
                self.pbest_pos[i] = np.copy(self.particles[i])
                
            if score < self.historical_best_score:
                self.historical_best_score = score
                self.historical_best_pos = np.copy(self.particles[i])

        for iteration in range(self.max_iter):
            w_current = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            # ==========================================
            # 阶段 1：公式位移与全局特工变量(disturb_ratio)生效
            # ==========================================
            mutate_count = int(self.num_particles * self.disturb_ratio) 
            mutate_indices = np.random.choice(self.num_particles, mutate_count, replace=False)
            best_idx = np.argmin(self.pbest_scores)
            
            for i in range(self.num_particles):
                if i in mutate_indices and i != best_idx:
                    decay = max(0.01, (1.0 - iteration / self.max_iter) ** 2)
                    noise = self._levy_step(self.dim) * 5.0 * decay * self.z_mask
                    self.particles[i] = self.historical_best_pos + noise
                    self.velocities[i] = 0.0 
                    self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)
                else:
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    
                    cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                    social = self.c2 * r2 * (self.historical_best_pos - self.particles[i])
                    
                    self.velocities[i] = w_current * self.velocities[i] + cognitive + social
                    self.velocities[i] = np.clip(self.velocities[i], -self.v_max_arr, self.v_max_arr)
                    
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

            self.positions = self.particles
            self.execute_universal_physics_directives()
            self.particles = self.positions
            del self.positions

            for i in range(self.num_particles):
                # 【修复核心】：评估时严格调用 _decode_path
                full_path = self._decode_path(self.particles[i])
                score, _, _ = self.evaluator.evaluate_particle(full_path)
                
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_pos[i] = np.copy(self.particles[i])
                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = np.copy(self.particles[i])

            is_radar = getattr(self, 'radar_guidance', False)
            is_emergency = getattr(self, 'emergency_escape', False)
            is_lift_up = getattr(self, 'lift_up', False)
            is_press_down = getattr(self, 'press_down', False)
            is_shatter = getattr(self, 'shattering_kick', False)

            if is_radar or is_emergency or is_lift_up or is_press_down or is_shatter:
                mutation_rate = getattr(self, 'mutation_rate', 0.1)

                if is_shatter: mutation_rate = getattr(self, 'mutation_rate', 0.8)
                elif is_emergency: mutation_rate = 0.5
                elif is_lift_up: mutation_rate = 0.4
                elif is_press_down: mutation_rate = 0.3
                
                do_mutation = np.random.rand(self.num_particles) < mutation_rate
                best_idx = np.argmin(self.pbest_scores)
                do_mutation[best_idx] = False 
                
                if is_radar:
                    targets_3d = []
                    for t in self.env.target_areas:
                        z_mid = (t.get('z_min', 0.0) + t.get('z_max', 10.0)) / 2.0
                        targets_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
                    
                    for i in range(self.num_particles):
                        if do_mutation[i]:
                            new_pos = np.zeros((self.num_waypoints, 3))
                            if len(targets_3d) > 0:
                                for j in range(self.num_waypoints):
                                    new_pos[j] = targets_3d[j % len(targets_3d)]
                            
                            noise = np.random.randn(self.num_waypoints, 3) * 2.0
                            self.particles[i] = np.clip((new_pos + noise).flatten(), self.lb, self.ub)
                            self.velocities[i] = 0.0 
                            
                            # 【修复核心】：突变后的评估
                            full_path = self._decode_path(self.particles[i])
                            score, _, _ = self.evaluator.evaluate_particle(full_path)
                            self.pbest_scores[i] = score
                            self.pbest_pos[i] = np.copy(self.particles[i])
                else:
                    mutation_scale = getattr(self, 'mutation_scale', 5.0)
                    levy_matrix = np.array([self._levy_step(self.dim) for _ in range(self.num_particles)])
                    noise = levy_matrix * (self.ub - self.lb) * mutation_scale * 0.01 * self.z_mask

                    if is_emergency:
                        for d in range(2, self.dim, 3): noise[:, d] = 15.0 
                    elif is_lift_up:
                        for d in range(2, self.dim, 3): noise[:, d] = 8.0  
                    elif is_press_down:
                        for d in range(2, self.dim, 3): noise[:, d] = -3.0 

                    mutated_particles = self.historical_best_pos + noise
                    
                    for i in range(self.num_particles):
                        if do_mutation[i]:
                            self.particles[i] = np.clip(mutated_particles[i], self.lb, self.ub)
                            self.velocities[i] = 0.0 
                            
                            # 【修复核心】：物理位移后的评估
                            full_path = self._decode_path(self.particles[i])
                            score, _, _ = self.evaluator.evaluate_particle(full_path)
                            self.pbest_scores[i] = score
                            self.pbest_pos[i] = np.copy(self.particles[i])

                for i in range(self.num_particles):
                    if self.pbest_scores[i] < self.historical_best_score:
                        self.historical_best_score = self.pbest_scores[i]
                        self.historical_best_pos = np.copy(self.pbest_pos[i])

                self.radar_guidance = False
                self.emergency_escape = False
                self.lift_up = False
                self.press_down = False
                self.shattering_kick = False

            self.convergence_curve.append(self.historical_best_score)
            
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 历史最优得分: {self.historical_best_score:,.2f}")
                
        return self._decode_path(self.historical_best_pos), self.convergence_curve

if __name__ == "__main__":
    save_dir = "PSO_random_map"
    os.makedirs(save_dir, exist_ok=True)
    
    num_runs = 5
    all_final_scores = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*20} 第 {run_idx+1}/{num_runs} 次运行 {'='*20}")
        planner = PSOPlanner(num_particles=50, max_iter=100, num_waypoints=12)
        best_path, history = planner.optimize()
        
        planner.plot_result(best_path, history, algo_name="PSO-3D", run_idx=run_idx, save_dir=save_dir)
        
        final_score = history[-1] if history else None
        all_final_scores.append(final_score)
        with open(os.path.join(save_dir, f"run_{run_idx:02d}_score.txt"), 'w') as f:
            f.write(f"Final score: {final_score:.2f}\n")
            
    # print("\n" + "="*50)
    # print("所有运行完成！结果保存在", save_dir)
    # print("各次最终得分:")
    # for i, score in enumerate(all_final_scores):
    #     print(f"  Run {i+1:02d}: {score:,.2f}")
    # if all_final_scores:
    #     avg = np.mean(all_final_scores)
    #     std = np.std(all_final_scores)
    #     print(f"\n平均得分: {avg:,.2f}  (±{std:,.2f})")
    # print("="*50)