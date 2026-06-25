import numpy as np
from base_planner import BasePlanner

class PSOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_particles=100, max_iter=200, num_waypoints=15):
        """ 继承自 BasePlanner 的 PSO 算法 """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_particles = num_particles
        
        # PSO 核心参数
        self.w_max = 0.9  
        self.w_min = 0.4  
        self.c1 = 1.5 
        self.c2 = 1.5 
        self.v_max = 8.0 
        
        # 初始化粒子位置和速度 (使用基类的 self.dim)
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((self.num_particles, self.dim))
        
        # 记录个体最优 (pbest) 和全局最优 (gbest)
        self.pbest_pos = np.copy(self.particles)
        self.pbest_scores = np.full(self.num_particles, np.inf)
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

    def _initialize_particles(self):
        """ 初始化粒子并实施投影排序防打结策略 """
        particles = np.zeros((self.num_particles, self.dim))
        direction_vec = self.env.end_point - self.env.start_point
        
        for i in range(self.num_particles):
            x_vals = np.linspace(self.env.start_point[0], self.env.end_point[0], self.num_waypoints + 2)[1:-1]
            y_vals = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_waypoints + 2)[1:-1]
            
            noise_x = np.random.uniform(-40, 40, self.num_waypoints)
            noise_y = np.random.uniform(-40, 40, self.num_waypoints)
            raw_waypoints = np.column_stack((x_vals + noise_x, y_vals + noise_y))
            
            projections = np.dot(raw_waypoints - self.env.start_point, direction_vec)
            sorted_waypoints = raw_waypoints[np.argsort(projections)]
            
            # 展平为一维存入，并限制边界
            particles[i] = np.clip(sorted_waypoints.flatten(), self.lb, self.ub)
            
        return particles

    def optimize(self):
        """ PSO 主循环 """
        print("🚀 开始 PSO 粒子群算法路径规划...")
        
        # 初始评估
        for i in range(self.num_particles):
            full_path = self._decode_path(self.particles[i])
            score, _ = self.evaluator.evaluate_pso_particle(full_path)
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_pos = np.copy(self.particles[i])
                
        for iteration in range(self.max_iter):
            w_current = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            for i in range(self.num_particles):
                # 完全矩阵化的一维计算
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest_pos - self.particles[i])
                self.velocities[i] = w_current * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)
                
                full_path = self._decode_path(self.particles[i])
                score = self.evaluator.evaluate_pso_particle(full_path)
                
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

if __name__ == "__main__":
    planner = PSOPlanner()
    best_path, history = planner.optimize()
    planner.plot_result(best_path, history, algo_name="PSO")