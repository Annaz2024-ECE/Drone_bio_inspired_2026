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
        """ 
        闭环专属初始化：全图随机撒点 + 极坐标雷达扫描排序 (防打结) 
        """
        particles = np.zeros((self.num_particles, self.dim))
        
        # 1. 寻找地图的近似中心点 (用于充当雷达中心)
        center_x = (self.env.x_bounds[0] + self.env.x_bounds[1]) / 2.0
        center_y = (self.env.y_bounds[0] + self.env.y_bounds[1]) / 2.0
        
        for i in range(self.num_particles):
            # 2. 在整个地图范围内广泛随机撒点 (稍微避开极端的贴墙边缘)
            rand_x = np.random.uniform(self.env.x_bounds[0] + 5, self.env.x_bounds[1] - 5, self.num_waypoints)
            rand_y = np.random.uniform(self.env.y_bounds[0] + 5, self.env.y_bounds[1] - 5, self.num_waypoints)
            raw_waypoints = np.column_stack((rand_x, rand_y))
            
            # 3. 闭环防打结核心：算每个点相对于地图中心的极坐标角度 (arctan2)
            # 角度范围从 -π 到 π，按角度排序就能把杂乱的点串成一个环！
            angles = np.arctan2(raw_waypoints[:, 1] - center_y, raw_waypoints[:, 0] - center_x)
            sorted_waypoints = raw_waypoints[np.argsort(angles)]
            
            # 4. 展平并限制边界
            particles[i] = np.clip(sorted_waypoints.flatten(), self.lb, self.ub)
            
        return particles
    def optimize(self):
        """ PSO 主循环 """
        print("开始 PSO 粒子群算法路径规划...")
        
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

if __name__ == "__main__":
    planner = PSOPlanner()
    best_path, history = planner.optimize()
    planner.plot_result(best_path, history, algo_name="PSO")