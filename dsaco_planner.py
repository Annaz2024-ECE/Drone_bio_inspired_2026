import numpy as np
import random
from base_planner import BasePlanner

class DSACOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_ants=50, max_iter=200, num_waypoints=9,
                 alpha=1.0, beta=4.0, rho=0.2, Q=50):
        """ 继承自 BasePlanner 的 DSACO 算法 """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.num_segments = self.num_waypoints + 1
        
        # ==== DSACO 论文新增核心参数 ====
        self.q0_min = 0.1         # 确定性选择概率的初始最小值
        self.q0_max = 0.9         # 确定性选择概率的最终最大值
        self.tau_max = 10.0       # MAX-MIN 信息素上限
        self.tau_min = 0.1        # MAX-MIN 信息素下限
        self.p_weight = 1.0       # 动态适应度权重参数 p
        # ================================
        
        # 预先计算 Y 轴的分段界限（从南门到北门稳步推进）
        self.y_coords = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_segments + 1)
        
        # 针对每个分段，生成离散的候选 X 轴坐标（允许全图左右横跳打卡）
        self.num_candidates = 30  
        self.x_candidates = np.linspace(self.env.x_bounds[0], self.env.x_bounds[1], self.num_candidates)
        
        # 初始化信息素矩阵 (初始化为 tau_max，鼓励初期充分探索)
        self.pheromone = []
        self.pheromone.append(np.full(self.num_candidates, self.tau_max))
        for i in range(self.num_segments - 2):
            self.pheromone.append(np.full((self.num_candidates, self.num_candidates), self.tau_max))
        self.pheromone.append(np.full(self.num_candidates, self.tau_max))
        
        self.global_best_path = None
        self.global_best_fitness = float('inf')

    def optimize(self):
        """算法主循环"""
        print("开始运行 DSACO (双策略蚁群优化) 路径规划...")
        score_history = [] 
        
        for idx in range(self.max_iter):
            all_paths = []
            all_fitness = []
            
            for ant in range(self.num_ants):
                # 传入当前代数 idx，用于计算动态双策略参数 q0
                path = self._construct_path(idx)
                fitness, _, _ = self.evaluator.evaluate_pso_particle(path)
                
                all_paths.append(path)
                all_fitness.append(fitness)
                
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_path = path
            
            self.convergence_curve.append(self.global_best_fitness)
            
            # 传入当前代数 idx，用于动态更新精英信息素
            self._update_pheromones(all_paths, all_fitness, idx)
            
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"迭代次数 [{idx+1}/{self.max_iter}] -> 当前历史最佳适应度(Fitness): {self.global_best_fitness:.2f}")
                
        print("优化完成！")
        return self.global_best_path, self.convergence_curve

    def _construct_path(self, current_iteration):
        """单只蚂蚁构建路点路径 (融合 DSACO 双策略)"""
        path_idx = [] 
        
        # 【DSACO 策略 1】：动态调整确定性选择概率 q0
        # 随着迭代进行，q0 线性增加。前期重探索(Roulette)，后期重开发(Argmax)
        q0 = self.q0_min + (self.q0_max - self.q0_min) * (current_iteration / self.max_iter)
        
        # 1. 决定从起点到第1个控制点
        prob = np.copy(self.pheromone[0])
        for j in range(self.num_candidates):
            p_next = np.array([self.x_candidates[j], self.y_coords[1]])
            if self.env.is_point_in_obstacle(p_next):
                prob[j] *= 0.01
            
            dist = self.env.calculate_distance(p_next, self.env.end_point)
            heuristic = 1.0 / (dist + 1e-4)
            prob[j] = (prob[j] ** self.alpha) * (heuristic ** self.beta)
            
        # 【DSACO 核心】：双策略状态转移
        q = random.random()
        if q <= q0:
            first_idx = np.argmax(prob) # 确定性选择：直接拿最大值，无需归一化
        else:
            # 随机探索：严格的轮盘赌与数值精度防御
            total_prob = np.sum(prob)
            if total_prob > 0:
                prob = prob / total_prob
            else:
                prob = np.ones_like(prob) / len(prob)
            prob = prob / np.sum(prob) 
            first_idx = np.random.choice(self.num_candidates, p=prob)
            
        path_idx.append(first_idx)
        
        # 2. 决定中间各列的选择
        for i in range(1, self.num_segments - 1):
            curr_x_idx = path_idx[-1] 
            prob = np.copy(self.pheromone[i][curr_x_idx])
            p_curr = np.array([self.x_candidates[curr_x_idx], self.y_coords[i]])
            
            for j in range(self.num_candidates):
                p_next = np.array([self.x_candidates[j], self.y_coords[i+1]])
                
                if self.env.is_segment_collision(p_curr, p_next, safe_margin=0.0):
                    prob[j] *= 0.001
                    
                dist_to_end = self.env.calculate_distance(p_next, self.env.end_point)
                dist_to_targets = [self.env.calculate_distance(p_next, t['center']) for t in self.env.target_areas]
                min_target_dist = min(dist_to_targets)
                
                heuristic = 1.0 / (dist_to_end + 0.3 * min_target_dist + 1e-4)
                prob[j] = (prob[j] ** self.alpha) * (heuristic ** self.beta)
            
            # 【DSACO 核心】：双策略状态转移
            q = random.random()
            if q <= q0:
                next_idx = np.argmax(prob) 
            else:
                total_prob = np.sum(prob)
                if total_prob > 0:
                    prob = prob / total_prob
                else:
                    prob = np.ones_like(prob) / len(prob)
                prob = prob / np.sum(prob) 
                next_idx = np.random.choice(self.num_candidates, p=prob)    
            
            path_idx.append(next_idx)
            
        # 3. 将索引组装为坐标
        actual_path = [self.env.start_point]
        for i, x_idx in enumerate(path_idx):
            actual_path.append(np.array([self.x_candidates[x_idx], self.y_coords[i+1]]))
        actual_path.append(self.env.end_point)
        
        return np.array(actual_path)

    def _update_pheromones(self, all_paths, all_fitness, current_iteration):
        """融合 DSACO 策略 2 与 3：精英动态更新与 MAX-MIN 边界"""
        
        # 【DSACO 核心】：每代仅允许最优蚂蚁释放信息素 (摒弃全员更新)
        best_idx_now = np.argmin(all_fitness)
        best_path_now = all_paths[best_idx_now]
        fitness_now = all_fitness[best_idx_now]
        
        # 1. 全局信息素蒸发
        for i in range(len(self.pheromone)):
            self.pheromone[i] *= (1.0 - self.rho)
            
        # 2. 动态自适应权重衰减 (论文 Eq 19)
        if current_iteration > 0 and current_iteration % 10 == 0:
            self.p_weight *= 0.9 

        # 3. 动态释放信息素 (仅由当代最优蚂蚁释放)
        denominator = (1.0 - self.p_weight) * fitness_now + self.p_weight * self.global_best_fitness
        delta_p = self.Q / (denominator + 1e-4)
        
        # 反推当前最优路径的 X 网格索引
        x_indices = []
        for p in best_path_now[1:-1]:
            idx = np.argmin(np.abs(self.x_candidates - p[0]))
            x_indices.append(idx)
            
        # 将精英增量叠加到对应的转移路径上
        self.pheromone[0][x_indices[0]] += delta_p
        for i in range(len(x_indices) - 1):
            self.pheromone[i+1][x_indices[i], x_indices[i+1]] += delta_p
        self.pheromone[-1][x_indices[-1]] += delta_p
        
        # 【DSACO 核心】：强制应用 MAX-MIN 边界限制，防止局部死锁
        for i in range(len(self.pheromone)):
            self.pheromone[i] = np.clip(self.pheromone[i], self.tau_min, self.tau_max)
    
if __name__ == "__main__":
    planner = DSACOPlanner()
    best_path, history = planner.optimize()
    planner.plot_result(best_path, history, algo_name="DSACO")