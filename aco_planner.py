import numpy as np
import random
from base_planner import BasePlanner

class ACOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_ants=40, max_iter=200, num_waypoints=9,
                 alpha=1.0, beta=3.0, rho=0.2, Q=40):
        """ 继承自 BasePlanner 的 3D ACO 算法 (Y轴推进，X-Z平面寻优) """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        self.num_segments = self.num_waypoints + 1 
        
        # 1. 推进轴：预先计算 Y 轴的分段界限
        self.y_coords = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_segments + 1)
        
        # 2. 寻优面：在 X 和 Z 轴分别离散化，生成 X-Z 候选平面网格
        self.num_x = 20  # X 方向候选数量
        self.num_z = 10  # Z 方向候选数量 (高度层)
        self.x_candidates = np.linspace(self.env.x_bounds[0], self.env.x_bounds[1], self.num_x)
        self.z_candidates = np.linspace(self.env.z_bounds[0], self.env.z_bounds[1], self.num_z)
        
        # 扁平化 X-Z 组合，方便后续计算概率
        self.xz_candidates = []
        for x in self.x_candidates:
            for z in self.z_candidates:
                self.xz_candidates.append((x, z))
        self.num_candidates = len(self.xz_candidates)  # 总候选点数 (20 * 10 = 200)
        
        # 3. 初始化 3D 信息素矩阵
        self.pheromone = []
        self.pheromone.append(np.ones(self.num_candidates))
        for i in range(self.num_segments - 2):
            self.pheromone.append(np.ones((self.num_candidates, self.num_candidates)))
        self.pheromone.append(np.ones(self.num_candidates))
        
        self.global_best_path = None
        self.global_best_fitness = float('inf')

    def optimize(self):
        """算法主循环"""
        print("开始运行 3D 蚂蚁系统 (ACO) 路径规划 (Y轴推进)...")
        for idx in range(self.max_iter):
            all_paths, all_fitness = [], []
            
            for ant in range(self.num_ants):
                path = self._construct_path()
                # 丢入 3D 评估器流水线
                fitness, _, _ = self.evaluator.evaluate_pso_particle(path)
                all_paths.append(path)
                all_fitness.append(fitness)
                
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_path = path
            
            self.convergence_curve.append(self.global_best_fitness)
            self._update_pheromones(all_paths, all_fitness)
            
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  > 迭代 {idx+1:03d}/{self.max_iter} | 全局最优得分: {self.global_best_fitness:,.2f}")
                
        return self.global_best_path, self.convergence_curve

    def _get_target_3d_centers(self):
        """ 提取所有打卡点的 3D 空间几何中心 """
        centers_3d = []
        for t in self.env.target_areas:
            z_mid = (t.get('z_min', 0.0) + t.get('z_max', 20.0)) / 2.0
            centers_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
        return centers_3d

    def _construct_path(self):
        """ 单只蚂蚁构建一条 3D 路径 """
        path_idx = [] 
        target_centers_3d = self._get_target_3d_centers()
        
        # 1. 起点到第一截面
        prob = np.copy(self.pheromone[0])
        for j in range(self.num_candidates):
            x, z = self.xz_candidates[j]
            p_next = np.array([x, self.y_coords[1], z])
            
            if self.env.is_point_in_obstacle(p_next):
                prob[j] *= 0.01
                
            dist = self.env.calculate_distance(p_next, self.env.end_point)
            heuristic = 1.0 / (dist + 1e-4)
            prob[j] = (prob[j] ** self.alpha) * (heuristic ** self.beta)
            
        prob = prob / (np.sum(prob) + 1e-12)
        prob = prob / np.sum(prob)
        first_idx = np.random.choice(self.num_candidates, p=prob)
        path_idx.append(first_idx)
        
        # 2. 截面间的推进
        for i in range(1, self.num_segments - 1):
            curr_idx = path_idx[-1]
            prob = np.copy(self.pheromone[i][curr_idx])
            
            curr_x, curr_z = self.xz_candidates[curr_idx]
            p_curr = np.array([curr_x, self.y_coords[i], curr_z])
            
            for j in range(self.num_candidates):
                next_x, next_z = self.xz_candidates[j]
                p_next = np.array([next_x, self.y_coords[i+1], next_z])
                
                # 3D 碰撞检测
                if self.env.is_segment_collision(p_curr, p_next, safe_margin=0.0):
                    prob[j] *= 0.001
                    
                dist_to_end = self.env.calculate_distance(p_next, self.env.end_point)
                
                # 计算离 3D 巡检目标的距离引力
                dist_to_targets = [self.env.calculate_distance(p_next, c) for c in target_centers_3d]
                min_target_dist = min(dist_to_targets) if dist_to_targets else 0
                
                heuristic = 1.0 / (dist_to_end + 0.3 * min_target_dist + 1e-4)
                prob[j] = (prob[j] ** self.alpha) * (heuristic ** self.beta)
            
            prob = prob / (np.sum(prob) + 1e-12)
            prob = prob / np.sum(prob)
            next_idx = np.random.choice(self.num_candidates, p=prob)    
            path_idx.append(next_idx)
            
        # 3. 组装最终 3D 路线
        actual_path = [self.env.start_point]
        for i, idx in enumerate(path_idx):
            x, z = self.xz_candidates[idx]
            actual_path.append(np.array([x, self.y_coords[i+1], z]))
        actual_path.append(self.env.end_point)
        
        return np.array(actual_path)

    def _update_pheromones(self, all_paths, all_fitness):
        """ 3D 空间的信息素更新 """
        self.pheromone[0] *= (1.0 - self.rho)
        self.pheromone[0] = np.clip(self.pheromone[0], 0.1, 10.0)
        for i in range(1, self.num_segments - 1):
            self.pheromone[i] *= (1.0 - self.rho)
            self.pheromone[i] = np.clip(self.pheromone[i], 0.1, 10.0)
        self.pheromone[-1] *= (1.0 - self.rho)
        self.pheromone[-1] = np.clip(self.pheromone[-1], 0.1, 10.0)
        
        for path, fitness in zip(all_paths, all_fitness):
            delta_p = self.Q / (fitness + 1e-4)
            
            xz_indices = []
            for p in path[1:-1]:
                # 反推当前航点在 X 和 Z 离散轴上的索引
                x_idx = np.argmin(np.abs(self.x_candidates - p[0]))
                z_idx = np.argmin(np.abs(self.z_candidates - p[2]))
                # 计算在扁平化列表 self.xz_candidates 中的组合索引
                flat_idx = x_idx * self.num_z + z_idx
                xz_indices.append(flat_idx)
                
            self.pheromone[0][xz_indices[0]] += delta_p
            for i in range(len(xz_indices) - 1):
                self.pheromone[i+1][xz_indices[i], xz_indices[i+1]] += delta_p
            self.pheromone[-1][xz_indices[-1]] += delta_p

if __name__ == "__main__":
    from path_evaluator import PathEvaluator
    
    evaluator = PathEvaluator()
    # 因为有 200 个候选点，蚂蚁数稍微增加有助于初期探索
    planner = ACOPlanner(evaluator=evaluator, num_waypoints=10, num_ants=40, max_iter=20)
    best_path, history = planner.optimize()
    
    # 调用通用的 3D 绘图接口
    planner.plot_result(best_path, history, algo_name="ACO")