import numpy as np
import random
from base_planner import BasePlanner

class ACOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_ants=50, max_iter=200, num_waypoints=9,
                 alpha=1.0, beta=3.0, rho=0.2, Q=40):
        """ 继承自 BasePlanner 的 ACO 算法 (离散网格) """
        # ACO 中的分段数即控制点数 + 1

        """
        :param env: UAVEnvironment2D 实例
        :param evaluator: PathEvaluator 实例
        :param num_ants: 蚂蚁数量
        :param num_iterations: 迭代步数
        :param alpha: 信息素重要程度
        :param beta: 启发式信息重要程度
        :param rho: 信息素挥发率 (Evaporation rate)
        :param Q: 信息素增强常数
        :param num_segments: 路径的中间分段数（决定了控制点的数量）
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.num_segments = self.num_waypoints + 1 
        
        # 预先计算 Y 轴的分段界限（从起点到终点）
        self.y_coords = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_segments + 1)
        
        # 针对每个分段，生成离散的候选 X 轴坐标（伪连续空间）
        self.num_candidates = 30  # 每一列可选的高度数量
        self.x_candidates = np.linspace(self.env.x_bounds[0], self.env.x_bounds[1], self.num_candidates)
        
        # 初始化信息素矩阵：形状为 (分段数-1, 当前列候选点数, 下一列候选点数)
        # 第一阶段和最后一阶段分别连接起点和终点
        self.pheromone = []
        # 起点到第一列候选点
        self.pheromone.append(np.ones(self.num_candidates))
        # 中间各列之间
        for i in range(self.num_segments - 2):
            self.pheromone.append(np.ones((self.num_candidates, self.num_candidates)))
        # 最后一列到终点
        self.pheromone.append(np.ones(self.num_candidates))
        
        self.global_best_path = None
        self.global_best_fitness = float('inf')

    def optimize(self):
        """算法主循环"""
        print("开始运行蚂蚁系统(ACO)路径规划...")
        score_history = [] #记录收敛曲线
        for idx in range(self.max_iter):
            all_paths, all_fitness = [], []
            
            for ant in range(self.num_ants):
                path = self._construct_path()
                fitness, _, _ = self.evaluator.evaluate_pso_particle(path)
                all_paths.append(path)
                all_fitness.append(fitness)
                
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_path = path
            
            self.convergence_curve.append(self.global_best_fitness)
            self._update_pheromones(all_paths, all_fitness)
            
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  > 迭代 {idx+1:03d}/{self.max_iter} | 全局最优得分: {self.global_best_fitness:.2f}")
                
        return self.global_best_path, self.convergence_curve

    def _construct_path(self):
        """单只蚂蚁构建一条从起点到终点的路点路径"""
        path_idx = [] # 记录每一列选中的Y候选点索引
        
        # 1. 决定从起点到第1个控制点
        prob = np.copy(self.pheromone[0])
        # 启发式信息：计算第1列的所有候选点到终点以及到巡检区的启发距离
        for j in range(self.num_candidates):
            p_next = np.array([self.x_candidates[j], self.y_coords[1]])
            # 如果直接撞墙，人为压低概率
            if self.env.is_point_in_obstacle(p_next):
                prob[j] *= 0.01
            # 距离目标的潜在吸引力
            dist = self.env.calculate_distance(p_next, self.env.end_point)
            heuristic = 1.0 / (dist + 1e-4)
            prob[j] = (prob[j] ** self.alpha) * (heuristic ** self.beta)
            
        total_prob = np.sum(prob)
        if total_prob > 0:
            prob /= total_prob
        else:
            prob = np.ones_like(prob) / len(prob) # 全撞墙时平均分配概率

        # 兜底：由于浮点数微小截断误差，强制让最后一项补齐 1.0
        prob /= np.sum(prob) 

        first_idx = np.random.choice(self.num_candidates, p=prob)
        path_idx.append(first_idx)
        
        # 2. 决定中间各列的选择
        for i in range(1, self.num_segments - 1):
            curr_x_idx = path_idx[-1] #修改XY 6.20
            prob = np.copy(self.pheromone[i][curr_x_idx])
            
            p_curr = np.array([self.x_candidates[curr_x_idx], self.y_coords[i]])
            
            for j in range(self.num_candidates):
                p_next = np.array([self.x_candidates[j], self.y_coords[i+1]])
                
                # 运动学粗筛：如果当前步和下一步直接导致撞墙，给与极大惩罚降低转移概率
                if self.env.is_segment_collision(p_curr, p_next, safe_margin=0.0):
                    prob[j] *= 0.001
                    
                # 启发式设计：向终点靠近的同时，加入对“未穿过巡检区”的动态重力吸引
                dist_to_end = self.env.calculate_distance(p_next, self.env.end_point)
                
                # 引导蚂蚁向巡检目标中心靠拢
                dist_to_targets = [self.env.calculate_distance(p_next, t['center']) for t in self.env.target_areas]
                min_target_dist = min(dist_to_targets)
                
                # 综合启发因子：既要看离终点的总距离，又要看离巡检目标的距离
                heuristic = 1.0 / (dist_to_end + 0.3 * min_target_dist + 1e-4)
                prob[j] = (prob[j] ** self.alpha) * (heuristic ** self.beta)
            
            total_prob = np.sum(prob)
            if total_prob > 0:
                prob /= total_prob
            else:
                prob = np.ones_like(prob) / len(prob)

            # 兜底：消除任何潜在的浮点数微小截断误差
            prob /= np.sum(prob) 

            next_idx = np.random.choice(self.num_candidates, p=prob)    
            
            path_idx.append(next_idx)
            
        # 3. 将索引组装为真正的全节点坐标路径集合 (包含起点和终点)
        actual_path = [self.env.start_point]
        for i, x_idx in enumerate(path_idx):
            actual_path.append(np.array([self.x_candidates[x_idx], self.y_coords[i+1]]))
        actual_path.append(self.env.end_point)
        
        return np.array(actual_path)

    def _update_pheromones(self, all_paths, all_fitness):
        """更新信息素（包含蒸发和增量释放）"""
        # 1. 信息素蒸发
        self.pheromone[0] *= (1.0 - self.rho)
        self.pheromone[0] = np.clip(self.pheromone[0], 0.1, 10.0) # 防止极端衰减
        
        for i in range(1, self.num_segments - 1):
            self.pheromone[i] *= (1.0 - self.rho)
            self.pheromone[i] = np.clip(self.pheromone[i], 0.1, 10.0)
            
        self.pheromone[-1] *= (1.0 - self.rho)
        self.pheromone[-1] = np.clip(self.pheromone[-1], 0.1, 10.0)
        
        # 2. 根据蚂蚁的路径质量释放新信息素
        for path, fitness in zip(all_paths, all_fitness):
            # 因为未通过目标点或撞墙的分数往往高达几十万，倒数会趋于0，
            # 这符合蚁群算法规则：只有合法且短的路径才会留存高浓度信息素。
            delta_p = self.Q / (fitness + 1e-4)
            
            # 反推路径在候选Y矩阵里的索引位置
            x_indices = []
            for p in path[1:-1]:
                idx = np.argmin(np.abs(self.x_candidates - p[0]))
                x_indices.append(idx)
                
            # 叠加到对应的转移路径上
            self.pheromone[0][x_indices[0]] += delta_p
            for i in range(len(x_indices) - 1):
                self.pheromone[i+1][x_indices[i], x_indices[i+1]] += delta_p
            self.pheromone[-1][x_indices[-1]] += delta_p


if __name__ == "__main__":
    planner = ACOPlanner()
    best_path, history = planner.optimize()
    planner.plot_result(best_path, history, algo_name="ACO")