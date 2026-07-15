import numpy as np
import random
from base_planner import BasePlanner

class ACOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_ants=50, max_iter=200, num_waypoints=5,
                 alpha=1.0, beta=3.0, rho=0.2, Q=400):
        """ 
        基于拓扑图节点的 TSP 蚁群算法 (闭环巡检模式)
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # ================= 重大修改 1：重构节点体系 =================
        # 节点 0 为起点(南门)，节点 1~N 为巡检区中心
        self.nodes = [self.env.start_point] + [t['center'] for t in self.env.target_areas]
        self.num_nodes = len(self.nodes)
        
        # 初始化信息素矩阵：形状为 (总节点数, 总节点数) 的全连接图
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))
        
        self.global_best_path = None
        self.global_best_fitness = float('inf')
        self.convergence_curve = []

    def optimize(self):
        """算法主循环"""
        print("开始运行基于拓扑图的 ACO 闭环路径规划...")
        
        for idx in range(self.max_iter):
            all_paths = []
            all_path_nodes = [] # 记录蚂蚁走过的节点顺序，用于更新信息素
            all_fitness = []
            
            for ant in range(self.num_ants):
                # 蚂蚁构建路径及其节点序列
                path, path_nodes = self._construct_path()
                
                # 计算适应度得分
                fitness, _, _ = self.evaluator.evaluate_pso_particle(path)
                
                all_paths.append(path)
                all_path_nodes.append(path_nodes)
                all_fitness.append(fitness)
                
                # 记录全局最优
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_path = path
            
            self.convergence_curve.append(self.global_best_fitness)
            
            # 传入节点序列更新信息素
            self._update_pheromones(all_path_nodes, all_fitness)
            
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  > 迭代 {idx+1:03d}/{self.max_iter} | 全局最优得分: {self.global_best_fitness:.2f}")
                
        return self.global_best_path, self.convergence_curve

    def _construct_path(self):
        """单只蚂蚁构建闭环路径"""
        # ================= 重大修改 2：禁忌表与状态转移 =================
        path_nodes = [0]        # 强制从起点（节点0）出发
        visited = {0}           # 禁忌表：记录已经访问过的节点
        
        while len(visited) < self.num_nodes:
            curr_node = path_nodes[-1]
            prob = np.zeros(self.num_nodes)
            
            for next_node in range(self.num_nodes):
                if next_node in visited:
                    continue # 已经去过的节点，概率为0
                    
                p_curr = self.nodes[curr_node]
                p_next = self.nodes[next_node]
                
                # 启发式信息：节点间的距离
                dist = self.env.calculate_distance(p_curr, p_next)
                
                # 运动学粗筛：如果两点直连撞墙，给予极大惩罚
                if self.env.is_segment_collision(p_curr, p_next, safe_margin=0.0):
                    heuristic = 1.0 / (dist + 1e4)
                else:
                    heuristic = 1.0 / (dist + 1e-4)
                    
                # 综合计算转移概率
                prob[next_node] = (self.pheromone[curr_node, next_node] ** self.alpha) * (heuristic ** self.beta)
                
            # 轮盘赌选择下一个节点
            total_prob = np.sum(prob)
            if total_prob > 0:
                prob /= total_prob
            else:
                # 兜底：如果全部撞墙导致概率极低，从未访问节点中平均分配概率
                unvisited = [n for n in range(self.num_nodes) if n not in visited]
                prob = np.zeros(self.num_nodes)
                for n in unvisited:
                    prob[n] = 1.0 / len(unvisited)
            
            prob /= np.sum(prob) # 消除浮点误差
            next_chosen = np.random.choice(self.num_nodes, p=prob)
            
            path_nodes.append(next_chosen)
            visited.add(next_chosen)
            
        # ================= 重大修改 3：强制闭环 =================
        path_nodes.append(0) # 访问完所有目标后，强制添加回节点0（南门）
        
        # 将节点索引组装为真正的全节点坐标路径集合
        actual_path = np.array([self.nodes[idx] for idx in path_nodes])
        
        return actual_path, path_nodes

    def _update_pheromones(self, all_path_nodes, all_fitness):
        """针对图节点的边进行信息素更新"""
        # 1. 信息素全局蒸发
        self.pheromone *= (1.0 - self.rho)
        self.pheromone = np.clip(self.pheromone, 0.1, 10.0) 
        
        # 2. 根据蚂蚁留下的路线释放新信息素
        for path_nodes, fitness in zip(all_path_nodes, all_fitness):
            delta_p = self.Q / (fitness + 1e-4)
            
            # 只在走过的“边”上增加信息素
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i+1]
                self.pheromone[u, v] += delta_p
                # 如果无人机来回飞行的代价不同（比如考虑风向），则不加下面这行。
                # 否则可以加上，使其成为无向图：self.pheromone[v, u] += delta_p

if __name__ == "__main__":
    planner = ACOPlanner(max_iter=100)
    best_path, history = planner.optimize()
    
    # 记得调用你之前写的平滑函数绘制最终平滑轨迹
    planner.plot_result(best_path, history, algo_name="ACO (TSP Closed-Loop)")