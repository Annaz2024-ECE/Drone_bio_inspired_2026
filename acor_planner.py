import numpy as np
from base_planner import BasePlanner

class ACOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_waypoints=10, max_iter=200, 
                 archive_size=50, num_ants=40, q=0.1, xi=0.85):
        """
        
        :param archive_size: (k) 路径档案库大小（保留的最优解数量）
        :param num_ants: (M) 每次迭代新生成的蚂蚁数量
        :param q: 局部性参数 (locality parameter)，越小越倾向于选择排名第一的解，建议 0.01~0.1
        :param xi: 探索半径缩放因子 (类似于蒸发率的补集)，越大探索范围越广，建议 0.5~1.0
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        self.k = archive_size
        self.num_ants = num_ants
        self.q = q
        self.xi = xi
        
        # 记录全局最优
        self.global_best_path = None
        self.global_best_fitness = float('inf')
        self.convergence_curve = []
        
        # 预先计算好排行榜的权重 (公式: w_l = exp(-(l-1)^2 / (2 * q^2 * k^2)) )
        # 排名 l 从 1 到 k
        l_ranks = np.arange(1, self.k + 1)
        self.weights = np.exp(-((l_ranks - 1) ** 2) / (2.0 * (self.q ** 2) * (self.k ** 2)))
        # 将权重归一化为选择概率 p
        self.probs = self.weights / np.sum(self.weights)

    def _format_path(self, flat_vars):
        """ 工具函数：将 1D 扁平数组 [x1,y1,z1...] 转换为 3D 路径矩阵，并拼接起终点 """
        # 将一维数组重塑为 (num_waypoints, 3)
        waypoints_3d = flat_vars.reshape((self.num_waypoints, 3))
        # 拼接起点和终点
        full_path = np.vstack((self.env.start_point, waypoints_3d, self.env.end_point))
        return full_path

    def optimize(self):
        print(f"开始运行... 优化维度: {self.dim}D")
        
        # ================= 1. 初始化路径档案库 (Archive) =================
        # 在空间边界内随机生成 k 条路径 (均匀分布)
        archive_vars = np.random.uniform(self.lb, self.ub, (self.k, self.dim))
        archive_fitness = np.zeros(self.k)
        
        # 评估初始档案库
        for i in range(self.k):
            path = self._format_path(archive_vars[i])
            fitness, _, _ = self.evaluator.evaluate_pso_particle(path)
            archive_fitness[i] = fitness
            
        # 根据适应度对档案库从小到大排序 (Fitness越低越好)
        sort_idx = np.argsort(archive_fitness)
        archive_vars = archive_vars[sort_idx]
        archive_fitness = archive_fitness[sort_idx]
        
        # ================= 2. 算法主循环 =================
        for idx in range(self.max_iter):
            new_vars = np.zeros((self.num_ants, self.dim))
            new_fitness = np.zeros(self.num_ants)
            
            # --- 生成新一代蚂蚁 ---
            for ant in range(self.num_ants):
                # a) 根据概率轮盘赌选择一个“导师”路径 l
                l = np.random.choice(self.k, p=self.probs)
                guide_solution = archive_vars[l]
                
                # b) 计算该导师的搜索半径 (标准差 sigma)
                # 公式: sigma_i = xi * sum( |S_j,i - S_l,i| ) / (k-1)
                # 计算导师解与档案库中所有解的绝对差值平均
                diffs = np.abs(archive_vars - guide_solution)
                sigma = self.xi * np.sum(diffs, axis=0) / (self.k - 1)
                
                # c) 以导师为均值，sigma为标准差，进行高斯采样生成新坐标
                new_solution = np.random.normal(loc=guide_solution, scale=sigma)
                
                # d) 边界裁剪，防止飞出地图
                new_solution = np.clip(new_solution, self.lb, self.ub)
                new_vars[ant] = new_solution
                
                # e) 打分评估
                path = self._format_path(new_solution)
                new_fitness[ant], _, _ = self.evaluator.evaluate_pso_particle(path)
                
            # --- 档案库更新 (合并与淘汰) ---
            # 把老的 k 个解和新的 num_ants 个解拼在一起
            combined_vars = np.vstack((archive_vars, new_vars))
            combined_fitness = np.concatenate((archive_fitness, new_fitness))
            
            # 重新排序
            sort_idx = np.argsort(combined_fitness)
            
            # 末位淘汰：只保留前 k 个最优秀的解
            archive_vars = combined_vars[sort_idx][:self.k]
            archive_fitness = combined_fitness[sort_idx][:self.k]
            
            # 记录本代最优
            best_fitness_now = archive_fitness[0]
            if best_fitness_now < self.global_best_fitness:
                self.global_best_fitness = best_fitness_now
                self.global_best_path = self._format_path(archive_vars[0])
                
            self.convergence_curve.append(self.global_best_fitness)
            
            # 打印进度
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  > 迭代 {idx+1:03d}/{self.max_iter} | 全局最优得分: {self.global_best_fitness:,.2f}")
                
        print("优化完成！")
        return self.global_best_path, self.convergence_curve

# ================== 测试与执行 ==================
if __name__ == "__main__":
    from path_evaluator import PathEvaluator
    
    # 初始化评价器
    evaluator = PathEvaluator()
    
    # 实例化 ACOR，设定需要 15 个中间自由航点
    planner = ACOPlanner(
        evaluator=evaluator, 
        num_waypoints=15,   
        max_iter=300,        
        archive_size=50,     
        num_ants=50,        
        q=0.1,           
        xi=0.85                  )
    
    best_path, history = planner.optimize()
    
    # 可视化结果
    planner.plot_result(best_path, history, algo_name="ACO")
