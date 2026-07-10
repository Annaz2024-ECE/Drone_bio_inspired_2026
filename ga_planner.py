import numpy as np
import concurrent.futures
import time
from base_planner import BasePlanner

class GAPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_waypoints=20, pop_size=60, max_iter=200, 
                 pc=0.8, pm=0.2, tournament_size=3):
        """
        基于遗传算法 (GA) 的 3D 路径规划器
        :param pc: 交叉概率 (Probability of Crossover)
        :param pm: 变异概率 (Probability of Mutation)
        :param tournament_size: 锦标赛选择的竞争者数量
        """
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.tournament_size = tournament_size
        
        # 初始化种群：随机生成 N 个个体
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # 提取 3D 巡检目标点 (Domain Knowledge) 用于变异时的定向注入
        self.target_anchors = []
        for target in self.env.target_areas:
            center_x, center_y = target['center'][:2]
            z_mid = (target.get('z_min', 4.0) + target.get('z_max', 8.0)) / 2.0
            self.target_anchors.append([center_x, center_y, z_mid])

        # 【优化】选取前 30% 的初始种群，强制将它们的前几个航点设为必经打卡点
        num_targets = len(self.target_anchors)
        if num_targets > 0:
            seed_count = int(self.pop_size * 0.3)
            for i in range(seed_count):
                # 将一个一维数组变回 (N, 3) 的矩阵以便操作
                individual = self.population[i].reshape((self.num_waypoints, 3))
                # 强制替换前几个控制点为打卡点 (由于 _decode_path 会重排，替换哪个位置无所谓)
                for j in range(min(num_targets, self.num_waypoints - 2)):
                    individual[j] = self.target_anchors[j]
                # 展平塞回去
                self.population[i] = individual.flatten()
        

    def _decode_path(self, position):
        """
        沿用强大的最近邻 (Nearest Neighbor) 重排逻辑
        防止遗传算法交叉变异后产生“意大利面条”式的打结航线
        """
        waypoints = position.reshape((self.num_waypoints, 3))
        sorted_waypoints = []
        current_point = self.env.start_point 
        remaining_indices = list(range(self.num_waypoints))
        
        while remaining_indices:
            best_idx = -1
            min_dist = float('inf')
            
            for idx in remaining_indices:
                dist = np.linalg.norm(waypoints[idx] - current_point)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
                    
            sorted_waypoints.append(waypoints[best_idx])
            current_point = waypoints[best_idx]
            remaining_indices.remove(best_idx)
            
        sorted_waypoints = np.array(sorted_waypoints)
        return np.vstack([self.env.start_point, sorted_waypoints, self.env.end_point])

    def _tournament_selection(self, fitness_scores):
        """ 
        锦标赛选择：每次随机挑出几个个体，选其中分数最低（适应度最好）的 
        """
        selected_indices = []
        for _ in range(self.pop_size):
            # 随机挑选竞争者
            candidates = np.random.choice(self.pop_size, self.tournament_size, replace=False)
            # 在你的评价器中，分数越低代表惩罚越少，路径越优
            best_candidate = candidates[np.argmin(fitness_scores[candidates])]
            selected_indices.append(best_candidate)
        return selected_indices

    def _crossover(self, parent1, parent2, pc):
        """ 
        单点交叉 (航点级别的交叉) 
        """
        if np.random.rand() < pc:
            p1 = parent1.reshape((self.num_waypoints, 3))
            p2 = parent2.reshape((self.num_waypoints, 3))
            
            # 随机选择一个交叉点 (至少保留一个航点不被切断)
            cross_point = np.random.randint(1, self.num_waypoints)
            
            # 交换基因片段
            child1 = np.vstack((p1[:cross_point], p2[cross_point:])).flatten()
            child2 = np.vstack((p2[:cross_point], p1[cross_point:])).flatten()
            return child1, child2
        
        # 如果不交叉，原样遗传
        return parent1.copy(), parent2.copy()

    def _mutate(self, child, pm):
        """ 
        高斯变异与目标锚点定向变异结合
        """
        if np.random.rand() < pm:
            c = child.reshape((self.num_waypoints, 3))
            # 随机挑选 1 到多个航点进行变异
            num_mutations = np.random.randint(1, max(2, self.num_waypoints // 3))
            indices = np.random.choice(self.num_waypoints, num_mutations, replace=False)
            
            for idx in indices:
                rand_val = np.random.rand()
                # 策略 1: 30% 概率，将某个变异点直接强行改成目标打卡点（精英引导）
                if rand_val < 0.3 and len(self.target_anchors) > 0:
                    c[idx] = self.target_anchors[np.random.choice(len(self.target_anchors))]
                # 策略 2: 40% 概率，在地图范围内重新随机生成一个全新航点
                elif rand_val < 0.7:
                    # 使用前 3 个维度 (X, Y, Z) 的边界
                    c[idx] = np.random.uniform(self.lb[:3], self.ub[:3])
                # 策略 3: 40% 概率，在当前位置叠加高斯噪声进行微调
                else:
                    noise = np.random.normal(0, 5.0, 3) # 标准差为 5.0 米
                    c[idx] = np.clip(c[idx] + noise, self.lb[:3], self.ub[:3])
                    
            return c.flatten()
        return child

    def _evaluate_single(self, position):
        path = self._decode_path(position)
        # 调用评价器时开启 fast_mode
        score, _, _ = self.evaluator.evaluate_pso_particle(path)
        return score

    def optimize(self):
        print("开始 3D 环境下 GA 遗传算法路径规划...")
        # 使用线程池并发评估 (避开多进程可能导致的 evaluator 无法 pickling 的问题)
        executor = concurrent.futures.ThreadPoolExecutor()
        # 并发初始种群评估
        fitness_scores = np.zeros(self.pop_size)
        results = list(executor.map(self._evaluate_single, self.population))
        for i, score in enumerate(results):
            fitness_scores[i] = score
            if score < self.historical_best_score:
                self.historical_best_score = score
                self.historical_best_pos = self.population[i].copy()
        # 开始进化迭代
        for t in range(self.max_iter):
            # 动态自适应概率：前期大步探索(高变异)，后期精细收敛(低变异)
            current_pc = self.pc - (self.pc - 0.5) * (t / self.max_iter)
            current_pm = self.pm - (self.pm - 0.05) * (t / self.max_iter)
            # 1. 选择 (Selection)
            selected_indices = self._tournament_selection(fitness_scores)
            new_population = []
            
            # 精英保留策略 (Elitism)：强制保留当前最好个体，防止退化
            new_population.append(self.historical_best_pos.copy())
            
            # 2. 交叉与变异 (Crossover & Mutation)
            # 每次取两个父代繁衍两个子代，直到填满新种群 (去掉一个精英的名额)
            for i in range(0, self.pop_size - 1, 2):
                parent1 = self.population[selected_indices[i]]
                # 确保 parent2 的索引不越界
                p2_idx = i + 1 if i + 1 < len(selected_indices) else 0
                parent2 = self.population[selected_indices[p2_idx]]
                
                child1, child2 = self._crossover(parent1, parent2, current_pc)
                
                child1 = self._mutate(child1, current_pm)
                child2 = self._mutate(child2, current_pm)
                
                new_population.extend([child1, child2])
                
            # 截断多余的个体（处理 pop_size 为奇数的情况）
            self.population = np.array(new_population[:self.pop_size])
            
            # 3. 评估新种群
            results = list(executor.map(self._evaluate_single, self.population))
            
            for i, score in enumerate(results):
                fitness_scores[i] = score
                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = self.population[i].copy()

            # 记录历史最优
            self.convergence_curve.append(self.historical_best_score)
            
            # 打印进度
            if (t + 1) % 50 == 0 or t == 0:
                print(f"代数: {t + 1}/{self.max_iter}, 当前最优得分: {self.historical_best_score:,.2f}")
        
        # 关闭线程池
        executor.shutdown()
        return self._decode_path(self.historical_best_pos), self.convergence_curve

if __name__ == "__main__":
    start_time = time.time()
    # 实例化规划器
    planner = GAPlanner(num_waypoints=20, pop_size=60, max_iter=200, pc=0.85, pm=0.2)
    
    # 运行优化
    best_path, convergence_history = planner.optimize()
    
    print(f"\nGA 规划完成！最终得分: {convergence_history[-1]:,.2f}")
    
    # 调用父类的绘图函数呈现结果
    planner.plot_result(best_path, convergence_history, algo_name="GA-3D")

    end_time = time.time()  # 记录终点时间
    elapsed_time = end_time - start_time  # 计算总差值（单位是秒）
    
    print("-" * 50)
    print(f"本次单线程运行总耗时: {elapsed_time:.2f} 秒 (约 {elapsed_time / 60:.2f} 分钟)")
    print("-" * 50)
