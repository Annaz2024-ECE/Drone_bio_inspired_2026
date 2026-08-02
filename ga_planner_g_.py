import numpy as np
import time 
import math
from base_planner import BasePlanner

class GAPlanner(BasePlanner):
    def __init__(self, num_waypoints=10, max_iter=200, evaluator=None, pop_size=50, pc=0.85, pm=0.50):
        """
        3D 连续空间遗传算法 (实数编码 - 满血联动版)
        """
        # 1. 继承 BasePlanner 初始化环境与评价器
        super().__init__(num_waypoints, max_iter, evaluator)
        
        # 2. GA 专属核心参数
        self.pop_size = pop_size
        self.pc = pc  # 交叉概率 (Crossover rate)
        self.pm = pm  # 变异概率 (Mutation rate)
        
        # 初始化种群位置与适应度
        self.positions = np.zeros((self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float("inf"))
        
        # 3. 显式声明 Agent（老中医）会动态篡改的专属参数与物理开关
        self.emergency_escape = False 
        self.radar_guidance = False
        self.press_down = False       
        self.lift_up = False          
        self.apply_laplacian = False  
        self.apply_repulsion = False

        # 4. 动态提取 JSON 中的 3D 巡检目标点 (先验知识)
        target_anchors_list = []
        if hasattr(self.env, 'target_areas') and self.env.target_areas:
            for target in self.env.target_areas:
                center_x, center_y = target['center'][:2]
                z_mid = (target.get('z_min', 4.0) + target.get('z_max', 8.0)) / 2.0
                target_anchors_list.append([center_x, center_y, z_mid])
                
        self.target_anchors = np.array(target_anchors_list) if len(target_anchors_list) > 0 else np.empty((0, 3))
    
    def _get_topologically_sorted_anchors(self):
        """ 按照最近邻 (Nearest Neighbor) 对打卡点进行空间拓扑排序，消除折返 """
        if len(self.target_anchors) == 0:
            return np.empty((0, 3))
            
        # 兼容性获取起点坐标
        if hasattr(self.env, 'start_point'):
            start_pos = np.array(self.env.start_point)
        elif hasattr(self.env, 'start_pos'):
            sp = self.env.start_pos
            start_pos = np.array([sp['x'], sp['y'], sp['z']]) if isinstance(sp, dict) else np.array(sp)
        else:
            start_pos = np.zeros(3)

        unvisited = list(self.target_anchors.copy())
        sorted_anchors = []
        
        curr = start_pos
        while unvisited:
            dists = [np.linalg.norm(np.array(p) - curr) for p in unvisited]
            nearest_idx = np.argmin(dists)
            nearest_node = unvisited.pop(nearest_idx)
            sorted_anchors.append(nearest_node)
            curr = np.array(nearest_node)
            
        return np.array(sorted_anchors)

    def _initialize_population(self):
        """ 多梯队种群初始化：精英骨架 + 拓扑注入 + 全域随机探索 """
        
        # 1. 个体 0：注入基础骨架（TSP + 线性插值保底）
        self.positions[0] = self._generate_basic_skeleton()
        
        # 2. 准备打卡点拓扑排序与分布插槽
        sorted_anchors = self._get_topologically_sorted_anchors()
        num_anchors = len(sorted_anchors)
        
        # 计算打卡点分配索引（防止插槽溢出）
        if num_anchors > 0 and self.num_waypoints > 2:
            usable_slots = self.num_waypoints - 2
            anchor_indices = np.linspace(1, usable_slots, min(num_anchors, usable_slots), dtype=int)
        else:
            anchor_indices = np.array([], dtype=int)

        top_30_count = int(self.pop_size * 0.3)
        sigma = (self.ub - self.lb) * 0.05

        # 3. 多梯队种群分配
        for i in range(1, self.pop_size):
            if i < top_30_count:
                # 梯队 A (1 ~ 30%)：基于骨架微调 + 强行拓扑打卡点注入 (局部精修)
                noisy_pos = self.positions[0] + np.random.normal(0, sigma, self.dim)
                waypoints = noisy_pos.reshape(self.num_waypoints, 3)
                
                if len(anchor_indices) > 0:
                    for idx, anchor in zip(anchor_indices, sorted_anchors[:len(anchor_indices)]):
                        waypoints[idx] = anchor
                        
                self.positions[i] = np.clip(waypoints.flatten(), self.lb, self.ub)
            else:
                # 梯队 B (30% ~ 100%)：全图均匀随机撒点，保留全局探索能力，避免早熟/跳不出障碍物
                self.positions[i] = np.random.uniform(self.lb, self.ub, self.dim)

    def _tournament_selection(self):
        """ 锦标赛选择算子：随机挑选两个互相竞争，保留适应度更低(更好)的个体 """
        new_positions = np.zeros_like(self.positions)
        for i in range(self.pop_size):
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            winner = idx1 if self.fitness[idx1] < self.fitness[idx2] else idx2
            new_positions[i] = self.positions[winner]
        self.positions = new_positions

    def _arithmetic_crossover(self):
        """ 算术交叉算子：连续空间基因融合 """
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size and np.random.rand() < self.pc:
                alpha = np.random.rand(self.dim)
                p1 = self.positions[i].copy()
                p2 = self.positions[i+1].copy()
                
                self.positions[i] = alpha * p1 + (1 - alpha) * p2
                self.positions[i+1] = alpha * p2 + (1 - alpha) * p1
                
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                self.positions[i+1] = np.clip(self.positions[i+1], self.lb, self.ub)

    def _hybrid_mutation(self):
        """ 带有保护机制与雷达引力拉回的混合变异算子 """
        for i in range(self.pop_size):
            if np.random.rand() < self.pm:
                # 逃生/高变异率 -> 莱维飞行大跨步；常态 -> 高斯微调
                if getattr(self, 'emergency_escape', False) or self.pm > 0.2:
                    levy_step = self._levy_step(self.dim)
                    scale = (self.ub - self.lb) * getattr(self, 'levy_scale', 0.05)
                    mutation_step = levy_step * scale
                else:
                    sigma = (self.ub - self.lb) * 0.03
                    mutation_step = np.random.normal(0, sigma, self.dim)
                    
                self.positions[i] += mutation_step
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                # 雷达空投/漏打卡强力引力修正
                if getattr(self, 'radar_guidance', False) and len(self.target_anchors) > 0:
                    waypoints = self.positions[i].reshape(self.num_waypoints, 3)
                    target = self.target_anchors[np.random.choice(len(self.target_anchors))]
                    dists = np.linalg.norm(waypoints - target, axis=1)
                    closest_wpt_idx = np.argmin(dists)
                    waypoints[closest_wpt_idx] = 0.5 * waypoints[closest_wpt_idx] + 0.5 * target
                    self.positions[i] = waypoints.flatten()

    def optimize(self, callback=None):
        """
        核心演化循环
        :param callback: 用于对接 CoordinatorAgent (老中医) 的决策上报回调函数
        """
        # 1. 触发多梯队种群注入初始化
        self._initialize_population()
        
        best_details_for_coord = None
        best_env_info_for_coord = None

        for t in range(self.max_iter):
            # 1.1 评估种群适应度
            for i in range(self.pop_size):
                path = self._decode_path(self.positions[i])
                
                # 调用黑盒评价器进行 3D 物理与业务计分
                score, details, env_info = self.evaluator.evaluate_particle(path)
                self.fitness[i] = score
                
                # 全局历史最优记录更新
                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = self.positions[i].copy()
                    best_details_for_coord = details
                    best_env_info_for_coord = env_info

            # 精英保留策略 (Elitism)：替换最差个体为历史最优
            worst_idx = np.argmax(self.fitness)
            self.positions[worst_idx] = self.historical_best_pos.copy()
            self.fitness[worst_idx] = self.historical_best_score

            self.convergence_curve.append(self.historical_best_score)

            # 2. 对接 CoordinatorAgent 调参
            if callback and best_details_for_coord is not None:
                algo_params, eval_params, specific_params, is_finished = callback(
                    self.historical_best_score, 
                    best_details_for_coord, 
                    best_env_info_for_coord, 
                    "GA"
                )
                
                # 接收老中医动态篡改的 GA 变异交叉率
                self.pc = specific_params.get('pc', self.pc)
                self.pm = specific_params.get('pm', self.pm)
                
                # 接收基类物理重力场/斥力场算子控制指令
                self.apply_laplacian = specific_params.get('apply_laplacian', False)
                self.apply_repulsion = specific_params.get('apply_repulsion', False)
                
                # 将调参指令同步更新给底层评价器
                self.evaluator.update_params(new_params=eval_params)
                
                if is_finished:
                    break

            # 3. 执行通用物理机制
            self.execute_universal_physics_directives()

            # 4. GA 仿生学遗传演化 (锦标赛选择 -> 算术交叉 -> 混合变异)
            self._tournament_selection()
            self._arithmetic_crossover()
            self._hybrid_mutation()
        
            # 5. 定期打印收敛进度
            if (t + 1) % 50 == 0 or t == 0:
                print(f"代数: {t + 1}/{self.max_iter}, 当前最优得分: {self.historical_best_score:,.2f}")
                
        # 返回解码后的最终完美 3D 轨迹及历史收敛数据
        best_full_path = self._decode_path(self.historical_best_pos)
        return best_full_path, self.convergence_curve
    
if __name__ == "__main__":
    # 单体独立测试入口
    start_time = time.time()
    
    # 实例化规划器
    planner = GAPlanner(num_waypoints=30, pop_size=40, max_iter=100, pc=0.85, pm=0.2)
    
    # 运行单体优化 (此时无 callback，跑纯粹的 GA 演化)
    best_path, convergence_history = planner.optimize()
    
    print(f"\nGA 独立规划完成！最终得分: {convergence_history[-1]:,.2f}")
    
    # 调用父类绘图组件呈现最终路线
    planner.plot_result(best_path, convergence_history, algo_name="GA-3D-Sequential")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 50)
    print(f"本次运行总耗时: {elapsed_time:.2f} 秒 (约 {elapsed_time / 60:.2f} 分钟)")
    print("-" * 50)