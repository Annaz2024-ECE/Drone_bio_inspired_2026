import numpy as np
import time 
import math
from base_planner import BasePlanner

class GAPlanner(BasePlanner):
    def __init__(self, num_waypoints=10, max_iter=200, evaluator=None, pop_size=50, pc=0.8, pm=0.1):
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
        # 防止 main_coordinator_test.py 执行 setattr 时找不到属性报错
        self.emergency_escape = False 
        self.radar_guidance = False
        self.press_down = False       
        self.lift_up = False          
        self.apply_laplacian = False  
        self.apply_repulsion = False

        # 4. 动态提取 JSON 中的 3D 巡检目标点 (先验知识)
        self.target_anchors = []
        if hasattr(self.env, 'target_areas') and self.env.target_areas:
            for target in self.env.target_areas:
                # 提取 XY 坐标中心
                center_x, center_y = target['center'][:2]
                # 计算 3D 空间的中位安全打卡高度，默认取 4.0 到 8.0 之间的均值
                z_mid = (target.get('z_min', 4.0) + target.get('z_max', 8.0)) / 2.0
                self.target_anchors.append([center_x, center_y, z_mid])
        self.target_anchors = np.array(self.target_anchors)

    def _generate_target_guided_skeleton(self):
        """
        利用提取的巡检点，通过线性插值强制拉出一条 100% 踩点的官方引导骨架。
        """
        # 获取起终点（兼容 2D/3D 环境环境）
        start_p = self.env.start_point if hasattr(self.env, 'start_point') else [self.env.x_bounds[0], self.env.y_bounds[0]]
        end_p = self.env.end_point if hasattr(self.env, 'end_point') else [self.env.x_bounds[1], self.env.y_bounds[1]]
        
        start_3d = np.array(list(start_p) + [0.0]) if len(start_p) == 2 else np.array(start_p)
        end_3d = np.array(list(end_p) + [0.0]) if len(end_p) == 2 else np.array(end_p)

        # 按逻辑顺序拼接：起点 -> 所有巡检目标点 -> 终点
        if len(self.target_anchors) > 0:
            key_points = np.vstack([start_3d, self.target_anchors, end_3d])
        else:
            key_points = np.vstack([start_3d, end_3d])
        
        # 将空间关键锚点均匀插值到 num_waypoints 个中间控制点上
        t_key = np.linspace(0, 1, len(key_points))
        t_waypoints = np.linspace(0, 1, self.num_waypoints + 2)[1:-1] 
        
        guided_x = np.interp(t_waypoints, t_key, key_points[:, 0])
        guided_y = np.interp(t_waypoints, t_key, key_points[:, 1])
        guided_z = np.interp(t_waypoints, t_key, key_points[:, 2])
        
        guided_waypoints = np.column_stack((guided_x, guided_y, guided_z))
        
        # 展平为一维染色体基因
        return guided_waypoints.flatten()
    
    def _levy_step(self, dim):
        """
        使用经典 Mantegna 算法生成莱维飞行步长向量
        """
        beta = 1.5  # 特征指数
        
        # 计算 Mantegna 算法中的标准差 sigma_u
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        
        # 生成正态分布随机数
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        
        # 计算莱维步长
        step = u / (np.abs(v) ** (1 / beta))
        return step
    
    def _initialize_population(self):
        """ 初始化多梯队种群，实施分级分批的先验基因注入 """
        
        # 个体 0：注入【100%覆盖巡检点】的官方强制引导骨架 (最强业务王牌)
        self.positions[0] = self._generate_target_guided_skeleton()
        
        # 个体 1：注入基类的【常规启发式直线骨架】(保留直飞备用基因)
        self.positions[1] = self._generate_heuristic_skeleton()
        
        # 剩余个体实施生态策略分配
        for i in range(2, self.pop_size):
            if i < self.pop_size // 2:
                # 策略 A (前 50%)：在【官方巡检骨架】附近实施微小的高斯微调
                # 产生大体方向正确、但局部产生绕行的护卫梯队，专门负责微调探索避障盲区
                sigma = (self.ub - self.lb) * 0.03  # 3% 的微小扰动
                self.positions[i] = self.positions[0] + np.random.normal(0, sigma, self.dim)
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
            else:
                # 策略 B (后 50%)：全图完全随机撒点
                # 维持物种多样性，负责在盲区撞运气，防止算法早期产生局部定势
                self.positions[i] = np.random.uniform(self.lb, self.ub)

    def _decode_path(self, position):
        """
        【核心修正】：恢复连续空间有序路径解码。
        放弃最近邻重排，第 i 个基因严格对应第 i 个时间阶段的空间控制点。
        让算术交叉算子重新获得“继承优秀平滑轨迹”的数学逻辑。
        """
        # 直接按时序还原为 (num_waypoints, 3) 的 3D 坐标矩阵
        waypoints = position.reshape((self.num_waypoints, 3))
        # 头尾拼上起点和终点，直接返回完整航线
        return np.vstack([self.env.start_point, waypoints, self.env.end_point])
             
    def _tournament_selection(self):
        """ 锦标赛选择算子：随机挑选两个互相竞争，保留适应度更低(更好)的个体 """
        new_positions = np.zeros_like(self.positions)
        for i in range(self.pop_size):
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            winner = idx1 if self.fitness[idx1] < self.fitness[idx2] else idx2
            new_positions[i] = self.positions[winner]
        self.positions = new_positions

    def _arithmetic_crossover(self):
        """ 算术交叉算子：完美契合连续空间时序基因，生成父母之间的平滑中间路径 """
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size and np.random.rand() < self.pc:
                # 随机生成各个维度的插值系数向量 alpha
                alpha = np.random.rand(self.dim)
                
                p1 = self.positions[i].copy()
                p2 = self.positions[i+1].copy()
                
                # 双向线性基因融合
                self.positions[i] = alpha * p1 + (1 - alpha) * p2
                self.positions[i+1] = alpha * p2 + (1 - alpha) * p1
                
                # 边界约束截断
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                self.positions[i+1] = np.clip(self.positions[i+1], self.lb, self.ub)

    def _gaussian_mutation(self):
        """ 高斯变异算子：对连续空间航点坐标进行局部扰动 """
        for i in range(self.pop_size):
            if np.random.rand() < self.pm:
                # 以 5% 的边界全幅作为标准差生成高斯噪声进行航线打磨
                sigma = (self.ub - self.lb) * 0.05
                mutation_step = np.random.normal(0, sigma, self.dim)
                
                self.positions[i] += mutation_step
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

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
            # 1.1 评估种群适应度 (单线程顺序执行，碰撞检测极其精准)
            for i in range(self.pop_size):
                path = self._decode_path(self.positions[i])
                
                # 调用黑盒评价器进行 3D 物理与业务计分
                score, details, env_info = self.evaluator.evaluate_pso_particle(path)
                self.fitness[i] = score
                
                # 全局历史最优记录更新
                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = self.positions[i].copy()
                    best_details_for_coord = details
                    best_env_info_for_coord = env_info

            # 精英保留策略 (Elitism)：强制保留历史第一的王牌，防止变异导致优秀退化
            # 我们将当前种群中最差的一个人，替换成历史最优个体
            worst_idx = np.argmax(self.fitness)
            self.positions[worst_idx] = self.historical_best_pos.copy()
            self.fitness[worst_idx] = self.historical_best_score

            self.convergence_curve.append(self.historical_best_score)

            # 2. 核心大招：无缝切脉对接 CoordinatorAgent (老中医实时调参)
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
                
                # 将宏观放宽限制等调参指令同步更新给底层评价器
                self.evaluator.update_params(new_params=eval_params)
                
                # 如果老中医觉得得分已经完美（如达到0分或不再收敛），直接提前出关
                if is_finished:
                    break

            # 3. 执行通用物理机制 (调用基类的拉普拉斯平滑、障碍物斥力武器)
            self.execute_universal_physics_directives()

            # 4. GA 仿生学遗传演化 (锦标赛选择 -> 算术交叉 -> 高斯突变)
            self._tournament_selection()
            self._arithmetic_crossover()
            self._gaussian_mutation()
        
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
    planner = GAPlanner(num_waypoints=30, pop_size=20, max_iter=500, pc=0.85, pm=0.2)
    
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