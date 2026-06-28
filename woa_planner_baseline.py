from path_evaluator import PathEvaluator
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from base_planner import BasePlanner

class WOAPlanner:
    def __init__(self, num_waypoints=20, pop_size=60, max_iter=150):
        self.evaluator = PathEvaluator()  
        self.env = self.evaluator.env
        
        self.num_waypoints = num_waypoints
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.convergence_curve = []
        # 【核心改变 1】算法优化的维度等于中间航点数（每个点只需优化一个垂直于基准线的偏移量）
        self.dim = self.num_waypoints 
        
        # 基础坐标向量计算
        self.start = np.array(self.env.start_point)
        self.end = np.array(self.env.end_point)
        self.base_vec = self.end - self.start
        self.base_len = np.linalg.norm(self.base_vec)
        self.dir_vec = self.base_vec / self.base_len  # 前进方向单位向量
        
        # 计算法向量（垂直于前进方向，向左为正，用于控制偏移）
        self.normal_vec = np.array([-self.dir_vec[1], self.dir_vec[0]])
        
        # 在基准线上，预先计算好绝对均匀递增的“脚印点”作为基准位置
        self.t_vals = np.linspace(0, 1, self.num_waypoints + 2)[1:-1]
        self.base_points = np.array([self.start + t * self.base_vec for t in self.t_vals])
        
        # 定义一维偏移量的上下界（比如最大允许偏离基准线 50 个单位）
        self.max_offset = 50.0 
        self.lb = np.array([-self.max_offset] * self.dim)
        self.ub = np.array([self.max_offset] * self.dim)

    def _decode_path(self, position):
        """
        【核心改变 2】解码函数：将一维偏移量转换为无序断裂风险的绝对二维坐标
        position: 长度为 dim 的一维数组，代表各中间航点的法向偏移距离
        """
        offsets = position[:, np.newaxis]  # 变形为 (N, 1) 方便广播矩阵运算
        
        # 绝对二维空间坐标 = 基准点 + 偏移量 * 法向量方向
        waypoints = self.base_points + offsets * self.normal_vec
        
        # 【物理硬裁剪】限制航点绝不飞出地图边界，保留安全边距
        waypoints[:, 0] = np.clip(waypoints[:, 0], self.env.x_bounds[0] + 2.0, self.env.x_bounds[1] - 2.0)
        waypoints[:, 1] = np.clip(waypoints[:, 1], self.env.y_bounds[0] + 2.0, self.env.y_bounds[1] - 2.0)
        
        # 首尾拼接起点和终点，天生具备完美的空间单向流动性，无需任何排序！
        path = [self.start] + waypoints.tolist() + [self.end]
        return np.array(path)

    def _initialize_population(self):
        """
        【核心改变 3】基于偏移量空间的特异性三批种群初始化
        """
        positions = np.zeros((self.pop_size, self.dim))
        tier_size = self.pop_size // 3
        
        # 计算西区和东区中心点，相对于基准线所需的垂直偏移量
        west_center = np.array(self.env.target_areas[0]['center'])
        east_center = np.array(self.env.target_areas[1]['center'])
        
        # 向量投影：计算点到基准线的法向垂距
        west_offset = np.dot(west_center - self.start, self.normal_vec)
        east_offset = np.dot(east_center - self.start, self.normal_vec)
        
        for i in range(self.pop_size):
            if i < tier_size:
                # 第一批：中路直飞型，偏移量围绕 0 随机抖动
                positions[i, :] = np.random.uniform(-5.0, 5.0, self.dim)
            elif i < 2 * tier_size:
                # 第二批：西区探路型，偏移量围绕西区投影中心抖动
                positions[i, :] = np.random.uniform(west_offset - 15.0, west_offset + 15.0, self.dim)
            else:
                # 第三批：东区探路型，偏移量围绕东区投影中心抖动
                positions[i, :] = np.random.uniform(east_offset - 15.0, east_offset + 15.0, self.dim)
                
        return np.clip(positions, self.lb, self.ub)

    def optimize(self):
        # 初始化一维偏移量种群
        positions = self._initialize_population()
        
        best_score = float('inf')
        best_position = np.zeros(self.dim)
        score_history = []

        print("开始鲸鱼优化算法路径规划（基准线投影模式）...")
        
        for t in range(self.max_iter):
            # 1. 评估所有个体
            for i in range(self.pop_size):
                path = self._decode_path(positions[i, :])
                score,_ , _ = self.evaluator.evaluate_pso_particle(path)
                
                if score < best_score:
                    best_score = score
                    best_position = positions[i, :].copy()
            
            # 2. 融入【余弦衰减策略】减缓参数 a 的前期衰减速度
            a = 2.0 * math.cos((math.pi / 2.0) * (t / self.max_iter))
            
            # 3. 鲸鱼位置更新（完全在一维偏移量标准空间内算数，不再有纵横错位）
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                
                p = np.random.rand()
                b = 1.0
                l = np.random.uniform(-1, 1)
                
                if p < 0.5:
                    if abs(A) < 1:
                        # 包围捕食机制
                        D = abs(C * best_position - positions[i, :])
                        positions[i, :] = best_position - A * D
                    else:
                        # 随机寻道机制
                        rand_idx = np.random.randint(0, self.pop_size)
                        X_rand = positions[rand_idx, :]
                        D = abs(C * X_rand - positions[i, :])
                        positions[i, :] = X_rand - A * D
                else:
                    # 气泡网捕食机制（螺旋更新）
                    D_prime = abs(best_position - positions[i, :])
                    positions[i, :] = D_prime * math.exp(b * l) * math.cos(2.0 * math.pi * l) + best_position
            
            # 4. 边界硬修剪（直接在一维空间限制最大偏离范围）
            positions = np.clip(positions, self.lb, self.ub)
            self.convergence_curve.append(best_score)

            if (t + 1) % 50 == 0 or t == self.max_iter - 1 or t == 0:
                print(f"迭代进度: {t+1}/{self.max_iter} | 当前最优全路径得分: {best_score:.2f}")
        
        # 最终解码得到最优的二维航点序列
        final_path = self._decode_path(best_position)
        return best_score, final_path, self.convergence_curve

    

# =====================================================================
# 3. 测试运行入口
# =====================================================================
if __name__ == "__main__":
    # 实例化规划器（20个航点，60只鲸鱼，150次迭代）
    planner = WOAPlanner(num_waypoints=20, pop_size=60, max_iter=150)
    
    best_score, final_path, score_history = planner.optimize()
    
    print("\n规划完成！最终得分（长度+碰撞惩罚）:", round(best_score, 2))
    print("最终生成的全向丝滑航点轨迹坐标：")
    # 设置 numpy 打印精度
    np.set_printoptions(suppress=True, precision=4)
    print(final_path)
    bp=BasePlanner()
    bp.plot_result(final_path, score_history)
