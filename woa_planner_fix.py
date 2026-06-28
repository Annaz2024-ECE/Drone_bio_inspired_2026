import numpy as np
import math
import random
import matplotlib.pyplot as plt
from base_planner import BasePlanner
from path_evaluator import PathEvaluator

class WOAPlannerArg(BasePlanner):
    def __init__(self, evaluator=None, num_waypoints=6, pop_size=50, max_iter=150):
        """
        初始化 WOA 路径规划器 按照y轴升序解码 初始化分成东部中部西部三路 
        :param num_waypoints: 起点和终点之间的中间控制点数量
        :param pop_size: 鲸鱼种群规模
        :param max_iter: 最大迭代次数
        """
        # === 新增：调用父类初始化 ===
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.pop_size = pop_size

        # __init__ 中添加
        self.safe_margin = 3.0   # 或更大，如 5.0
        self.effective_lb = self.lb + self.safe_margin
        self.effective_ub = self.ub - self.safe_margin
        # 提取精英数量
        self.top_30_percent = int(self.pop_size * 0.3)

    def _decode_path(self, position):
        # 重写基类方法
        waypoints = position.reshape((self.num_waypoints, 2))
        # 按照 Y 轴坐标从小到大排序控制点
        waypoints = waypoints[np.argsort(waypoints[:, 1])] 
        full_path = np.vstack([self.env.start_point, waypoints, self.env.end_point])
        return full_path

    def _initialize_population(self):
        positions = np.zeros((self.pop_size, self.dim))
        
        west_center = self.env.target_areas[0]['center']
        east_center = self.env.target_areas[1]['center']
        tier_size = self.pop_size // 3
        
        max_retries = 20    # 每个个体最多重试 20 次
        
        for i in range(self.pop_size):
            # 生成基础插值点
            if i < tier_size:
                x_vals = np.linspace(self.env.start_point[0], self.env.end_point[0], self.num_waypoints + 2)[1:-1].tolist()
                y_vals = np.linspace(self.env.start_point[1], self.env.end_point[1], self.num_waypoints + 2)[1:-1].tolist()
            elif i < 2 * tier_size:
                mid_idx = self.num_waypoints // 2
                x_vals1 = np.linspace(self.env.start_point[0], west_center[0], mid_idx + 2)[1:].tolist()
                y_vals1 = np.linspace(self.env.start_point[1], west_center[1], mid_idx + 2)[1:].tolist()
                rem_points = self.num_waypoints - len(x_vals1)
                x_vals2 = np.linspace(west_center[0], self.env.end_point[0], rem_points + 2)[1:-1].tolist()
                y_vals2 = np.linspace(west_center[1], self.env.end_point[1], rem_points + 2)[1:-1].tolist()
                x_vals = x_vals1 + x_vals2
                y_vals = y_vals1 + y_vals2
            else:
                mid_idx = self.num_waypoints // 2
                x_vals1 = np.linspace(self.env.start_point[0], east_center[0], mid_idx + 2)[1:].tolist()
                y_vals1 = np.linspace(self.env.start_point[1], east_center[1], mid_idx + 2)[1:].tolist()
                rem_points = self.num_waypoints - len(x_vals1)
                x_vals2 = np.linspace(east_center[0], self.env.end_point[0], rem_points + 2)[1:-1].tolist()
                y_vals2 = np.linspace(east_center[1], self.env.end_point[1], rem_points + 2)[1:-1].tolist()
                x_vals = x_vals1 + x_vals2
                y_vals = y_vals1 + y_vals2
            
            base = np.column_stack((x_vals, y_vals)).flatten()
            
            # 带安全边界检查的噪声添加
            valid = False
            for retry in range(max_retries):
                noise = np.random.uniform(-10, 10, self.dim) if i < tier_size else np.random.uniform(-5, 5, self.dim)
                candidate = base + noise
                candidate = np.clip(candidate, self.lb, self.ub)
                
                if np.all(candidate >= self.effective_lb) and np.all(candidate <= self.effective_ub):
                    valid = True
                    break
            
            if not valid:
                candidate = np.clip(candidate, effective_lb, effective_ub)
            
            # Y 轴排序同步
            candidate_pts = candidate.reshape((self.num_waypoints, 2))
            
            
            # === 新增：强制前 30% 的个体，其第2个和第4个控制点踩在固定目标上 ===
            # (对应 1D 数组的 index 2,3 和 6,7)
            if i < self.top_30_percent and self.num_waypoints >= 4:
                candidate_pts[0] = [22.0, 40.0] 
                candidate_pts[1] = [78.0, 70.0] 
            
            positions[i, :] = candidate_pts.flatten() # 存入有序的基因
        
        return positions
      
    
    def optimize(self):
        """
        执行鲸鱼优化算法的主循环
        """
        print("开始 WOA 鲸鱼优化算法路径规划...")
        
        # 1. 初始化种群
        positions = self._initialize_population()

        # 计算初始种群适应度
        for i in range(self.pop_size):
            path = self._decode_path(positions[i, :])
            score, _ = self.evaluator.evaluate_pso_particle(path)
            if score < self.historical_best_score:
                self.historical_best_score = score
                self.historical_best_pos = positions[i, :].copy()

        # 2. WOA 主循环
        for t in range(self.max_iter):
            k = 0.5  # 引入非线性衰减
            a = 2.0 * ((1.0 - t / self.max_iter) ** k)
            
            for i in range(self.pop_size):
                r1 = random.random()
                r2 = random.random()
                
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = random.random()
                b = 1.0
                l = random.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) >= 1:
                        # 随机寻找猎物 (探索全局)
                        rand_idx = random.randint(0, self.pop_size - 1)
                        rand_pos = positions[rand_idx, :]
                        D_x_rand = abs(C * rand_pos - positions[i, :])
                        new_pos = rand_pos - A * D_x_rand
                    else:
                        # 包围猎物 (局部寻优)
                        D_Leader = abs(C * self.historical_best_pos - positions[i, :])
                        new_pos = self.historical_best_pos - A * D_Leader
                else:
                    # 气泡网攻击 (螺旋更新)
                    D_Leader = abs(self.historical_best_pos - positions[i, :])
                    new_pos = D_Leader * math.exp(b * l) * math.cos(2 * math.pi * l) + self.historical_best_pos

                # 边界处理 6/25改
                clipped_pos = np.clip(new_pos, self.effective_lb, self.effective_ub)

                # 位置更新后，立刻同步 Y 轴排序，防止基因错位
                waypoints_temp = clipped_pos.reshape((self.num_waypoints, 2))
                
                # 【修复逻辑冲突】先替换锚定点，再执行Y轴排序 6/27
                if i < self.top_30_percent and self.num_waypoints >= 4:
                    waypoints_temp[0] = [22.0, 40.0]
                    waypoints_temp[1] = [78.0, 70.0]

                # 让排序机制自动帮我们把巡检点归入合理的顺序
                waypoints_temp = waypoints_temp[np.argsort(waypoints_temp[:, 1])] 

                positions[i, :] = waypoints_temp.flatten()

                # 评估新位置
                path = self._decode_path(positions[i, :])
                score, _ = self.evaluator.evaluate_pso_particle(path)

                # 更新全局最优
                if score < self.historical_best_score and np.any(positions[i,:]):
                    self.historical_best_score = score
                    self.historical_best_pos = positions[i, :].copy()

            self.convergence_curve.append(self.historical_best_score)
            
            # 控制台输出进度 (每 50 次迭代)
            if (t + 1) % 50 == 0 or t == 0:
                print(f"迭代次数: {t + 1}/{self.max_iter}, 当前最优得分 (适应度): {self.historical_best_score:.2f}")

        return self._decode_path(self.historical_best_pos), self.convergence_curve


# ================= 运行测试 =================
if __name__ == "__main__":
    # 实例化规划器，可适当增加控制点数量以绕开复杂障碍
    planner = WOAPlannerArg(num_waypoints=15, pop_size=60, max_iter=200)
    
    best_path, convergence_history = planner.optimize()
    
    print(f"\n规划完成！最终得分: {convergence_history[-1]:.2f}")
    print(best_path)
    
    planner.plot_result(best_path, convergence_history, algo_name="WOA-Ysort")