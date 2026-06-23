import numpy as np
import matplotlib.pyplot as plt
import os

from environment_buildup import UAVEnvironment2D
from path_evaluator import PathEvaluator

class SSAPlanner:
    def __init__(self, num_sparrows=100, num_waypoints=8, max_iter=150):
        """
        初始化 SSA (麻雀搜索算法) 路径规划器
        :param num_sparrows: 种群大小（麻雀数量）
        :param num_waypoints: 每条路径的中间控制点数量
        :param max_iter: 最大迭代次数
        """
        self.evaluator = PathEvaluator()
        self.env = self.evaluator.env
        
        self.num_sparrows = num_sparrows
        self.num_waypoints = num_waypoints
        self.max_iter = max_iter
        
        # SSA 核心参数 (基于经典麻雀算法设定)
        self.PD = 0.2  # 发现者 (Producers) 比例，通常为 20%
        self.SD = 0.1  # 侦察者/警戒者 (Scouts) 比例，通常为 10%
        self.ST = 0.8  # 安全阈值 (Safety Threshold)，通常在 [0.5, 0.9] 之间
        
        self.num_producers = int(self.num_sparrows * self.PD)
        self.num_scouts = int(self.num_sparrows * self.SD)
        
        # 边界约束
        self.x_bounds = self.env.x_bounds
        self.y_bounds = self.env.y_bounds
        
        # 初始化麻雀位置
        self.sparrows = self._initialize_sparrows()
        self.fitness = np.full(num_sparrows, np.inf)
        
        # 记录全局最优
        self.gbest_pos = None
        self.gbest_score = np.inf

    def _initialize_sparrows(self):
        """ 初始化 在起点和终点的连线附近生成随机点 """
        start = self.env.start_point
        end = self.env.end_point
        sparrows = np.zeros((self.num_sparrows, self.num_waypoints, 2))
        
        for i in range(self.num_sparrows):
            # 在起点和终点之间均匀取点
            x_vals = np.linspace(start[0], end[0], self.num_waypoints + 2)[1:-1]
            y_vals = np.linspace(start[1], end[1], self.num_waypoints + 2)[1:-1]
            
            # 【改进】加大初始扰动 (噪音)，确保麻雀群能覆盖地图两侧，绕开中间大障碍物
            noise_x = np.random.uniform(-40, 40, self.num_waypoints)
            noise_y = np.random.uniform(-40, 40, self.num_waypoints)
            
            # 限制在地图边界内
            sparrows[i, :, 0] = np.clip(x_vals + noise_x, self.x_bounds[0], self.x_bounds[1])
            sparrows[i, :, 1] = np.clip(y_vals + noise_y, self.y_bounds[0], self.y_bounds[1])
            
        return sparrows

    def construct_full_path(self, sparrow):
        """ 将起点、中间控制点、终点拼接成一条完整的路径序列 """
        path = [self.env.start_point]
        path.extend(sparrow)
        path.append(self.env.end_point)
        return np.array(path)

    def optimize(self):
        """ SSA 主循环 """
        best_scores_history = []
        
        print("开始 SSA 麻雀搜索算法路径规划迭代...")
        
        # 初始评估
        for i in range(self.num_sparrows):
            full_path = self.construct_full_path(self.sparrows[i])
            # 【核心修改】调用 evaluate_pso_particle，内部会生成 Chaikin 曲线并计算综合适应度
            self.fitness[i] = self.evaluator.evaluate_pso_particle(full_path)
            if self.fitness[i] < self.gbest_score:
                self.gbest_score = self.fitness[i]
                self.gbest_pos = np.copy(self.sparrows[i])
                
        for iteration in range(self.max_iter):
            # 获取当前种群中适应度排序的索引
            sort_indices = np.argsort(self.fitness)
            best_idx_current = sort_indices[0]
            worst_idx_current = sort_indices[-1]
            
            best_pos_current = np.copy(self.sparrows[best_idx_current])
            worst_pos_current = np.copy(self.sparrows[worst_idx_current])
            best_fit_current = self.fitness[best_idx_current]
            worst_fit_current = self.fitness[worst_idx_current]
            
            new_sparrows = np.copy(self.sparrows)
            
            # ==========================================
            # 1. 发现者 (Producers) 更新位置
            # ==========================================
            R2 = np.random.rand() # 预警值
            for i in range(self.num_producers):
                idx = sort_indices[i]
                if R2 < self.ST:
                    # 【防(0,0)坍缩】
                    alpha = np.random.rand()
                    step = np.random.randn(*self.sparrows[idx].shape) * 15.0 
                    new_sparrows[idx] = self.sparrows[idx] + step * np.exp(-(iteration + 1) / (alpha * self.max_iter + 1e-8))
                else:
                    # 发现危险：随机游走撤离
                    Q = np.random.randn(*self.sparrows[idx].shape) * 2.0
                    new_sparrows[idx] = self.sparrows[idx] + Q
                    
            # ==========================================
            # 2. 加入者 (Scroungers) 更新位置
            # ==========================================
            for i in range(self.num_producers, self.num_sparrows):
                idx = sort_indices[i]
                if i > self.num_sparrows / 2:
                    # 【全图重生，保证多样性】
                    new_x = np.random.uniform(self.x_bounds[0], self.x_bounds[1], self.num_waypoints)
                    new_y = np.random.uniform(self.y_bounds[0], self.y_bounds[1], self.num_waypoints)
                    new_sparrows[idx, :, 0] = new_x
                    new_sparrows[idx, :, 1] = new_y
                else:
                    # 适应度较好的加入者，在全局最优解附近觅食
                    A = np.random.choice([-1, 1], size=(self.num_waypoints, 2))
                    new_sparrows[idx] = best_pos_current + np.abs(self.sparrows[idx] - best_pos_current) * (A / 2.0)
                    
            # ==========================================
            # 3. 侦察者/警戒者 (Scouts) 更新位置
            # ==========================================
            scout_indices = np.random.choice(self.num_sparrows, self.num_scouts, replace=False)
            for idx in scout_indices:
                if self.fitness[idx] > best_fit_current:
                    # 处于边缘的麻雀，向最优位置靠拢
                    beta = np.random.randn(*self.sparrows[idx].shape)
                    new_sparrows[idx] = best_pos_current + beta * np.abs(self.sparrows[idx] - best_pos_current)
                else:
                    # 最优附近的麻雀随机逃窜
                    K = np.random.uniform(-1, 1)
                    new_sparrows[idx] = self.sparrows[idx] + K * (np.abs(self.sparrows[idx] - worst_pos_current) / (self.fitness[idx] - worst_fit_current + 1e-8))

            # 4. 边界约束及适应度更新
            for i in range(self.num_sparrows):
                # 强制边界约束 (防止飞出地图)
                new_sparrows[i, :, 0] = np.clip(new_sparrows[i, :, 0], self.x_bounds[0], self.x_bounds[1])
                new_sparrows[i, :, 1] = np.clip(new_sparrows[i, :, 1], self.y_bounds[0], self.y_bounds[1])
                
                # 重新评估适应度并更新历史全局最优
                full_path = self.construct_full_path(new_sparrows[i])
                # 【核心修改】调用 evaluate_pso_particle
                score = self.evaluator.evaluate_pso_particle(full_path)
                
                # 更新自身
                self.sparrows[i] = new_sparrows[i]
                self.fitness[i] = score
                
                # 更新全局最优
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos = np.copy(new_sparrows[i])
                    
            best_scores_history.append(self.gbest_score)
            
            # 每 50 代打印一次进度
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 当前最优得分: {self.gbest_score:.2f}")
                
        print(f"规划完成！最终得分: {self.gbest_score:.2f}")
        
        # 【新增：生成完美展示用的平滑路径】
        best_raw_path = self.construct_full_path(self.gbest_pos)
        # 迭代 4 次生成极度丝滑的真实飞行轨迹，用于最终画图
        smooth_best_path = self.evaluator.generate_chaikin_path(best_raw_path, iterations=4)
        
        return best_raw_path, smooth_best_path, best_scores_history

    def plot_result(self, raw_path, smooth_path, score_history, run_idx=None, save_dir=None):
        """ 
        绘制结果：左图为地图路径，右图为收敛曲线
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 绘制地图和路径
        self.env.draw_environment(ax=ax1)
        
        # 【新增】画出底层的控制点 (用灰色虚线和叉叉表示)
        ax1.plot(raw_path[:, 0], raw_path[:, 1], color='gray', linestyle='--', 
                 linewidth=1.5, marker='x', markersize=6, label='SSA Control Waypoints')
                 
        # 【新增】画出平滑后的真实飞行曲线 (亮粉色实线)
        ax1.plot(smooth_path[:, 0], smooth_path[:, 1], color='#e91e63', 
                 linewidth=3, label='Chaikin Smooth Path')
        
        # 在地图左上角显示最终得分
        final_score = score_history[-1]
        ax1.text(0.02, 0.98, f'Final Score: {final_score:.2f}', 
                 transform=ax1.transAxes, fontsize=16, fontweight='bold', 
                 color='#d32f2f', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#d32f2f'))
                 
        ax1.legend(loc='lower right')
        
        # 2. 绘制收敛曲线
        ax2.plot(score_history, color='#9c27b0', linewidth=2)
        title = 'SSA Convergence Curve'
        if run_idx is not None:
            title += f' (Run {run_idx})'
            
        ax2.set_title(title, fontweight='bold', fontsize=14)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Fitness Score (Lower is better)', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图像逻辑
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"ssa_run_{run_idx:02d}.png" if run_idx else "ssa_result.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"✅ 图像已保存至: {filepath}")
            plt.close(fig) # 关闭图像，防止内存泄漏和弹窗卡死
        else:
            plt.show()

# ==========================================
# 批量运行测试代码
# ==========================================
if __name__ == "__main__":
    log_directory = "SSA_log"
    num_runs = 10
    
    print("=" * 50)
    print(f"🚀 开始 SSA 算法批量测试，共计 {num_runs} 次运行")
    print(f"📁 结果将保存在当前目录的 '{log_directory}' 文件夹下")
    print("=" * 50)
    
    for current_run in range(1, num_runs + 1):
        print(f"\n--- 正在进行第 {current_run}/{num_runs} 次运行 ---")
        
        # 参数: 种群100, 航点8个, 迭代200次
        planner = SSAPlanner(num_sparrows=100, num_waypoints=8, max_iter=200)
        
        # 运行优化 (现在返回三个值：原始控制点、平滑曲线、历史得分)
        raw_path, smooth_path, history = planner.optimize()
        
        # 可视化展示并保存到文件夹
        planner.plot_result(raw_path, smooth_path, history, run_idx=current_run, save_dir=log_directory)
        
    print("\n🎉 所有批量运行已完成！请前往 SSA_log 文件夹查看结果。")