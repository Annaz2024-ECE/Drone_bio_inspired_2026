import numpy as np
import matplotlib.pyplot as plt
import os

from environment_buildup import UAVEnvironment2D
from path_evaluator import PathEvaluator


class PSOPlanner:
    def __init__(self, num_particles=100, num_waypoints=15, max_iter=200):
        """
        初始化 PSO (粒子群算法) 路径规划器
        :param num_particles: 粒子数量（种群大小）
        :param num_waypoints: 每条路径的中间控制点数量
        :param max_iter: 最大迭代次数
        """
        self.evaluator = PathEvaluator()
        self.env = self.evaluator.env
        
        self.num_particles = num_particles
        self.num_waypoints = num_waypoints
        self.max_iter = max_iter
        
        # PSO 核心参数
        self.w_max = 0.9  # 最大惯性权重 (前期注重探索)
        self.w_min = 0.4  # 最小惯性权重 (后期注重开发)
        self.c1 = 1.5 # 个体认知因子
        self.c2 = 1.5 # 社会认知因子
        self.v_max = 8.0 # 【核心修复1】降低最大速度限制！防止飞行途中前后控制点超越打结
        
        # 边界约束
        self.x_bounds = self.env.x_bounds
        self.y_bounds = self.env.y_bounds
        
        # 初始化粒子位置和速度
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((num_particles, num_waypoints, 2))
        
        # 记录个体最优 (pbest) 和全局最优 (gbest)
        self.pbest_pos = np.copy(self.particles)
        self.pbest_scores = np.full(num_particles, np.inf)
        
        self.gbest_pos = None
        self.gbest_score = np.inf

    def _initialize_particles(self):
        """ 初始化 在起点和终点的连线附近生成随机点 """
        start = self.env.start_point
        end = self.env.end_point
        particles = np.zeros((self.num_particles, self.num_waypoints, 2))
        
        # 【通用防打结核心】计算起点到终点的主方向向量
        direction_vec = end - start
        
        for i in range(self.num_particles):
            # 在起点和终点之间均匀取点
            x_vals = np.linspace(start[0], end[0], self.num_waypoints + 2)[1:-1]
            y_vals = np.linspace(start[1], end[1], self.num_waypoints + 2)[1:-1]
            
            # 【改进】加大初始扰动 (噪音)，确保粒子群能覆盖地图两侧
            noise_x = np.random.uniform(-40, 40, self.num_waypoints)
            noise_y = np.random.uniform(-40, 40, self.num_waypoints)
            
            # 组合成 (N, 2) 的坐标数组
            raw_waypoints = np.column_stack((x_vals + noise_x, y_vals + noise_y))
            
            # 【核心修复2：通用投影排序法】
            # 将每个随机生成的控制点，投影到“起点->终点”的大方向向量上
            # 投影值越小，说明在物理宏观上离起点越近
            projections = np.dot(raw_waypoints - start, direction_vec)
            
            # 按投影值从小到大强制排序，理顺“毛线团”
            sort_idx = np.argsort(projections)
            sorted_waypoints = raw_waypoints[sort_idx]
            
            # 限制在地图边界内
            particles[i, :, 0] = np.clip(sorted_waypoints[:, 0], self.x_bounds[0], self.x_bounds[1])
            particles[i, :, 1] = np.clip(sorted_waypoints[:, 1], self.y_bounds[0], self.y_bounds[1])
            
        return particles

    def construct_full_path(self, particle):
        """ 将起点、中间控制点、终点拼接成一条完整的路径序列 """
        path = [self.env.start_point]
        path.extend(particle)
        path.append(self.env.end_point)
        return np.array(path)

    def optimize(self):
        """ PSO 主循环 """
        best_scores_history = []
        
        print("开始 PSO 粒子群算法路径规划迭代...")
        
        # 初始评估
        for i in range(self.num_particles):
            full_path = self.construct_full_path(self.particles[i])
            # 【核心修改】调用 evaluate_pso_particle，内部会生成 Chaikin 曲线并计算综合适应度
            score = self.evaluator.evaluate_pso_particle(full_path)
            
            self.pbest_scores[i] = score
            self.pbest_pos[i] = np.copy(self.particles[i])
            
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_pos = np.copy(self.particles[i])
                
        for iteration in range(self.max_iter):
            # 【核心修复2】动态递减惯性权重 w，让 PSO 前期全图大范围搜索，后期精准收敛
            w_current = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            # 更新粒子的速度和位置
            for i in range(self.num_particles):
                # 生成随机矩阵
                r1 = np.random.rand(self.num_waypoints, 2)
                r2 = np.random.rand(self.num_waypoints, 2)
                
                # 速度更新公式
                cognitive_velocity = self.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                social_velocity = self.c2 * r2 * (self.gbest_pos - self.particles[i])
                self.velocities[i] = w_current * self.velocities[i] + cognitive_velocity + social_velocity
                
                # 【核心修复3】严格限制速度范围！这是解决“贴边飞”和“乱绕圈”的根本方法
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                
                # 位置更新
                self.particles[i] += self.velocities[i]
                
                # 强制边界约束 (防止飞出地图)
                self.particles[i, :, 0] = np.clip(self.particles[i, :, 0], self.x_bounds[0], self.x_bounds[1])
                self.particles[i, :, 1] = np.clip(self.particles[i, :, 1], self.y_bounds[0], self.y_bounds[1])
                
                # 评估新位置
                full_path = self.construct_full_path(self.particles[i])
                score = self.evaluator.evaluate_pso_particle(full_path)
                
                # 更新个体最优
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_pos[i] = np.copy(self.particles[i])
                    
                # 更新全局最优
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos = np.copy(self.particles[i])
                    
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
        
        # 画出底层的控制点 (用灰色虚线和叉叉表示)
        ax1.plot(raw_path[:, 0], raw_path[:, 1], color='gray', linestyle='--', 
                 linewidth=1.5, marker='x', markersize=6, label='PSO Control Waypoints')
                 
        # 画出平滑后的真实飞行曲线 (橙色实线，区分于SSA的粉色)
        ax1.plot(smooth_path[:, 0], smooth_path[:, 1], color='#ff5722', 
                 linewidth=3, label='Chaikin Smooth Path')
        
        # 在地图左上角显示最终得分
        final_score = score_history[-1]
        ax1.text(0.02, 0.98, f'Final Score: {final_score:.2f}', 
                 transform=ax1.transAxes, fontsize=16, fontweight='bold', 
                 color='#d32f2f', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#d32f2f'))
                 
        ax1.legend(loc='lower right')
        
        # 2. 绘制收敛曲线
        ax2.plot(score_history, color='#1976d2', linewidth=2)
        title = 'PSO Convergence Curve'
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
            filename = f"pso_run_{run_idx:02d}.png" if run_idx else "pso_result.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"图像已保存至: {filepath}")
            plt.close(fig) # 关闭图像，防止内存泄漏和弹窗卡死
        else:
            plt.show()

# ==========================================
# 批量运行测试代码
# ==========================================
if __name__ == "__main__":
    log_directory = "PSO_log"
    num_runs = 10
    
    print("=" * 50)
    print(f"开始 PSO 算法批量测试，共计 {num_runs} 次运行")
    print(f"结果将保存在当前目录的 '{log_directory}' 文件夹下")
    print("=" * 50)
    
    for current_run in range(1, num_runs + 1):
        print(f"\n--- 正在进行第 {current_run}/{num_runs} 次运行 ---")
        
        # 参数: 种群100, 航点15个, 迭代200次
        planner = PSOPlanner(num_particles=100, num_waypoints=15, max_iter=200)
        
        # 运行优化 (返回三个值：原始控制点、平滑曲线、历史得分)
        raw_path, smooth_path, history = planner.optimize()
        
        # 可视化展示并保存到文件夹
        planner.plot_result(raw_path, smooth_path, history, run_idx=current_run, save_dir=log_directory)
        
    print("\n所有批量运行已完成！请前往 PSO_log 文件夹查看结果。")