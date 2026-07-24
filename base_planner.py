import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # 🔥 引入非对称布局模块
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import time
from path_evaluator import PathEvaluator
from video_animator import UAVVideoAnimator

class BasePlanner:
    def __init__(self, num_waypoints=10, max_iter=200, evaluator=None):
        """
        所有路径规划算法的通用基类 (3D升级版)
        :param num_waypoints: 中间控制点数量
        :param max_iter: 最大迭代次数
        :param evaluator: 路径评价器
        """
        # 1. 统一管理评价器与环境
        self.evaluator = evaluator if evaluator else PathEvaluator()
        self.env = self.evaluator.env
        
        # 2. 统一管理基础参数
        self.num_waypoints = num_waypoints
        self.max_iter = max_iter
        
        # 搜索维度从 2 升级到 3 (X, Y, Z)
        self.dim = self.num_waypoints * 3  
        
        # 精确的 3D 边界控制，防止高度越界
        self.lb = np.tile([self.env.x_bounds[0], self.env.y_bounds[0], self.env.z_bounds[0]], self.num_waypoints)
        self.ub = np.tile([self.env.x_bounds[1], self.env.y_bounds[1], self.env.z_bounds[1]], self.num_waypoints)
        
        # 4. 统一的数据记录容器
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = float("inf")
        self.convergence_curve = []

        # 莱维通用默认参数
        self.levy_beta = 1.5      # 莱维飞行指数
        self.levy_scale = 0.05    # 步长缩放因子（步长权重）
        # 【新增】算法实例化的瞬间，按下计时秒表！
        self.start_time = time.time()

    def _decode_path(self, position):
        """ [通用方法] 将一维的位置向量还原为包含起终点的完整 3D 路径 """
        waypoints = position.reshape((self.num_waypoints, 3))
        full_path = np.vstack([self.env.start_point, waypoints, self.env.end_point])
        return full_path

    def _levy_step(self, dim):
        """
        使用经典 Mantegna 算法生成通用莱维飞行步长向量。
        支持通过 self.levy_beta 动态调节其分布形状。
        """
        # 动态获取当前实例的 beta 值，若未定义则使用默认值 1.5
        beta = getattr(self, 'levy_beta', 1.5)
        
        # 计算 Mantegna 算法中的标准差 sigma_u
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        sigma_v = 1.0
        
        # 生成正态分布随机数
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, sigma_v, dim)
        
        # 计算步长
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _generate_basic_skeleton(self):
        """ 
        [新增] 基础版 TSP 骨架生成器
        不包含三点微簇锚固和高空拱门技术。
        仅通过最近邻算法连接打卡点，并将剩余的航点均匀插值在连线上。
        适合不需要强制贯穿控制或者航点数量较少的场景。
        """
        start = self.env.start_point
        end = self.env.end_point
        
        targets_3d = []
        for t in self.env.target_areas:
            center = t['center']
            z_mid = (t.get('z_min', 0) + t.get('z_max', 10)) / 2.0
            targets_3d.append(np.array([center[0], center[1], z_mid]))
        
        # 1. 最近邻 TSP 排序
        unvisited = targets_3d.copy()
        sorted_targets = []
        current = start
        while unvisited:
            distances = [np.linalg.norm(p - current) for p in unvisited]
            idx = np.argmin(distances)
            nearest = unvisited.pop(idx)
            sorted_targets.append(nearest)
            current = nearest
        
        # 2. 基础连线：每个目标只给 1 个控制点 (不再是 3 个微簇)
        control_pts = []
        for t in sorted_targets:
            control_pts.append(t)
            
        # 3. 均匀插值：在最长间距中插入剩余的控制点 (不加高空拱门逻辑)
        while len(control_pts) < self.num_waypoints:
            max_gap = -1.0
            max_idx = 0
            temp_path = [start] + control_pts + [end]
            for i in range(len(temp_path) - 1):
                gap = np.linalg.norm(temp_path[i+1] - temp_path[i])
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
                    
            # 简单的线性中点插值
            mid_point = (temp_path[max_idx] + temp_path[max_idx+1]) / 2.0
            
            if max_idx == 0: 
                control_pts.insert(0, mid_point)
            elif max_idx == len(temp_path) - 1: 
                control_pts.append(mid_point)
            else: 
                control_pts.insert(max_idx, mid_point)
                
        control_pts = control_pts[:self.num_waypoints]
        return np.clip(np.array(control_pts).flatten(), self.lb, self.ub)

    def _generate_heuristic_skeleton(self):
        """ [新增通用方法] 生成 3D 启发式拓扑骨架 (超级基因) """
        start = self.env.start_point
        end = self.env.end_point
        
        targets_3d = []
        for t in self.env.target_areas:
            center = t['center']
            z_mid = (t.get('z_min', 0) + t.get('z_max', 10)) / 2.0
            targets_3d.append(np.array([center[0], center[1], z_mid]))
        
        unvisited = targets_3d.copy()
        sorted_targets = []
        current = start
        while unvisited:
            distances = [np.linalg.norm(p - current) for p in unvisited]
            idx = np.argmin(distances)
            nearest = unvisited.pop(idx)
            sorted_targets.append(nearest)
            current = nearest
        
        control_pts = []
        for t in sorted_targets:
            control_pts.append(t + np.array([-0.3, -0.3, -0.1]))
            control_pts.append(t)
            control_pts.append(t + np.array([0.3, 0.3, 0.1]))
            
        safe_arch_z = 8.0 
        while len(control_pts) < self.num_waypoints:
            max_gap = -1.0
            max_idx = 0
            temp_path = [start] + control_pts + [end]
            for i in range(len(temp_path) - 1):
                gap = np.linalg.norm(temp_path[i+1] - temp_path[i])
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
                    
            mid_point = (temp_path[max_idx] + temp_path[max_idx+1]) / 2.0
            
            if max_gap > 8.0:
                mid_point[2] = max(mid_point[2], safe_arch_z) 
            
            if max_idx == 0: 
                control_pts.insert(0, mid_point)
            elif max_idx == len(temp_path) - 1: 
                control_pts.append(mid_point)
            else: 
                control_pts.insert(max_idx, mid_point)
                
        # 严格截断到指定的控制点数量
        control_pts = control_pts[:self.num_waypoints]
        
        # 返回被环境物理边界安全修剪过的一维超级基因数组
        return np.clip(np.array(control_pts).flatten(), self.lb, self.ub)

    # ==========================================
    # 🔥 核心升级：将拉普拉斯平滑算子下沉为通用“基类武器”
    # ==========================================
    def apply_laplacian_smoothing(self, positions, apply_ratio=0.5):
        """ 【基类通用方法】拉普拉斯平滑算子 (弹性带理论) """
        pop_size = positions.shape[0]
        smoothed_positions = np.copy(positions)
        
        for i in range(pop_size):
            if np.random.rand() < apply_ratio:
                pts = smoothed_positions[i].reshape((self.num_waypoints, 3))
                new_pts = np.copy(pts)
                # 像拉橡皮筋一样，把中间的控制点向前后两端拉扯
                for j in range(1, self.num_waypoints - 1):
                    new_pts[j] = (pts[j-1] + 2.0 * pts[j] + pts[j+1]) / 4.0
                smoothed_positions[i] = np.clip(new_pts.flatten(), self.lb, self.ub)
                
        return smoothed_positions


    # 自适应法向斥力 (专门对付 Margin Violation)
    def apply_margin_repulsion(self, positions, apply_ratio=0.5):
        """ 【基类通用方法】法向推离算子 (人工势场法变体) """
        pop_size = positions.shape[0]
        repelled_positions = np.copy(positions)
        
        for i in range(pop_size):
            if np.random.rand() < apply_ratio:
                pts = repelled_positions[i].reshape((self.num_waypoints, 3))
                new_pts = np.copy(pts)
                
                for j in range(1, self.num_waypoints - 1):
                    # 1. 计算当前航段的飞行方向向量
                    forward_vec = pts[j+1] - pts[j-1]
                    forward_vec[2] = 0 # 仅在水平面计算法向量（因为建筑物通常是垂直的）
                    norm = np.linalg.norm(forward_vec)
                    
                    if norm > 0.1:
                        # 2. 逆时针旋转 90 度得到垂直法向量
                        normal_vec = np.array([-forward_vec[1], forward_vec[0], 0]) / norm
                        
                        # 3. 随机向左或向右推开 1.5 米 (必然有一个方向是远离墙壁的)
                        push_dir = 1.0 if np.random.rand() < 0.5 else -1.0
                        push_force = normal_vec * push_dir * 1.5 
                        
                        # 4. 施加推力，同时给予微弱的安全升力 (爬升 0.5 米，对抗楼顶边缘擦碰)
                        new_pts[j] += push_force
                        new_pts[j][2] += 0.5 
                        
                repelled_positions[i] = np.clip(new_pts.flatten(), self.lb, self.ub)
                
        return repelled_positions

    def execute_universal_physics_directives(self):
        """ 
        【基类通用方法】统一执行老中医下发的物理动作！(渐进式升级版)
        """
        # 读取指挥部下发的“油门踏板”强度。如果之前的老代码没传，默认为 0.0 (关闭)
        lap_intensity = getattr(self, 'laplacian_intensity', 0.0)
        rep_intensity = getattr(self, 'repulsion_intensity', 0.0)

        # 1. 响应【拉普拉斯平滑】渐进指令 (消除锯齿)
        if lap_intensity > 0.0:
            # 将 0.0~1.0 的强度直接作为种群的覆盖概率
            self.positions = self.apply_laplacian_smoothing(self.positions, apply_ratio=lap_intensity)
            
        # 2. 响应【侧向斥力推离】渐进指令 (消除擦墙)
        if rep_intensity > 0.0:
            # 将 0.0~1.0 的强度直接作为种群的覆盖概率
            self.positions = self.apply_margin_repulsion(self.positions, apply_ratio=rep_intensity)

    def optimize(self):
        """ [抽象方法] 核心的迭代寻优逻辑，必须由继承的子类自己实现！ """
        raise NotImplementedError("子类必须实现 optimize() 方法！")

    def plot_result(self, best_path, score_history, algo_name="Algorithm", run_idx=None, save_dir=None, global_start_time=None, event_history=None):
    # ----- 计时与性能信息 -----
        end_time = time.time()
        round_time = end_time - self.start_time
        
        def get_time_str(seconds):
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            if h > 0:
                return f"{seconds:.2f}s ({h}h {m}m {s}s)"
            elif m > 0:
                return f"{seconds:.2f}s ({m}m {s}s)"
            else:
                return f"{seconds:.2f}s"

        if global_start_time is not None:
            total_time = end_time - global_start_time
            print(f"\n[性能监控] {algo_name} 本轮耗时: {get_time_str(round_time)} | 多智能体系统总耗时: \033[93m{get_time_str(total_time)}\033[0m")
            time_display = f"Total Time: {get_time_str(total_time)}"
        else:
            print(f"\n[性能监控] {algo_name} 算法计算耗时: \033[93m{get_time_str(round_time)}\033[0m")
            time_display = f"Time Cost: {get_time_str(round_time)}"
        print("-" * 65)

        # ----- 生成平滑路径和速度数据（供所有图共用）-----
        smooth_path = self.evaluator.generate_bspline_path(best_path, num_points=100)
        
        path_distances = [0.0]
        cumulative_dist = 0.0
        speeds = []
        for i in range(len(smooth_path)):
            if i > 0:
                dist = np.linalg.norm(smooth_path[i] - smooth_path[i-1])
                cumulative_dist += dist
                path_distances.append(cumulative_dist)
            if hasattr(self.evaluator, '_get_local_speed'):
                v = self.evaluator._get_local_speed(smooth_path[i])
            else:
                v = self.evaluator.params.get('v_cruise', 13.17)
            speeds.append(v)
        speeds = np.array(speeds)

        v_cruise = self.evaluator.params.get('v_cruise', 13.17)
        v_insp = self.evaluator.params.get('v_inspection', 5.0)

        # ==========================================
        # 图 1：速度剖面图（保持不变）
        # ==========================================
        fig_speed = plt.figure(figsize=(10, 5))
        ax_speed = fig_speed.add_subplot(111)
        ax_speed.plot(path_distances, speeds, color='#1976d2', linewidth=3, linestyle='-')
        ax_speed.fill_between(path_distances, speeds, color='#bbdefb', alpha=0.4)
        ax_speed.axhline(y=v_cruise, color='gray', linestyle='--', alpha=0.7, label=f'Cruise ({v_cruise} m/s)')
        ax_speed.axhline(y=v_insp, color='#d32f2f', linestyle='--', alpha=0.7, label=f'Inspection ({v_insp} m/s)')
        title_speed = f'{algo_name} Dynamic Speed Profile'
        if run_idx is not None: title_speed += f' (Run {run_idx})'
        ax_speed.set_title(title_speed, fontsize=14, fontweight='bold')
        ax_speed.set_xlabel('Distance along path (m)', fontsize=12)
        ax_speed.set_ylabel('Target Speed (m/s)', fontsize=12)
        ax_speed.set_ylim(0, max(max(speeds), v_cruise) * 1.3)
        ax_speed.legend(loc='lower right')
        ax_speed.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # ------------------------------------------
        # 图 2：3D 环境 + 轨迹（速度热力）
        # ------------------------------------------
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        self.env.draw_environment_3d(ax=ax_3d)
        ax_3d.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2],
                color='#d32f2f', linewidth=3, label='3D Flight Path', zorder=6)
        ax_3d.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2],
                color='gray', linewidth=1, linestyle='--',
                marker='o', markersize=5, label='Raw Waypoints', alpha=0.6, zorder=5)
        title_3d = f'{algo_name} 3D Trajectory'
        if run_idx is not None: title_3d += f' (Run {run_idx})'
        ax_3d.set_title(title_3d, fontsize=14, fontweight='bold')
        ax_3d.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

        # ==========================================
        # 图 3：2D 俯视图（独立）
        # ==========================================
        fig_2d = plt.figure(figsize=(10, 8))
        ax_2d = fig_2d.add_subplot(111)
        ax_2d.set_xlim(self.env.x_bounds)
        ax_2d.set_ylim(self.env.y_bounds)

        # 绘制障碍物（2D）
        for obs in self.env.obstacles:
            z_max = obs.get('z_max', 20.0)
            if z_max <= 1.5: color = '#e3f2fd'
            elif z_max <= 3.0: color = '#90caf9'
            elif z_max <= 5.0: color = '#1e88e5'
            else: color = '#0d47a1'
            if obs['type'] == 'circle':
                circle = plt.Circle(obs['center'][:2], obs['radius'], color=color, alpha=0.7, ec='black')
                ax_2d.add_patch(circle)
            elif obs['type'] == 'rect':
                from matplotlib.patches import Rectangle
                rect = Rectangle(obs['bottom_left'][:2], obs['width'], obs['height'],
                                angle=obs.get('angle', 0), color=color, alpha=0.7, ec='black')
                ax_2d.add_patch(rect)
            elif obs['type'] == 'polygon':
                from matplotlib.patches import Polygon as mpl_Polygon
                poly = mpl_Polygon(obs['points'], closed=True, color=color, alpha=0.7, ec='black')
                ax_2d.add_patch(poly)

        # 绘制目标区域
        for target in self.env.target_areas:
            circle = plt.Circle(target['center'][:2], target['radius'],
                                color='#4caf50', fill=False, linestyle='--', linewidth=2)
            ax_2d.add_patch(circle)
            ax_2d.text(target['center'][0], target['center'][1], target['name'],
                    color='#2e7d32', fontweight='bold', ha='center', va='center', fontsize=8)

        # 准备 2D 线段
        points_2d = smooth_path[:, :2].reshape(-1, 1, 2)
        segments_2d = np.concatenate([points_2d[:-1], points_2d[1:]], axis=1)
        speed_values = speeds[:-1]  # 线段数 = 点数-1

        cmap = plt.get_cmap('plasma')
        norm = mcolors.Normalize(vmin=min(speeds)-2, vmax=max(speeds)+2)

        lc_2d = LineCollection(segments_2d, cmap=cmap, norm=norm, linewidths=3.5, zorder=6)
        lc_2d.set_array(speed_values)
        ax_2d.add_collection(lc_2d)

        # 原始控制点
        ax_2d.plot(best_path[:, 0], best_path[:, 1], 'o', color='gray', markersize=3, alpha=0.5, zorder=5)
        ax_2d.scatter(*self.env.start_point[:2], color='#fbc02d', s=100, marker='*', edgecolors='black', zorder=10)
        ax_2d.scatter(*self.env.end_point[:2], color='#d32f2f', s=80, marker='^', zorder=10)

        # 颜色条
        cbar_2d = fig_2d.colorbar(lc_2d, ax=ax_2d, shrink=0.6, pad=0.05)
        cbar_2d.set_label('Target Speed (m/s)', fontweight='bold')

        title_2d = f'{algo_name} Top-down 2D Speed Map'
        if run_idx is not None: title_2d += f' (Run {run_idx})'
        ax_2d.set_title(title_2d, fontsize=14, fontweight='bold')
        ax_2d.set_xlabel('X (m)')
        ax_2d.set_ylabel('Y (m)')
        ax_2d.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()

        # ==========================================
        # 图 4：收敛曲线 + 干预事件 + 参数（独立）
        # ==========================================
        fig_curve = plt.figure(figsize=(10, 8))
        ax_curve = fig_curve.add_subplot(111)
        ax_curve.plot(score_history, color='#2e7d32', linewidth=2)
        ax_curve.set_title(f'{algo_name} Convergence Curve', fontsize=14, fontweight='bold')
        ax_curve.set_xlabel('Iteration', fontsize=12)
        ax_curve.set_ylabel('Score (Log)', fontsize=12)
        ax_curve.set_yscale('log')
        ax_curve.grid(True, linestyle=':', alpha=0.6)

        # 干预事件标记
        if event_history:
            event_colors = ['#d32f2f', '#1976d2', '#f57c00', '#388e3c', '#8e24aa']
            for i, (iter_idx, action_tag) in enumerate(event_history):
                if iter_idx >= len(score_history): continue
                color = event_colors[i % len(event_colors)]
                ax_curve.axvline(x=iter_idx, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
                y_offset = 0.10 + (i % 3) * 0.20
                ax_curve.text(iter_idx, y_offset, f" {action_tag}",
                            transform=ax_curve.get_xaxis_transform(),
                            rotation=0, color=color,
                            fontsize=7, linespacing=1.2, fontweight='bold',
                            va='bottom', ha='left',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color))

        # 参数展示
        excluded_params = ['start_time', 'emergency_escape', 'radar_guidance',
                        'lift_up', 'press_down', 'apply_laplacian', 'apply_repulsion',
                        'laplacian_intensity', 'repulsion_intensity']
        params_list = []
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float)) and not k.startswith('_') \
            and 'score' not in k and 'pos' not in k and 'bound' not in k \
            and k not in ['lb', 'ub', 'dim'] and k not in excluded_params:
                val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                params_list.append(f"  {k}: {val_str}")
        params_list.append("-" * 15)
        params_list.append(f"  {time_display}")
        params_text = "Parameters:\n" + "\n".join(params_list)
        ax_curve.text(0.05, 0.95, params_text, transform=ax_curve.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # 罚分明细
        _, details, _ = self.evaluator.evaluate_particle(best_path)
        details_list = [f"  {k}: {v:,.0f}" for k, v in details.items() if v > 0]
        if not details_list: details_list.append("  Perfect! No penalties.")
        details_text = "Fitness Breakdown:\n" + "\n".join(details_list)
        ax_curve.text(0.95, 0.85, details_text, transform=ax_curve.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='#eef7ff', alpha=0.9, edgecolor='#1976d2'))

        final_score = score_history[-1] if len(score_history) > 0 else 0
        ax_curve.text(0.95, 0.95, f'Best Score: {final_score:,.2f}',
                    transform=ax_curve.transAxes, fontsize=10, fontweight='bold',
                    color='white', horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#d32f2f', alpha=0.9, edgecolor='none'))

        plt.tight_layout()

        # ==========================================
        # 保存或显示所有图片
        # ==========================================
        if run_idx is not None:
            # 如果指定了 run_idx，则在 save_dir 下创建子文件夹
            if save_dir is None:
                save_dir = "."  # 如果没有给定总目录，默认当前目录
            sub_dir = os.path.join(save_dir, f"run_{run_idx:02d}")
            os.makedirs(sub_dir, exist_ok=True)
            
            # 保存速度图
            fig_speed.savefig(os.path.join(sub_dir, f"{algo_name}_speed.png"), dpi=300)
            plt.close(fig_speed)
            # 保存3D图
            fig_3d.savefig(os.path.join(sub_dir, f"{algo_name}_3d.png"), dpi=300)
            plt.close(fig_3d)
            # 保存2D图
            fig_2d.savefig(os.path.join(sub_dir, f"{algo_name}_2d.png"), dpi=300)
            plt.close(fig_2d)
            # 保存收敛曲线图
            fig_curve.savefig(os.path.join(sub_dir, f"{algo_name}_curve.png"), dpi=300)
            plt.close(fig_curve)
            print(f"  ✅ 四张图已保存至 {sub_dir}")

            # ==========================================
            # 🔥 新增：直接调用外部的视频渲染器类生成 MP4
            # ==========================================
            # 1. 设置视频要保存的绝对路径 (和那 4 张图片存在一起)
            video_filename = os.path.join(sub_dir, f"{algo_name}_flight.mp4")
            
            # 2. 实例化并调用！
            animator = UAVVideoAnimator(self.evaluator)
            animator.create_flight_video(best_path, filename=video_filename, duration=18.0)
            
        else:
            # 调试模式：依次弹出显示
            print("\n  [展示提示] 显示速度剖面图...")
            plt.show()
            print("  [展示提示] 显示3D轨迹图...")
            plt.show()
            print("  [展示提示] 显示2D俯视图...")
            plt.show()
            print("  [展示提示] 显示收敛曲线图...")
            plt.show()