import numpy as np
import matplotlib.pyplot as plt
import os
from path_evaluator import PathEvaluator

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

    def _decode_path(self, position):
        """ [通用方法] 将一维的位置向量还原为包含起终点的完整 3D 路径 """
        waypoints = position.reshape((self.num_waypoints, 3))
        full_path = np.vstack([self.env.start_point, waypoints, self.env.end_point])
        return full_path

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

    # 将拉普拉斯平滑算子下沉为通用“基类武器”
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

    def execute_universal_physics_directives(self):
        """ 
        【基类通用方法】统一执行老中医下发的物理动作！
        任何继承本基类的算法，只需在迭代末尾调用此方法，就能拥有全套物理技能！
        """
        # 响应【拉普拉斯平滑】指令
        if getattr(self, 'apply_laplacian', False):
            self.positions = self.apply_laplacian_smoothing(self.positions, apply_ratio=0.5)

    def optimize(self):
        """ [抽象方法] 核心的迭代寻优逻辑，必须由继承的子类自己实现！ """
        raise NotImplementedError("子类必须实现 optimize() 方法！")

    def plot_result(self, best_path, score_history, algo_name="Algorithm", run_idx=None, save_dir=None):
        # ... (画图代码与之前保持一致) ...
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        
        self.env.draw_environment_3d(ax=ax1)
        smooth_path = self.evaluator.generate_bspline_path(best_path, num_points=100)
        
        ax1.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2], 
                 color='#e65100', linewidth=3, label=f'{algo_name} Smooth Path', zorder=6)
        ax1.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2], 
                 color='gray', linewidth=1, linestyle='--',
                 marker='o', markersize=5, label='Raw Waypoints', alpha=0.6, zorder=5)
                 
        ax1.legend(loc='upper left', fontsize=10)
        ax2.plot(score_history, color='#2e7d32', linewidth=2)
        
        title = f'{algo_name} Convergence Curve'
        if run_idx is not None: title += f' (Run {run_idx})'
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Fitness Score (Log Scale)', fontsize=12)
        ax2.set_yscale('log') 
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        params_list = []
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float)) and not k.startswith('_') \
               and 'score' not in k and 'pos' not in k and 'bound' not in k \
               and k not in ['lb', 'ub', 'dim']:
                val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                params_list.append(f"  {k}: {val_str}")
        
        params_text = f"Parameters:\n" + "\n".join(params_list)
        ax2.text(0.20, 0.95, params_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        _, details, _ = self.evaluator.evaluate_pso_particle(best_path)
        details_list = []
        for k, v in details.items():
            if v > 0: details_list.append(f"  {k}: {v:,.0f}")
                
        if not details_list: details_list.append("  Perfect! No penalties.")
        details_text = "Fitness Breakdown:\n" + "\n".join(details_list)
        ax2.text(0.95, 0.88, details_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='#eef7ff', alpha=0.9, edgecolor='#1976d2'))
        
        final_score = score_history[-1] if len(score_history) > 0 else 0
        ax2.text(0.95, 0.95, f'Best Score: {final_score:,.2f}', 
                 transform=ax2.transAxes, fontsize=12, fontweight='bold', 
                 color='white', horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#d32f2f', alpha=0.9, edgecolor='none'))
        
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{algo_name}_run_{run_idx:02d}.png" if run_idx else f"{algo_name}_result.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close(fig)
        else:
            plt.show()