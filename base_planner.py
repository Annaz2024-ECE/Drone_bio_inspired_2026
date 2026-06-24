import numpy as np
import matplotlib.pyplot as plt
import os
from path_evaluator import PathEvaluator

class BasePlanner:
    def __init__(self, num_waypoints=10, max_iter=200, evaluator=None):
        """
        所有路径规划算法的通用基类
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
        self.dim = self.num_waypoints * 2  # 搜索维度 (X和Y)
        
        # 3. 统一管理边界
        self.lb = min(self.env.x_bounds[0], self.env.y_bounds[0])
        self.ub = max(self.env.x_bounds[1], self.env.y_bounds[1])
        
        # 4. 统一的数据记录容器
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = float("inf")
        self.convergence_curve = []

    def _decode_path(self, position):
        """
        [通用方法] 将一维的位置向量还原为包含起终点的完整二维路径
        """
        waypoints = position.reshape((self.num_waypoints, 2))
        full_path = np.vstack([self.env.start_point, waypoints, self.env.end_point])
        return full_path

    def optimize(self):
        """
        [抽象方法] 核心的迭代寻优逻辑，必须由继承的子类 (如 PSO, GWO) 自己实现！
        """
        raise NotImplementedError("子类必须实现 optimize() 方法！")

    def plot_result(self, best_path, score_history, algo_name="Algorithm", run_idx=None, save_dir=None):
        """ 
        [通用方法] 统一的绘制结果函数：左图为地图路径，右图为收敛曲线 
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 绘制地图和原始控制点路径
        self.env.draw_environment(ax=ax1)
        
        # 生成 Chaikin 平滑后的飞行曲线
        smooth_path = self.evaluator.generate_chaikin_path(best_path, iterations=3)
        
        # 红色实线表示 smooth 路径 (置于顶层)
        ax1.plot(smooth_path[:, 0], smooth_path[:, 1], color='#e65100', linewidth=3, label=f'{algo_name} Smooth Path', zorder=6)
        
        # 灰色虚线表示 raw 路径 (置于底层)
        ax1.plot(best_path[:, 0], best_path[:, 1], color='gray', linewidth=1, linestyle='--',
                 marker='o', markersize=5, label='Raw Waypoints', alpha=0.6, zorder=5)
                 
        ax1.legend(loc='upper left', fontsize=10)
        
        # 2. 绘制对数坐标系的收敛曲线 (使用深绿色)
        ax2.plot(score_history, color='#2e7d32', linewidth=2)
        
        # 支持显示运行批次 (Run Index)
        title = f'{algo_name} Convergence Curve'
        if run_idx is not None:
            title += f' (Run {run_idx})'
        ax2.set_title(title, fontsize=14, fontweight='bold')
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Fitness Score (Log Scale)', fontsize=12)
        ax2.set_yscale('log') 
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # ==========================================
        # 智能参数提取模块
        # ==========================================
        # 动态提取当前算法的超参数 (过滤掉内部矩阵、坐标和无关变量)
        params_list = []
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float)) and not k.startswith('_') \
               and 'score' not in k and 'pos' not in k and 'bound' not in k \
               and k not in ['lb', 'ub', 'dim']:
                # 稍微格式化一下浮点数
                val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                params_list.append(f"  {k}: {val_str}")
        
        params_text = f"Parameters:\n" + "\n".join(params_list)
        
        # 在左上角添加参数信息框
        ax2.text(0.05, 0.95, params_text,
                 transform=ax2.transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 在右上角显示最终得分框 (醒目红色)
        final_score = score_history[-1] if len(score_history) > 0 else 0
        ax2.text(0.95, 0.95, f'Best Score: {final_score:,.2f}', 
                 transform=ax2.transAxes, fontsize=12, fontweight='bold', 
                 color='white', horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#d32f2f', alpha=0.9, edgecolor='none'))
        
        plt.tight_layout()
        
        # 支持批量自动命名保存
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{algo_name}_run_{run_idx:02d}.png" if run_idx else f"{algo_name}_result.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close(fig)
        else:
            plt.show()