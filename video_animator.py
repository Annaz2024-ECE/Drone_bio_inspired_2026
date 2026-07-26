import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

class UAVVideoAnimator:
    def __init__(self, evaluator):
        """
        无人机 3D 动态飞行视频/动画生成器
        :param evaluator: 包含 env 环境与 generate_bspline_path 方法的 PathEvaluator 实例
        """
        self.evaluator = evaluator
        self.env = evaluator.env

    def create_flight_video(self, best_path, filename="uav_flight.mp4", duration=15.0, fps=30, trail_time=1.5):
        """
        生成依据真实物理动力学的 3D 飞行视频
        :param best_path: 关键控制点路径 (N, 3)
        :param filename: 输出文件名 (.mp4 或 .gif)
        :param duration: 视频总时长 (秒)。调大这个值可以让整体动画变慢。
        :param fps: 每秒帧数 (默认 30)
        :param trail_time: 动态高亮尾迹代表的过去时间长度 (秒)
        """
        print(f"\n 正在依据物理动力学渲染 3D 飞行视频 (模拟变速巡航)...")
        
        # ----------------------------------------------------
        # 核心升级：基于真实飞行时间的运动学重采样 (Kinematic Resampling)
        # ----------------------------------------------------
        # 1. 获取一个极高密度的基础骨架（1000个点），用于计算精细的物理参数
        base_points = 1000
        dense_path = self.evaluator.generate_bspline_path(best_path, num_points=base_points)
        
        distances = np.zeros(base_points)
        times = np.zeros(base_points)
        speeds = np.zeros(base_points)
        
        # 2. 逐点计算真实速度和真实累积耗时
        for i in range(1, base_points):
            dist = np.linalg.norm(dense_path[i] - dense_path[i-1])
            distances[i] = distances[i-1] + dist
            
            # 获取环境设定的该坐标点下的理论速度 (巡航 13.17 vs 巡检 5.0)
            if hasattr(self.evaluator, '_get_local_speed'):
                v = self.evaluator._get_local_speed(dense_path[i])
            else:
                v = self.evaluator.params.get('v_cruise', 13.17)
                
            speeds[i] = v
            # 时间 = 距离 / 速度
            times[i] = times[i-1] + (dist / max(v, 0.1)) # max防除0
            
        speeds[0] = speeds[1] # 修复起点速度显示
        total_real_time = times[-1]
        
        # 3. 将真实飞行时间映射到视频的总帧数上
        total_frames = int(duration * fps)
        # 生成均匀分布的视频帧对应的时间点
        frame_times = np.linspace(0, total_real_time, total_frames)
        
        # 4. 基于时间进行插值，生成给每一帧的具体坐标和速度
        smooth_path = np.zeros((total_frames, 3))
        for dim in range(3):
            smooth_path[:, dim] = np.interp(frame_times, times, dense_path[:, dim])
            
        frame_speeds = np.interp(frame_times, times, speeds)
        
        # 尾迹长度（帧数）
        trail_length = int(trail_time * (total_frames / duration))
        
        # ----------------------------------------------------
        # 渲染画布初始化
        # ----------------------------------------------------
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制背景 3D 环境
        self.env.draw_environment_3d(ax=ax)
        
        # 预先画出整条浅色的“预定航线”（作为参考虚线）
        ax.plot(dense_path[:, 0], dense_path[:, 1], dense_path[:, 2],
                color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='Planned Flight Path')
        
        # 初始化动画元素
        line_history, = ax.plot([], [], [], color='#f57c00', linewidth=1.5, alpha=0.6)
        line_trail, = ax.plot([], [], [], color='#d32f2f', linewidth=3.5, label='Active UAV Trail')
        drone_head = ax.scatter([], [], [], color='#d32f2f', s=120, marker='o', edgecolors='black', zorder=10, label='UAV Position')
        arrow_holder = [None]

        # ----------------------------------------------------
        # 逐帧更新逻辑
        # ----------------------------------------------------
        def update(frame):
            curr_pos = smooth_path[frame]
            curr_spd = frame_speeds[frame]
            
            # --- 更新历史轨迹与尾迹 ---
            line_history.set_data(smooth_path[:frame+1, 0], smooth_path[:frame+1, 1])
            line_history.set_3d_properties(smooth_path[:frame+1, 2])
            
            start_idx = max(0, frame - trail_length)
            line_trail.set_data(smooth_path[start_idx:frame+1, 0], smooth_path[start_idx:frame+1, 1])
            line_trail.set_3d_properties(smooth_path[start_idx:frame+1, 2])
            
            # --- 更新无人机位置 ---
            drone_head._offsets3d = ([curr_pos[0]], [curr_pos[1]], [curr_pos[2]])
            
            # --- 计算方向向量并更新 3D 箭头 ---
            if frame > 0:
                direction = curr_pos - smooth_path[frame - 1]
                norm = np.linalg.norm(direction)
                if norm > 1e-5:
                    direction = direction / norm * 4.0 
                else:
                    direction = np.array([0, 0, 1])
            else:
                direction = np.array([0, 0, 1])

            if arrow_holder[0] is not None:
                arrow_holder[0].remove()
            
            arrow_holder[0] = ax.quiver(
                curr_pos[0], curr_pos[1], curr_pos[2],
                direction[0], direction[1], direction[2],
                color='#ff1744', linewidth=2, arrow_length_ratio=0.4
            )

            # 动态呈现物理速度
            status_symbol = "Cruising" if curr_spd > 10.0 else "Inspecting"
            ax.set_title(
                f'UAV 3D Kinematic Simulation | Alt: {curr_pos[2]:.1f}m\n{status_symbol} Speed: {curr_spd:.1f} m/s',
                fontsize=14, fontweight='bold', color='#1a237e',
                pad=15   # 单位：点（points），增加标题与图形上边框的距离
            )

            return line_history, line_trail, drone_head

        ani = animation.FuncAnimation(
            fig, update, frames=total_frames, interval=1000/fps, blit=False
        )

        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.93])   # 0.93 表示保留顶部 7% 空间

        # ----------------------------------------------------
        # 导出文件
        # ----------------------------------------------------
        if filename.endswith(".gif"):
            print(f"  正在压制 GIF 动画...")
            writer = animation.PillowWriter(fps=fps)
            ani.save(filename, writer=writer)
            print(f"  动态 GIF 已成功导出至: {filename}")
        else:
            print(f"  正在使用 FFMpeg 导出高画质 MP4 视频...")
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                ani.save(filename, writer=writer)
                print(f"  物理测速视频已成功导出至: {filename}")
            except Exception as e:
                print(f"  FFMpeg 导出失败 ({e})，正在自动降级导出为 GIF...")
                gif_name = filename.replace(".mp4", ".gif")
                writer = animation.PillowWriter(fps=fps)
                ani.save(gif_name, writer=writer)
                print(f"  降级导出完成！动态 GIF 已保存至: {gif_name}")

        plt.close(fig)
