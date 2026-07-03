import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyjson5
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class UAVEnvironment3D:
    """升级版3D地图环境：读取JSON配置，提供3D碰撞检测与3D可视化"""
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = pyjson5.load(f)
        
        self.name = data['name']
        self.x_bounds = [0, data['bounds'][0]]
        self.y_bounds = [0, data['bounds'][1]]
        
        # 寻找地图中的最高建筑，用于设置 Z 轴边界
        max_z = 30.0 
        for obs in data['obstacles']:
            max_z = max(max_z, obs.get('z_max', 20.0))
        self.z_bounds = [0, max_z + 10] # 天花板留出 10m 余量

        # 起点和终点：如果 JSON 中只有二维，默认 Z=0，建议后续传入 [x, y, z]
        def to_3d(point):
            return np.array(list(point) + [0.0]) if len(point) == 2 else np.array(point)
            
        self.start_point = to_3d(data['start_point'])
        self.end_point = to_3d(data['end_point'])
        
        self.obstacles = []
        for obs in data['obstacles']:
            o = obs.copy()
            if 'bottom_left' in o: o['bottom_left'] = np.array(o['bottom_left'])
            if 'center' in o: o['center'] = np.array(o['center'])
            # 补齐默认高度，防止 json 漏填
            o['z_min'] = o.get('z_min', 0.0)
            o['z_max'] = o.get('z_max', 20.0)
            self.obstacles.append(o)
            
        self.target_areas = data['target_areas']
        for target in self.target_areas:
            target['center'] = np.array(target['center'])

    def calculate_distance(self, point1, point2):
        """ 计算两点之间的 3D 欧氏距离 """
        return np.linalg.norm(point1 - point2)

    def is_point_in_obstacle(self, point, safe_margin=0.0):
        """ 
        【核心检测】检测 3D 空间中的单点是否在障碍物内部 (包含高度判断) 
        point 格式应为: [x, y, z]
        """
        for obs in self.obstacles:
            # 1. 优先进行高度层 (Z轴) 筛选，不在该高度范围直接跳过
            z_min = obs['z_min'] - safe_margin
            z_max = obs['z_max'] + safe_margin
            if not (z_min <= point[2] <= z_max):
                continue

            # 2. 如果高度命中，再判断 2D 投影面
            if obs['type'] == 'circle':
                # 仅计算 XY 平面的距离
                dist = np.linalg.norm(point[:2] - obs['center'])
                if dist <= (obs['radius'] + safe_margin):
                    return True

            elif obs['type'] == 'rect':
                bl = obs['bottom_left']
                w, h = obs['width'], obs['height']
                angle_deg = obs.get('angle', 0.0)
                
                dx = point[0] - bl[0]
                dy = point[1] - bl[1]
                
                theta = math.radians(-angle_deg)
                rx = dx * math.cos(theta) - dy * math.sin(theta)
                ry = dx * math.sin(theta) + dy * math.cos(theta)
                
                if -safe_margin <= rx <= w + safe_margin and -safe_margin <= ry <= h + safe_margin:
                    return True
        return False

    def is_segment_collision(self, p1, p2, safe_margin=0.0, step=0.5):
        """
        【3D线段检测】使用离散采样法检测 3D 线段(p1->p2)是否碰撞。
        通过在无人机两点航线中以 step(米) 为步长进行插值采样。
        这是无人机 3D 路径规划中最常用且鲁棒的做法。
        """
        dist = self.calculate_distance(p1, p2)
        if dist == 0:
            return self.is_point_in_obstacle(p1, safe_margin)

        # 根据步长计算需要采样的点数，确保至少检测两端点
        num_steps = max(2, int(dist / step))
        
        for i in range(num_steps + 1):
            t = i / num_steps
            # 3D 空间线性插值
            pt = p1 + t * (p2 - p1) 
            if self.is_point_in_obstacle(pt, safe_margin):
                return True
                
        return False

    def draw_environment_3d(self, ax=None):
        """ 绘制 3D 环境地图 """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.set_zlim(self.z_bounds)
        
        # 调整3D视角比例，避免Z轴被拉伸得太夸张
        ax.set_box_aspect([1, 1, 0.4]) 
        
        ax.set_title('ZJU Haining Campus - 3D UAV Environment', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Altitude (Z) / m')

        # 绘制障碍物 (3D 建筑物)
        for obs in self.obstacles:
            z_min = obs['z_min']
            z_max = obs['z_max']
            
            if obs['type'] == 'circle':
                center = obs['center']
                r = obs['radius']
                
                # 1. 产生圆柱侧面的网格
                z_side = np.linspace(z_min, z_max, 2)
                theta = np.linspace(0, 2*np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z_side)
                x_side = center[0] + r * np.cos(theta_grid)
                y_side = center[1] + r * np.sin(theta_grid)
                # 绘制侧面
                ax.plot_surface(x_side, y_side, z_grid, color='#5c6bc0', alpha=0.6, edgecolor='none')
                
                # 2. 产生圆柱上下底面（盖子）的网格
                r_vals = np.linspace(0, r, 2)
                r_grid, theta_grid_cap = np.meshgrid(r_vals, theta)
                x_cap = center[0] + r_grid * np.cos(theta_grid_cap)
                y_cap = center[1] + r_grid * np.sin(theta_grid_cap)
                
                # 绘制底面 (z = z_min)
                ax.plot_surface(x_cap, y_cap, np.full(x_cap.shape, z_min), color='#5c6bc0', alpha=0.6, edgecolor='none')
                # 绘制顶面 (z = z_max)
                ax.plot_surface(x_cap, y_cap, np.full(x_cap.shape, z_max), color='#5c6bc0', alpha=0.6, edgecolor='none')
                
                
            elif obs['type'] == 'rect':
                # 绘制 3D 长方体 (处理了旋转)
                bl = obs['bottom_left']
                w, h = obs['width'], obs['height']
                angle = math.radians(obs.get('angle', 0.0))
                cos_t, sin_t = math.cos(angle), math.sin(angle)
                
                # 计算底面4个角点
                c0 = np.array([bl[0], bl[1]])
                c1 = np.array([bl[0] + w*cos_t, bl[1] + w*sin_t])
                c2 = np.array([bl[0] + w*cos_t - h*sin_t, bl[1] + w*sin_t + h*cos_t])
                c3 = np.array([bl[0] - h*sin_t, bl[1] + h*cos_t])
                
                # 构建 8 个顶点
                verts = []
                for z in [z_min, z_max]:
                    verts.extend([[c0[0], c0[1], z], [c1[0], c1[1], z], [c2[0], c2[1], z], [c3[0], c3[1], z]])
                
                # 定义 6 个面
                faces = [
                    [verts[0], verts[1], verts[2], verts[3]], # 底面
                    [verts[4], verts[5], verts[6], verts[7]], # 顶面
                    [verts[0], verts[1], verts[5], verts[4]], # 侧面1
                    [verts[1], verts[2], verts[6], verts[5]], # 侧面2
                    [verts[2], verts[3], verts[7], verts[6]], # 侧面3
                    [verts[3], verts[0], verts[4], verts[7]]  # 侧面4
                ]
                ax.add_collection3d(Poly3DCollection(faces, facecolors='#5c6bc0', linewidths=0.5, edgecolors='black', alpha=0.6))
            
        # 绘制巡检目标区域 (画在地面 Z=0 的绿色虚线圆)
        for target in self.target_areas:
            t_center = target['center']
            t_r = target['radius']
            theta = np.linspace(0, 2*np.pi, 50)
            x_line = t_center[0] + t_r * np.cos(theta)
            y_line = t_center[1] + t_r * np.sin(theta)
            z_line = np.zeros_like(x_line)
            ax.plot(x_line, y_line, z_line, color='#43a047', linestyle='--', linewidth=2)
            ax.text(t_center[0], t_center[1], 0, target['name'], color='#2e7d32', fontweight='bold')

        # 绘制起点和终点 (放在 Z=0，也可以根据需要调整)
        ax.scatter(*self.start_point, color='#fbc02d', s=100, marker='*', label='Start Gate', edgecolors='black', zorder=5)
        ax.scatter(*self.end_point, color='#d32f2f', s=80, marker='^', label='End Gate', zorder=5)
        
        ax.legend(loc='upper left')
        return ax

# ==========================================
# 3D 环境本地测试代码
# ==========================================
if __name__ == "__main__":
    # 注意：确保文件路径与你的 json5 对应
    env = UAVEnvironment3D('maps/haining.json5')
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    env.draw_environment_3d(ax)
    
    # ---------------- 3D 碰撞检测测试 ----------------
    
    # 路线1: 【安全航线】直接拔高到 30m 高空，径直飞跃中心建筑群 (最高建筑为 25m)
    p_safe_1 = np.array([43.0, 3.0, 30.0])
    p_safe_2 = np.array([43.0, 50.0, 30.0]) # 飞跃中心圆塔 (高 25m)
    p_safe_3 = np.array([51.0, 94.0, 30.0]) # 飞向终点
    
    # 路线2: 【穿模航线】在 5m 低空，直线横穿中心建筑群
    p_collide_1 = np.array([43.0, 3.0, 5.0])
    p_collide_2 = np.array([43.0, 50.0, 5.0]) # 必定撞楼
    p_collide_3 = np.array([51.0, 94.0, 5.0])
    
    # 检测相交 (任意一段碰撞即为 True)
    safe_collision = env.is_segment_collision(p_safe_1, p_safe_2) or env.is_segment_collision(p_safe_2, p_safe_3)
    collide_collision = env.is_segment_collision(p_collide_1, p_collide_2) or env.is_segment_collision(p_collide_2, p_collide_3)
    
    print("-" * 40)
    print(f"✅ 测试 - 高空安全航线检测 (应为 False): {safe_collision}")
    print(f"❌ 测试 - 低空穿模航线检测 (应为 True): {collide_collision}")
    print("-" * 40)
    
    # 在 3D 图上画出这两条测试航线
    ax.plot([p_safe_1[0], p_safe_2[0], p_safe_3[0]], 
            [p_safe_1[1], p_safe_2[1], p_safe_3[1]], 
            [p_safe_1[2], p_safe_2[2], p_safe_3[2]], 
            color='#4caf50', linestyle='-', linewidth=2.5, label='Safe Test Path (Alt: 30m)')
            
    ax.plot([p_collide_1[0], p_collide_2[0], p_collide_3[0]], 
            [p_collide_1[1], p_collide_2[1], p_collide_3[1]], 
            [p_collide_1[2], p_collide_2[2], p_collide_3[2]], 
            color='#f44336', linestyle='--', linewidth=2.5, label='Collision Path (Alt: 5m)')
            
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()