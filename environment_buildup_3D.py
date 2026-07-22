import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyjson5
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d
from shapely.geometry import Point, LineString, Polygon
import shapely.affinity as affinity
import os


class UAVEnvironment3D:
    """升级版3D地图环境：读取JSON配置，提供3D碰撞检测与3D可视化"""
    def __init__(self, json_path=None, data_dict=None):
        # 兼容两种初始化方式：读取文件 或 直接传入字典
        if json_path:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = pyjson5.load(f)
        elif data_dict:
            data = data_dict
        else:
            raise ValueError("必须提供 json_path 或 data_dict")
        
        self.name = data['name']
        self.x_bounds = [0, data['bounds'][0]]
        self.y_bounds = [0, data['bounds'][1]]
        
        # 寻找地图中的最高建筑，用于设置 Z 轴边界
        max_z = 11.0 
        for obs in data['obstacles']:
            max_z = max(max_z, obs.get('z_max', 11.0))
        self.z_bounds = [0, max_z + 1] # 天花板留出 10m 余量

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

        # ==========================================
        # 【Shapely 升级】将 JSON 几何数据预编译为 Shapely 对象
        # ==========================================
        self.shapely_obstacles = []
        for obs in self.obstacles:
            shapely_obs = obs.copy()
            
            if obs['type'] == 'circle':
                # 圆形：利用 Point 缓冲生成多边形
                shapely_obs['poly_2d'] = Point(obs['center'][:2]).buffer(obs['radius'])
                
            elif obs['type'] == 'rect':
                bl = obs['bottom_left'][:2]
                w, h = obs['width'], obs['height']
                angle = obs.get('angle', 0.0)
                
                # 构建未旋转的基础矩形
                rect = Polygon([
                    (bl[0], bl[1]),
                    (bl[0] + w, bl[1]),
                    (bl[0] + w, bl[1] + h),
                    (bl[0], bl[1] + h)
                ])
                # 绕左下角进行旋转
                rotated_rect = affinity.rotate(rect, angle, origin=(bl[0], bl[1]), use_radians=False)
                shapely_obs['poly_2d'] = rotated_rect
            
            elif obs['type'] == 'polygon':
                # 多边形最简单，直接把点阵传给 Shapely 即可
                shapely_obs['poly_2d'] = Polygon(obs['points'])
                
            self.shapely_obstacles.append(shapely_obs)

        # ==========================================
        # 【新增】1. 假设无人机相机水平视场角 (FOV) 为 90度
        # 主流消费级无人机（如大疆 Mavic 系列）HFOV 通常在 70°~85°，取 90° 好算且留有余量
        # ==========================================
        self.default_camera_fov_deg = 90.0  

        # ==========================================
        # 【核心修改】2. 动态计算巡检区域 (Target Areas) 的 Z 轴范围
        # ==========================================
        self.target_areas = data['target_areas']
        for target in self.target_areas:
            target['center'] = np.array(target['center'])
            
            # 获取该目标对应的半径
            radius = target['radius']
            
            # 如果有特殊目标需要自定义 FOV，可以从 json 读取，没有则用全局默认
            fov_deg = target.get('camera_fov', self.default_camera_fov_deg)
            half_fov_rad = math.radians(fov_deg / 2.0)
            
            # ---- 物理公式：覆盖半径 R 所需的最佳悬停高度 ----
            # 几何关系：R = H * tan(FOV/2)  =>  H = R / tan(FOV/2)
            optimal_height = radius / math.tan(half_fov_rad) - 0.5
            
            # 设置绝对安全最低高度，防止飞得太低撞到地面凸起物（如路灯、行人）
            optimal_height = max(optimal_height, 1.5) 
            target['z_min'] = optimal_height
            
            # 2. z_max：设置为最佳高度 + 8米（或 1.5倍），飞高永远能覆盖更大范围，
            #    且高空障碍物极少，给算法极大的自由度去“命中”目标。
            target['z_max'] = optimal_height + 3
            
            # (可选) 为了方便你调试，可以把计算出的值打出来看看
            # print(f"目标 {target['name']}: 半径={radius}m, 最佳高度={optimal_height:.1f}m, 设定范围=[{target['z_min']:.1f}, {target['z_max']:.1f}]")

    def calculate_distance(self, point1, point2):
        """ 计算两点之间的 3D 欧氏距离 """
        return np.linalg.norm(point1 - point2)

    def is_point_in_obstacle(self, point, safe_margin=0.0):
        """ 
        【Shapely版】检测 3D 空间中的单点是否在障碍物内部
        """
        pt_2d = Point(point[0], point[1])
        
        for obs in self.shapely_obstacles:
            # 1. Z轴高度拦截 (最廉价的计算)
            z_min = obs['z_min'] - safe_margin
            z_max = obs['z_max'] + safe_margin
            if not (z_min <= point[2] <= z_max):
                continue
                
            # 2. 调用 Shapely 底层 C 引擎计算 2D 距离
            if obs['poly_2d'].distance(pt_2d) <= safe_margin:
                return True
                
        return False

    def is_segment_collision(self, p1, p2, safe_margin=0.0, step=0.5):
        """
        【Shapely极速版 3D线段检测】
        使用 Broad-phase (宽相) + Narrow-phase (窄相插值) 架构
        """
        dist = self.calculate_distance(p1, p2)
        if dist == 0:
            return self.is_point_in_obstacle(p1, safe_margin)

        # ==========================================
        # 核心提速：2D 宽相检测 (Broad-phase)
        # ==========================================
        # 将 3D 航线直接拍扁成 2D 线段
        line_2d = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
        
        potential_obstacles = []
        for obs in self.shapely_obstacles:
            # 如果在 2D 俯视图上，航线连这栋楼的边都擦不到，直接剔除！
            if obs['poly_2d'].distance(line_2d) <= safe_margin:
                potential_obstacles.append(obs)
                
        # 如果 2D 投影完全碰不到任何障碍物，航线绝对安全，直接返回！
        # (这省去了原来 90% 以上的 3D 插值计算点)
        if not potential_obstacles:
            return False

        # ==========================================
        # 窄相检测：仅对有 2D 嫌疑的建筑进行 3D 高度插值判定
        # ==========================================
        num_steps = max(2, int(dist / step))
        for i in range(num_steps + 1):
            t = i / num_steps
            pt = p1 + t * (p2 - p1) 
            pt_2d = Point(pt[0], pt[1])
            
            for obs in potential_obstacles:
                z_min = obs['z_min'] - safe_margin
                z_max = obs['z_max'] + safe_margin
                # 高度穿模，并且水平面距离过近，才算真撞
                if z_min <= pt[2] <= z_max:
                    if obs['poly_2d'].distance(pt_2d) <= safe_margin:
                        return True
                        
        return False

    def draw_environment_3d(self, ax=None, theme='ORIGINAL'):
        """ 
        绘制 3D 环境地图
        :param theme: 'ORIGINAL' (JSON原色), 'REGION' (东西分色), 'HEIGHT' (蓝色高度层级), 'BLUEPRINT' (蓝图)
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.set_zlim(self.z_bounds)
        ax.set_box_aspect([1, 1, 0.4]) 
        
        title_suffix = "" if theme == 'ORIGINAL' else f" ({theme} Theme)"
        ax.set_title(f'{self.name} - 3D Environment{title_suffix}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Altitude (Z)')

        # 绘制障碍物 (3D 建筑物)
        for obs in self.obstacles:
            z_min = obs['z_min']
            z_max = obs['z_max']
            
            # ==========================================
            # 强制应用 HEIGHT 高度热力图颜色 (彻底忽略 JSON 中的颜色)
            # ==========================================
            if z_max <= 1.5:
                color = '#e3f2fd'  # 矮房：极浅蓝
            elif z_max <= 3.0:
                color = '#90caf9'  # 裙楼：浅蓝色
            elif z_max <= 5.0:
                color = '#1e88e5'  # 高楼：深蓝色
            else:
                color = '#0d47a1'  # 地标：藏青色/极深蓝
            # ==========================================
            
            if obs['type'] == 'circle':
                center = obs['center']
                r = obs['radius']
                theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
                pts = np.array([[center[0] + r * math.cos(t), center[1] + r * math.sin(t)] for t in theta])
                
                verts_bottom = [[p[0], p[1], z_min] for p in pts]
                verts_top = [[p[0], p[1], z_max] for p in pts]
                
                faces = [verts_bottom, verts_top] 
                num_pts = len(pts)
                for i in range(num_pts):
                    next_i = (i + 1) % num_pts
                    side_face = [verts_bottom[i], verts_bottom[next_i], verts_top[next_i], verts_top[i]]
                    faces.append(side_face)
                    
                ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.85))

            elif obs['type'] == 'rect':
                bl = obs['bottom_left']
                w, h = obs['width'], obs['height']
                angle = math.radians(obs.get('angle', 0.0))
                cos_t, sin_t = math.cos(angle), math.sin(angle)
                
                c0 = np.array([bl[0], bl[1]])
                c1 = np.array([bl[0] + w*cos_t, bl[1] + w*sin_t])
                c2 = np.array([bl[0] + w*cos_t - h*sin_t, bl[1] + w*sin_t + h*cos_t])
                c3 = np.array([bl[0] - h*sin_t, bl[1] + h*cos_t])
                
                verts = []
                for z in [z_min, z_max]:
                    verts.extend([[c0[0], c0[1], z], [c1[0], c1[1], z], [c2[0], c2[1], z], [c3[0], c3[1], z]])
                
                faces = [
                    [verts[0], verts[1], verts[2], verts[3]],
                    [verts[4], verts[5], verts[6], verts[7]],
                    [verts[0], verts[1], verts[5], verts[4]],
                    [verts[1], verts[2], verts[6], verts[5]],
                    [verts[2], verts[3], verts[7], verts[6]],
                    [verts[3], verts[0], verts[4], verts[7]] 
                ]
                ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.85))
            
            elif obs['type'] == 'polygon':
                pts = np.array(obs['points'])
                verts_bottom = [[p[0], p[1], z_min] for p in pts]
                verts_top = [[p[0], p[1], z_max] for p in pts]
                
                faces = [verts_bottom, verts_top]
                num_pts = len(pts)
                for i in range(num_pts):
                    next_i = (i + 1) % num_pts
                    side_face = [verts_bottom[i], verts_bottom[next_i], verts_top[next_i], verts_top[i]]
                    faces.append(side_face)
                    
                ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.85))
            
        # 绘制巡检目标区域圆柱体
        for target in self.target_areas:
            t_center = target['center']
            t_r = target['radius']
            z_min = target.get('z_min', 0)
            z_max = target.get('z_max', 0)

            u = np.linspace(0, 2*np.pi, 30)
            v = np.linspace(z_min, z_max, 10)
            u, v = np.meshgrid(u, v)
            X = t_center[0] + t_r * np.cos(u)
            Y = t_center[1] + t_r * np.sin(u)
            Z = v
            ax.plot_surface(X, Y, Z, alpha=0.2, color='green', edgecolor='none')

            theta = np.linspace(0, 2*np.pi, 50)
            x_circ = t_center[0] + t_r * np.cos(theta)
            y_circ = t_center[1] + t_r * np.sin(theta)
            ax.plot(x_circ, y_circ, np.full_like(theta, z_min), color='#43a047', linewidth=2, alpha=0.5)
            ax.plot(x_circ, y_circ, np.full_like(theta, z_max), color='#43a047', linewidth=2, alpha=0.5)

            ax.text(t_center[0], t_center[1], (z_min + z_max) / 2,
                    target['name'], color='#2e7d32', fontweight='bold')

        ax.scatter(*self.start_point, color='#fbc02d', s=100, marker='*', label='Start Gate', edgecolors='black', zorder=5)
        ax.scatter(*self.end_point, color='#d32f2f', s=80, marker='^', label='End Gate', zorder=5)
        
        ax.legend(loc='upper left')
        return ax

# # ==========================================
# # 3D 环境本地测试代码
# # ==========================================
# if __name__ == "__main__":
#     # 注意：确保文件路径与你的 json5 对应
#     env = UAVEnvironment3D('maps/medium_map.json5')
    
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     env.draw_environment_3d(ax)

#     # ax.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()


# ==========================================
# 随机地图生成工具类
# ==========================================
class RandomMapGenerator:
    @staticmethod
    def generate_random_targets(base_json_path, num_targets=6, radius_range=(2.0, 5.0), safe_margin=1.0):
        """
        保持基础地图不变，随机生成不碰撞、不重叠的巡检目标。
        
        :param base_json_path: 基础地图文件路径 (提供建筑物、起终点、边界)
        :param num_targets: 需要生成的巡检区数量 (默认 6)
        :param radius_range: 巡检区半径范围 (min_radius, max_radius)
        :param safe_margin: 目标区域与建筑物之间、目标与目标之间留出的最小安全缓冲距离
        :return: 包含随机目标的新字典 (可以直接丢给 UAVEnvironment3D)
        """
        # 1. 加载基础地图数据
        with open(base_json_path, 'r', encoding='utf-8') as f:
            data = pyjson5.load(f)
            
        x_bounds = data['bounds'][0]
        y_bounds = data['bounds'][1]
        
        # 2. 将所有障碍物转换为 Shapely 对象，以便判断不交叠
        shapely_obstacles = []
        for obs in data['obstacles']:
            if obs['type'] == 'circle':
                poly = Point(obs['center'][:2]).buffer(obs['radius'])
            elif obs['type'] == 'rect':
                bl = obs['bottom_left'][:2]
                w, h = obs['width'], obs['height']
                angle = obs.get('angle', 0.0)
                rect = Polygon([
                    (bl[0], bl[1]), (bl[0] + w, bl[1]), 
                    (bl[0] + w, bl[1] + h), (bl[0], bl[1] + h)
                ])
                poly = affinity.rotate(rect, angle, origin=(bl[0], bl[1]), use_radians=False)
            elif obs['type'] == 'polygon':
                poly = Polygon(obs['points'])
            shapely_obstacles.append(poly)
            
        # 3. 开始投点拒绝算法 (Generate & Test)
        new_targets = []
        attempts = 0
        max_attempts = 5000  # 防止死循环
        
        while len(new_targets) < num_targets and attempts < max_attempts:
            attempts += 1
            
            # 随机生成半径和圆心 (注意边界限制，不能让圆切到地图外)
            r = np.random.uniform(radius_range[0], radius_range[1])
            cx = np.random.uniform(r + safe_margin, x_bounds - r - safe_margin)
            cy = np.random.uniform(r + safe_margin, y_bounds - r - safe_margin)
            
            target_poly = Point(cx, cy).buffer(r + safe_margin)
            
            is_valid = True
            
            # 校验一：不能撞击建筑物
            for obs_poly in shapely_obstacles:
                if target_poly.intersects(obs_poly):
                    is_valid = False
                    break
                    
            if not is_valid: continue
            
            # 校验二：不能和已经生成的目标重叠
            for existing_target in new_targets:
                ex_pt = Point(existing_target['center'][0], existing_target['center'][1])
                ex_poly = ex_pt.buffer(existing_target['radius'] + safe_margin)
                if target_poly.intersects(ex_poly):
                    is_valid = False
                    break
                    
            if not is_valid: continue
            
            # 校验三：不能覆盖起终点
            start_pt = Point(data['start_point'][:2])
            end_pt = Point(data['end_point'][:2])
            if target_poly.intersects(start_pt) or target_poly.intersects(end_pt):
                continue
                
            # 校验通过，保存目标
            new_targets.append({
                "center": [float(cx), float(cy)],
                "radius": float(r),
                "name": f"L{len(new_targets) + 1}"
            })
            
        if len(new_targets) < num_targets:
            print(f"警告: 只成功生成了 {len(new_targets)} 个目标 (请求 {num_targets} 个)。可能是地图太满。")
            
        # 4. 更新并返回新的地图字典
        data['target_areas'] = new_targets
        data['name'] = f"{data['name']} (Randomized)"
        return data


# ==========================================
# 3D 环境本地测试代码 (现在测试随机地图生成)
# ==========================================
if __name__ == "__main__":
    # 使用基础地图模板生成随机目标地图
    # （这里的文件路径填你本地真实存在的如 'maps/easy_map.json5'）
    base_map_path = 'maps/medium_map.json5' 
    
    if os.path.exists(base_map_path):
        random_map_dict = RandomMapGenerator.generate_random_targets(
            base_json_path=base_map_path,
            num_targets=6,          # 你要多少个巡检区
            radius_range=(2.0, 5.0), # 大小的分布
            safe_margin=1.0          # 保留1米的缓冲不贴脸
        )
        
        # 使用生成的字典初始化环境
        env = UAVEnvironment3D(data_dict=random_map_dict)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        env.draw_environment_3d(ax)
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"请提供真实的地图路径。找不到文件: {base_map_path}")