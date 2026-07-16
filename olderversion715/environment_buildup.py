import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyjson5
import math
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
import shapely.affinity as affinity

class UAVEnvironment2D:
    """通用地图环境：读取JSON配置并提供碰撞与绘图功能 (Shapely极速版)"""
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = pyjson5.load(f)
        
        self.name = data['name']
        self.x_bounds = [0, data['bounds'][0]]
        self.y_bounds = [0, data['bounds'][1]]
        self.start_point = np.array(data['start_point'])
        self.end_point = np.array(data['end_point'])
        
        # 将列表转换为 Numpy 数组，方便后续计算
        self.obstacles = []
        for obs in data['obstacles']:
            o = obs.copy()
            if 'bottom_left' in o: o['bottom_left'] = np.array(o['bottom_left'])
            if 'center' in o: o['center'] = np.array(o['center'])
            self.obstacles.append(o)
            
        self.target_areas = data['target_areas']
        for target in self.target_areas:
            target['center'] = np.array(target['center'])

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
                rect = ShapelyPolygon([
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
                shapely_obs['poly_2d'] = ShapelyPolygon(obs['points'])
                
            self.shapely_obstacles.append(shapely_obs)

    def calculate_distance(self, point1, point2):
        """ 计算两点之间的欧氏距离 """
        return np.linalg.norm(point1 - point2)

    def is_point_in_obstacle(self, point, safe_margin=0.0):
        """ 
        【Shapely版】检测单个点是否在障碍物内部或安全边界内
        """
        pt_2d = Point(point[0], point[1])
        for obs in self.shapely_obstacles:
            if obs['poly_2d'].distance(pt_2d) <= safe_margin:
                return True
        return False

    def is_segment_collision(self, p1, p2, safe_margin=0.0):
        """
        【Shapely版】检测线段(p1->p2)是否与任何障碍物相交
        2D的检测比3D简单得多，直接计算线段到多边形的距离即可，无需插值！
        """
        line_2d = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
        
        for obs in self.shapely_obstacles:
            if obs['poly_2d'].distance(line_2d) <= safe_margin:
                return True
                
        return False

    def draw_environment(self, ax=None):
        """ 绘制基础环境地图 (支持 Polygon 与 颜色映射) """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.set_aspect('equal')
        ax.set_title(f'{self.name} - 2D UAV Path Planning', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # 绘制浅蓝色背景
        ax.set_facecolor('#e6f3ff')
        ax.add_patch(patches.Rectangle((0,0), self.x_bounds[1], self.y_bounds[1], color='#eef7ff', zorder=0))

        # 绘制障碍物 (建筑物)
        for obs in self.obstacles:
            # 获取颜色，默认使用蓝灰色
            color = obs.get('color', '#5c6bc0')
            
            if obs['type'] == 'circle':
                patch = plt.Circle(obs['center'], obs['radius'], color=color, alpha=0.85, ec='black')
                ax.add_patch(patch)
            elif obs['type'] == 'rect':
                angle_deg = obs.get('angle', 0.0)
                patch = patches.Rectangle(
                    obs['bottom_left'], obs['width'], obs['height'], 
                    angle=angle_deg,
                    color=color, alpha=0.85, ec='black'
                )
                ax.add_patch(patch)
            elif obs['type'] == 'polygon':
                # 新增的多边形绘制逻辑
                patch = patches.Polygon(
                    obs['points'], closed=True, 
                    color=color, alpha=0.85, ec='black'
                )
                ax.add_patch(patch)
            
        # 绘制巡检目标区域 (绿色虚线)
        for target in self.target_areas:
            circle = plt.Circle(target['center'], target['radius'], color='#43a047', 
                                fill=False, linestyle='--', linewidth=2)
            ax.add_patch(circle)
            ax.text(target['center'][0], target['center'][1]+target['radius']+1, 
                    target['name'], ha='center', color='#2e7d32', fontweight='bold')

        # 绘制起点和终点
        ax.plot(*self.start_point[:2], '*', color='#fbc02d', markersize=15, label='Start Gate', markeredgecolor='black', zorder=5)
        ax.plot(*self.end_point[:2], '^', color='#d32f2f', markersize=12, label='End Gate', zorder=5)
        
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper left')
        return ax

# ==========================================
# 本地测试代码
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的紫金港地图 JSON
    env = UAVEnvironment2D('maps/zijingang_2.json5')
    
    # 1. 绘制环境
    fig, ax = plt.subplots(figsize=(12, 10))
    env.draw_environment(ax)
    plt.tight_layout()
    plt.show()
