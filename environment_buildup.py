import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyjson5
import math

class UAVEnvironment2D:
    """通用地图环境：读取JSON配置并提供碰撞与绘图功能"""
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

    def calculate_distance(self, point1, point2):
        """ 计算两点之间的欧氏距离 """
        return np.linalg.norm(point1 - point2)

    def is_point_in_obstacle(self, point, safe_margin=0.0):
        """ 检测单个点是否在障碍物内部 """
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                if self.calculate_distance(point, obs['center']) <= (obs['radius'] + safe_margin):
                    return True

            elif obs['type'] == 'rect':
                bl = obs['bottom_left']
                w = obs['width']
                h = obs['height']
                angle_deg = obs.get('angle', 0.0) # 兼容没有角度的旧数据
                
                # 平移到原点
                dx = point[0] - bl[0]
                dy = point[1] - bl[1]
                
                # 坐标反向旋转
                theta = math.radians(-angle_deg)
                rx = dx * math.cos(theta) - dy * math.sin(theta)
                ry = dx * math.sin(theta) + dy * math.cos(theta)
                
                # 判断反旋转后的点是否在 AABB 内
                if -safe_margin <= rx <= w + safe_margin and -safe_margin <= ry <= h + safe_margin:
                    return True
        return False

    def _ccw(self, A, B, C):
        """ 辅助函数：判断三点是否逆时针 """
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _segment_intersect(self, A, B, C, D):
        """ 辅助函数: 判断线段AB与线段CD是否相交 """
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)

    def is_segment_collision(self, p1, p2, safe_margin=0.0):
        """
        【核心函数】检测线段(p1->p2)是否与任何障碍物(矩形或圆形)相交
        """
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                # 圆形碰撞检测
                center = obs['center']
                radius = obs['radius'] + safe_margin
                line_vec = p2 - p1
                point_vec = center - p1
                line_len_sq = np.dot(line_vec, line_vec)
                
                if line_len_sq == 0:
                    distance = self.calculate_distance(center, p1)
                else:
                    t = max(0.0, min(1.0, np.dot(point_vec, line_vec) / line_len_sq))
                    projection = p1 + t * line_vec
                    distance = self.calculate_distance(center, projection)
                if distance <= radius:
                    return True
                    
            elif obs['type'] == 'rect':
                bl = obs['bottom_left']
                w = obs['width']
                h = obs['height']
                angle_deg = obs.get('angle', 0.0)
                theta = math.radians(angle_deg)
                
                # 计算旋转后的四个顶点 (未加 safe_margin 的基础顶点)
                # 顺序：左下, 右下, 右上, 左上
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                
                corners = [
                    np.array([bl[0], bl[1]]),
                    np.array([bl[0] + w * cos_t, bl[1] + w * sin_t]),
                    np.array([bl[0] + w * cos_t - h * sin_t, bl[1] + w * sin_t + h * cos_t]),
                    np.array([bl[0] - h * sin_t, bl[1] + h * cos_t])
                ]
                
                # 1. 检查线段端点是否直接在矩形内 (复用前面修改过的点检测逻辑)
                if self.is_point_in_obstacle(p1, safe_margin) or self.is_point_in_obstacle(p2, safe_margin):
                    return True
                
                # 2. 检查线段是否与矩形的四条边相交 (直接使用旋转后的 corners)
                # 注意：如果 safe_margin > 0，严格来说需要将 4 条边向外平移扩展。
                # 简单的近似做法是对计算碰撞时的物体尺寸做膨胀，或者依靠上面端点检测时的 safe_margin。
                edges = [
                    (corners[0], corners[1]), (corners[1], corners[2]),
                    (corners[2], corners[3]), (corners[3], corners[0])
                ]
                for q1, q2 in edges:
                    if self._segment_intersect(p1, p2, q1, q2):
                        return True
                        
        return False

    def draw_environment(self, ax=None):
        """ 绘制基础环境地图 """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.set_aspect('equal')
        ax.set_title('ZJU Haining Campus - 2D UAV Path Planning', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # 绘制浅蓝色背景
        ax.set_faceclass = '#e6f3ff'
        ax.add_patch(patches.Rectangle((0,0), self.x_bounds[1], self.y_bounds[1], color='#eef7ff', zorder=0))

        # 绘制障碍物 (建筑物)
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                patch = plt.Circle(obs['center'], obs['radius'], color='#5c6bc0', alpha=0.8, ec='black')
                ax.add_patch(patch)
            elif obs['type'] == 'rect':
                angle_deg = obs.get('angle', 0.0)
                patch = patches.Rectangle(
                    obs['bottom_left'], obs['width'], obs['height'], 
                    angle=angle_deg,  # 传入旋转角度
                    color='#5c6bc0', alpha=0.8, ec='black'
                )
                ax.add_patch(patch)
            
        # 绘制巡检目标区域 (绿色虚线)
        for target in self.target_areas:
            circle = plt.Circle(target['center'], target['radius'], color='#43a047', 
                                fill=False, linestyle='--', linewidth=2)
            ax.add_patch(circle)
            ax.text(target['center'][0], target['center'][1]+target['radius']+1, 
                    target['name'], ha='center', color='#2e7d32', fontweight='bold')

        # 绘制起点(南门)和终点(北门)
        ax.plot(*self.start_point, '*', color='#fbc02d', markersize=15, label='Start Gate', markeredgecolor='black')
        ax.plot(*self.end_point, '^', color='#d32f2f', markersize=12, label='End Gate')
        
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')
        return ax

# ==========================================
# 本地测试代码 (仅zxy调试时使用)
# ==========================================
if __name__ == "__main__":
    env = UAVEnvironment2D('maps/haining.json5')
    
    # 1. 绘制环境
    fig, ax = plt.subplots(figsize=(10, 10))
    env.draw_environment(ax)
    
    # 2. 测试碰撞检测算法
    # 路线1: 绕过障碍物、顺路经过东西目标区域的【安全折线】
    p_safe_1 = np.array([43.0, 3.0])
    p_safe_2 = np.array([43.0, 28.0]) # 直行穿过南侧两座大建筑的缝隙
    p_safe_3 = np.array([25.0, 28.0]) # 直角左转
    p_safe_4 = np.array([25.0, 45.0]) # 直行朝北
    p_safe_5 = np.array([50.0, 68.0]) # 越过中部教学群上方
    p_safe_6 = np.array([78.0, 70.0]) # 进入东侧巡检区
    p_safe_7 = np.array([51.0, 94.0]) # 飞向终点
    
    # 路线2: 故意横穿中心障碍圆的【碰撞折线】
    p_collide_1 = np.array([43.0, 3.0])
    p_collide_2 = np.array([43.0, 50.0]) # 这一段会精准穿过 [43, 38] 圆形障碍物
    p_collide_3 = np.array([51.0, 94.0])
    
    # 检测所有安全线段
    safe_collision = (env.is_segment_collision(p_safe_1, p_safe_2) or 
                      env.is_segment_collision(p_safe_2, p_safe_3) or 
                      env.is_segment_collision(p_safe_3, p_safe_4) or 
                      env.is_segment_collision(p_safe_4, p_safe_5) or 
                      env.is_segment_collision(p_safe_5, p_safe_6) or
                      env.is_segment_collision(p_safe_6, p_safe_7))
                      
    # 检测碰撞线段
    collide_collision = (env.is_segment_collision(p_collide_1, p_collide_2) or 
                         env.is_segment_collision(p_collide_2, p_collide_3))
    
    print("-" * 30)
    print(f"✅ 测试 - 多段安全折线相交检测 (应为False): {safe_collision}")
    print(f"❌ 测试 - 碰撞线段相交检测 (应为True): {collide_collision}")
    print("-" * 30)
    
    # 在图上画出这两条测试路线
    #safe_xs = [p_safe_1[0], p_safe_2[0], p_safe_3[0], p_safe_4[0], p_safe_5[0], p_safe_6[0], p_safe_7[0]]
    #safe_ys = [p_safe_1[1], p_safe_2[1], p_safe_3[1], p_safe_4[1], p_safe_5[1], p_safe_6[1], p_safe_7[1]]
    #ax.plot(safe_xs, safe_ys, color='#4caf50', linestyle='-', linewidth=2.5, label='Safe Test Path')
            
    #collide_xs = [p_collide_1[0], p_collide_2[0], p_collide_3[0]]
    #collide_ys = [p_collide_1[1], p_collide_2[1], p_collide_3[1]]
    #ax.plot(collide_xs, collide_ys, color='#f44336', linestyle='--', linewidth=2.5, label='Collision Test Path')
            
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()