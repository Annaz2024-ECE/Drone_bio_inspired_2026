import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class UAVEnvironment2D:
    def __init__(self):
        """
        初始化 2D 无人机巡检环境 (基于ZJU海宁校区地图简化)
        """
        # 1. 地图边界 (0~100)
        self.x_bounds = [0, 100]
        self.y_bounds = [0, 100]
        
        # 2. 起点和终点 (模拟从南校门飞往北校门)
        self.start_point = np.array([43.0, 3.0])
        self.end_point = np.array([51.0, 94.0])
        
        # 3. 障碍物集合 (支持 'rect' 和 'circle')
        # 依据海宁校区平面图，将建筑群简化为矩形和圆形
        self.obstacles = [
            # ==== 南侧建筑群 ====
            {'type': 'rect', 'bottom_left': np.array([29, 11]), 'width': 10, 'height': 15},   # 1A-1E 
            {'type': 'rect', 'bottom_left': np.array([47, 11]), 'width': 10, 'height': 15},   # 2A-2E 
            {'type': 'rect', 'bottom_left': np.array([30, 31]), 'width': 4, 'height': 8},     # 3A,3B 
            {'type': 'rect', 'bottom_left': np.array([52, 31]), 'width': 5, 'height': 10},    # 4 

            {'type': 'rect', 'bottom_left': np.array([36, 35]), 'width': 3, 'height': 3},     # 7 
            {'type': 'rect', 'bottom_left': np.array([47, 35]), 'width': 3, 'height': 3},     # 8 
            {'type': 'circle', 'center': np.array([37, 32]), 'radius': 1},                    # 5 
            {'type': 'circle', 'center': np.array([48, 32]), 'radius': 1},                    # 6 
            {'type': 'circle', 'center': np.array([43, 38]), 'radius': 2},                    # 9 

            
            # ==== 西侧教学楼 ====
            {'type': 'rect', 'bottom_left': np.array([13, 51]), 'width': 5, 'height': 4},     # 25 
            {'type': 'rect', 'bottom_left': np.array([15, 40]), 'width': 9, 'height': 5},     # 26 
            {'type': 'rect', 'bottom_left': np.array([20, 36]), 'width': 4, 'height': 4},     # 26 
            {'type': 'rect', 'bottom_left': np.array([32, 60]), 'width': 4, 'height': 6},     # 24 

            # ==== 东侧教学楼、图书馆、食堂和书院 ====
            {'type': 'rect', 'bottom_left': np.array([51, 63]), 'width': 6, 'height': 4},     # 21 
            {'type': 'rect', 'bottom_left': np.array([59, 64]), 'width': 5, 'height': 4},     # 20 
            {'type': 'rect', 'bottom_left': np.array([61, 57]), 'width': 3, 'height': 5},     # 19 
            {'type': 'circle', 'center': np.array([56, 59]), 'radius': 3},                    # 18

            {'type': 'rect', 'bottom_left': np.array([59, 44]), 'width': 5, 'height': 7},     # 16

            {'type': 'rect', 'bottom_left': np.array([68, 58]), 'width': 7, 'height': 10},    # 15
            {'type': 'rect', 'bottom_left': np.array([79, 58]), 'width': 7, 'height': 10},    # 12
            {'type': 'rect', 'bottom_left': np.array([75, 42]), 'width': 10, 'height': 12},   # 11
            {'type': 'rect', 'bottom_left': np.array([69, 29]), 'width': 12, 'height': 10},   # 10

            # ==== 北侧体育馆 ====
            {'type': 'rect', 'bottom_left': np.array([52, 74]), 'width': 6, 'height': 13},    # 23
            {'type': 'rect', 'bottom_left': np.array([72, 86]), 'width': 3, 'height': 3}      # 22
        ]
        
        # 4. 目标巡检区域（假设需要顺路侦视的重点区域）
        self.target_areas = [
            {'center': np.array([22.0, 40.0]), 'radius': 6.0, 'name': 'West Area'},
            {'center': np.array([78.0, 70.0]), 'radius': 6.0, 'name': 'East Area'}
        ]

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
                min_x = obs['bottom_left'][0] - safe_margin
                max_x = obs['bottom_left'][0] + obs['width'] + safe_margin
                min_y = obs['bottom_left'][1] - safe_margin
                max_y = obs['bottom_left'][1] + obs['height'] + safe_margin
                if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
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
                # 矩形碰撞检测 (AABB bounding box)
                min_x = obs['bottom_left'][0] - safe_margin
                max_x = obs['bottom_left'][0] + obs['width'] + safe_margin
                min_y = obs['bottom_left'][1] - safe_margin
                max_y = obs['bottom_left'][1] + obs['height'] + safe_margin
                
                # 1. 检查线段端点是否直接在矩形内
                if (min_x <= p1[0] <= max_x and min_y <= p1[1] <= max_y) or \
                   (min_x <= p2[0] <= max_x and min_y <= p2[1] <= max_y):
                    return True
                
                # 2. 检查线段是否与矩形的四条边相交
                corners = [
                    np.array([min_x, min_y]), np.array([max_x, min_y]),
                    np.array([max_x, max_y]), np.array([min_x, max_y])
                ]
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
        ax.add_patch(patches.Rectangle((0,0), 100, 100, color='#eef7ff', zorder=0))

        # 绘制障碍物 (建筑物)
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                patch = plt.Circle(obs['center'], obs['radius'], color='#5c6bc0', alpha=0.8, ec='black')
                ax.add_patch(patch)
            elif obs['type'] == 'rect':
                patch = patches.Rectangle(obs['bottom_left'], obs['width'], obs['height'], 
                                          color='#5c6bc0', alpha=0.8, ec='black')
                ax.add_patch(patch)
            
        # 绘制巡检目标区域 (绿色虚线)
        for target in self.target_areas:
            circle = plt.Circle(target['center'], target['radius'], color='#43a047', 
                                fill=False, linestyle='--', linewidth=2)
            ax.add_patch(circle)
            ax.text(target['center'][0], target['center'][1]+target['radius']+1, 
                    target['name'], ha='center', color='#2e7d32', fontweight='bold')

        # 绘制起点(南门)和终点(北门)
        ax.plot(*self.start_point, '^', color='#d32f2f', markersize=12, label='South Gate (Start)')
        ax.plot(*self.end_point, '*', color='#fbc02d', markersize=15, label='North Gate (End)', markeredgecolor='black')
        
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')
        return ax

# ==========================================
# 本地测试代码 (仅zxy调试时使用)
# ==========================================
if __name__ == "__main__":
    env = UAVEnvironment2D()
    
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
    safe_xs = [p_safe_1[0], p_safe_2[0], p_safe_3[0], p_safe_4[0], p_safe_5[0], p_safe_6[0], p_safe_7[0]]
    safe_ys = [p_safe_1[1], p_safe_2[1], p_safe_3[1], p_safe_4[1], p_safe_5[1], p_safe_6[1], p_safe_7[1]]
    ax.plot(safe_xs, safe_ys, color='#4caf50', linestyle='-', linewidth=2.5, label='Safe Test Path')
            
    collide_xs = [p_collide_1[0], p_collide_2[0], p_collide_3[0]]
    collide_ys = [p_collide_1[1], p_collide_2[1], p_collide_3[1]]
    ax.plot(collide_xs, collide_ys, color='#f44336', linestyle='--', linewidth=2.5, label='Collision Test Path')
            
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()