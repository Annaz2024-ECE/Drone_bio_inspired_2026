import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制底层数学计算引擎只用单线程，防止内存冲突
import numpy as np
import matplotlib.pyplot as plt
from environment_buildup import UAVEnvironment3D  # 【修改1：导入3D环境】
import scipy.interpolate as spl

class PathEvaluator:
    def __init__(self):
        # 实例化3D环境
        self.env = UAVEnvironment3D('maps/haining.json5')
        
        # 基础惩罚权重
        self.penalties = {
            'fatal_collision': 1000000.0,  
            'missed_target': 1000000.0,     
            'sharp_turn': 10000.0,         
            'margin_violation': 5000.0,     
            'altitude_violation': 50000.0   # 飞入地下或飞得太高的惩罚
        }

        self.params = {
            'max_turn_angle': 120.0,     # 3D转弯更复杂，放宽到120度
            'chaikin_iterations': 3,     
            'min_waypoint_dist': 5.0,    
            'margin_layers': [0.5, 0.2]  # 只留两层安全洋葱皮，减少擦边误伤
        }
        
        self.ideal_min_distance = self._calculate_ideal_min_distance()
        print(f" [环境加载] 当前3D地图理论最短直线距离: {self.ideal_min_distance:.1f} 米")

    def _calculate_ideal_min_distance(self):
        """ 计算 起点 -> 所有打卡点 -> 终点 的直线距离 (Numpy 向量运算自动适应3D!) """
        pts = [self.env.start_point]
        
        # 提取所有打卡点
        targets = [t['center'] for t in self.env.target_areas]
        
        # 为了算距离，如果目标点只有2D，给它补个 Z=0
        for i in range(len(targets)):
            if len(targets[i]) == 2:
                targets[i] = np.array([targets[i][0], targets[i][1], 0.0])
                
        targets.sort(key=lambda p: p[0])
        pts.extend(targets)
        pts.append(self.env.end_point)

        dist = 0.0
        for i in range(len(pts) - 1):
            dist += self.env.calculate_distance(pts[i], pts[i+1])
        return dist

    def update_params(self, new_penalties=None, new_params=None):
        if new_penalties: self.penalties.update(new_penalties)
        if new_params: self.params.update(new_params)

    # def generate_chaikin_path(self, waypoints, iterations=4):
    #     """ Chaikin 割角算法：完美兼容3D坐标！ """
    #     pts = np.array(waypoints)
    #     for _ in range(iterations):
    #         new_pts = [pts[0]] 
    #         for i in range(len(pts) - 1):
    #             p0 = pts[i]
    #             p1 = pts[i+1]
    #             Q = 0.75 * p0 + 0.25 * p1
    #             R = 0.25 * p0 + 0.75 * p1
    #             new_pts.extend([Q, R])
    #         new_pts.append(pts[-1]) 
    #         pts = np.array(new_pts)
    #     return pts

    
    def generate_bspline_path(self, waypoints, num_points=100):
        """ 
        【全面升级为 3D B-Spline 插值算法】 
        极致平滑，二阶导数连续，符合真实无人机空气动力学飞行轨迹。
        """
        waypoints = np.array(waypoints)
        
        # 1. 剔除极其接近的重复点，防止插值算法除以0报错
        unique_waypoints = [waypoints[0]]
        for pt in waypoints[1:]:
            if np.linalg.norm(pt - unique_waypoints[-1]) > 0.1:
                unique_waypoints.append(pt)
        unique_waypoints = np.array(unique_waypoints)

        # 2. 检查点数。B样条默认需要至少 4 个点 (阶数 k=3)
        num_wp = len(unique_waypoints)
        if num_wp < 3:
            return unique_waypoints # 点太少，画不出平滑曲线，直接返回原线
            
        k = 3 if num_wp >= 4 else num_wp - 1

        # 3. 提取 3D 空间的 X, Y, Z
        x = unique_waypoints[:, 0]
        y = unique_waypoints[:, 1]
        z = unique_waypoints[:, 2]

        # 4. 计算 B样条参数 
        # s=0 表示强制曲线精确穿过你给定的每一个控制点
        tck, u = spl.splprep([x, y, z], s=0, k=k)

        # 5. 生成均匀分布的新参数，并计算出平滑的三维坐标点
        u_new = np.linspace(0, 1.0, num_points)
        x_new, y_new, z_new = spl.splev(u_new, tck)

        # 6. 把 X, Y, Z 重新拼成 3D 坐标组
        smooth_path = np.column_stack((x_new, y_new, z_new))
        return smooth_path

    def calculate_spacing_penalty(self, raw_waypoints):
        min_dist = self.params.get('min_waypoint_dist', 5.0)
        penalty = 0.0
        for i in range(len(raw_waypoints) - 1):
            dist = np.linalg.norm(raw_waypoints[i+1] - raw_waypoints[i])
            if dist < min_dist:
                penalty += ((min_dist - dist) ** 2) * 5.0
        return penalty

    def calculate_path_length(self, path_points):
        total_length = 0.0
        for i in range(len(path_points) - 1):
            total_length += self.env.calculate_distance(path_points[i], path_points[i+1])
        return total_length

    def calculate_turn_angle(self, p1, p2, p3):
        """ 计算 3D 空间转弯角度 (向量点乘魔法，完美适应3D) """
        v1 = p2 - p1
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def point_to_segment_distance(self, point, seg_a, seg_b):
        line_vec = seg_b - seg_a
        point_vec = point - seg_a
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.linalg.norm(point - seg_a)
        t = max(0.0, min(1.0, np.dot(point_vec, line_vec) / line_len_sq))
        projection = seg_a + t * line_vec
        return np.linalg.norm(point - projection)

    def calculate_target_penalty(self, path_points):
        total_target_penalty = 0.0
        
        for target in self.env.target_areas:
            # 强制把目标点的中心只取 X 和 Y (抛弃 Z 高度)
            center_2d = target['center'][:2] 
            radius = target['radius']
            min_dist_to_target = float('inf')
            
            for i in range(len(path_points) - 1):
                # 将航线线段也强行“踩扁”到 2D 平面
                p1_2d = path_points[i][:2]
                p2_2d = path_points[i+1][:2]
                
                # 计算 2D 投影距离 (只要飞过正上方就算打卡！)
                dist = self.point_to_segment_distance(center_2d, p1_2d, p2_2d)
                if dist < min_dist_to_target:
                    min_dist_to_target = dist
            
            if min_dist_to_target > radius:
                missed_distance = min_dist_to_target - radius
                # 方便3D算法探索
                total_target_penalty += 500000.0 + (missed_distance * 20000.0)
                
        return total_target_penalty

    def calculate_fitness(self, path_points):
        details = {
            'distance': 0.0,          
            'fatal_collision': 0.0,   
            'margin_violation': 0.0,  
            'smoothness': 0.0,        
            'sharp_turn': 0.0,        
            'missed_target': 0.0,
            'altitude_violation': 0.0  # 新增高度惩罚
        }

        # 1. 距离算分
        details['distance'] = self.calculate_path_length(path_points)
        
        margin_layers = self.params.get('margin_layers', [0.5, 0.2])
        layer_penalty = self.penalties.get('margin_violation', 5000.0) / len(margin_layers)
        fatal_penalty = self.penalties.get('fatal_collision', 1000000.0)
        alt_penalty = self.penalties.get('altitude_violation', 50000.0)

        # 2. 高空管制 与 碰撞检测
        for i in range(len(path_points)):
            # 钻地 (撞向地面)
            if path_points[i][2] < 0:
                details['fatal_collision'] += fatal_penalty
            # 突破天际 (超过环境最高限制)
            elif path_points[i][2] > self.env.z_bounds[1]:
                details['altitude_violation'] += alt_penalty

        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            if self.env.is_segment_collision(p1, p2, safe_margin=0.0):
                details['fatal_collision'] += fatal_penalty
                continue 
                
            if not self.env.is_segment_collision(p1, p2, safe_margin=0.5):
                continue

            for m in margin_layers:
                if self.env.is_segment_collision(p1, p2, safe_margin=m):
                    details['margin_violation'] += layer_penalty

        # 3. 急转弯计算 (无需修改)
        max_turn = self.params.get('max_turn_angle', 120.0)
        sharp_turn_pen = self.penalties.get('sharp_turn', 10000.0)

        for i in range(len(path_points) - 2):
            angle = self.calculate_turn_angle(path_points[i], path_points[i+1], path_points[i+2])
            if angle > 10.0: # 3D死区放大到10度
                details['smoothness'] += ((angle - 10.0) ** 2) * 0.2
            if angle > max_turn:
                details['sharp_turn'] += sharp_turn_pen

        # 4. 2D投影打卡检测
        details['missed_target'] += self.calculate_target_penalty(path_points)

        env_info = {
            'ideal_distance': self.ideal_min_distance,
            'obstacle_count': len(self.env.obstacles)
        }
                
        total_score = sum(details.values())
        return total_score, details, env_info

    def evaluate_pso_particle(self, raw_waypoints):
        spacing_penalty = self.calculate_spacing_penalty(raw_waypoints)
        raw_score, raw_details, env_info = self.calculate_fitness(raw_waypoints)
        
        # 提前拦截
        if raw_details['fatal_collision'] > 0 or raw_details['missed_target'] > 0 or raw_details['altitude_violation'] > 0:
            raw_details['spacing_penalty'] = spacing_penalty
            total_score = sum(raw_details.values())
            return total_score, raw_details, env_info
            
        iters = self.params.get('chaikin_iterations', 3)
        smooth_path = self.generate_bspline_path(raw_waypoints, iterations=iters)
        
        base_score, smooth_details, _ = self.calculate_fitness(smooth_path)
        smooth_details['spacing_penalty'] = spacing_penalty
        total_score = sum(smooth_details.values())
        
        return total_score, smooth_details, env_info

# ==========================================
# 3D 测试用例 
# ==========================================
if __name__ == "__main__":
    evaluator = PathEvaluator()
    
    # 【全面更新为3D测试路径】
    # 高空安全飞跃
    path_3d_safe = np.array([[43.0, 3.0, 30.0], [43.0, 50.0, 30.0], [51.0, 94.0, 30.0]])
    # 低空穿模撞大楼
    path_3d_crash = np.array([[43.0, 3.0, 5.0], [43.0, 50.0, 5.0], [51.0, 94.0, 5.0]])
    # 钻入地下
    path_3d_underground = np.array([[43.0, 3.0, 30.0], [43.0, 50.0, -5.0], [51.0, 94.0, 30.0]])

    test_cases = [
        ("高空安全直达 (完美路线)", path_3d_safe),
        ("低空莽夫直飞 (撞大楼)", path_3d_crash),
        ("遁地路线 (钻入地下)", path_3d_underground)
    ]

    print("=" * 50)
    print("开始执行 3D 评价器黑盒压力测试...")
    print("=" * 50)

    for name, raw_path in test_cases:
        score, details, env_info = evaluator.evaluate_pso_particle(raw_path)
        
        print(f"【{name}】")
        if score > 5000:
            print(f"   最终得分: \033[91m{score:,.2f}\033[0m")
        else:
            print(f"   最终得分: \033[92m{score:,.2f}\033[0m (安全无误！)")
            
        print("   >>> 状态明细(State):")
        for k, v in details.items():
            if v > 0: 
                color = "\033[91m" if v > 1000 else "\033[0m" 
                print(f"       - {k}: {color}{v:,.2f}\033[0m")
        print("-" * 50)

    # ==========================================
    # 【新增：3D 可视化绘图代码】
    # ==========================================
    # 我们拿第一条安全的完美路线来画图展示
    path_to_draw = path_3d_safe 
    
    # 【修复】：把这里传入的参数改为 num_points=100 以适配 B-Spline
    smooth_path_to_draw = evaluator.generate_bspline_path(path_to_draw, num_points=100)
    
    fig = plt.figure(figsize=(12, 10))
    # 必须声明 projection='3d'，告诉 matplotlib 这是一个 3D 画布
    ax = fig.add_subplot(111, projection='3d') 
    
    # 1. 把 3D 建筑物和目标圈画出来
    evaluator.env.draw_environment_3d(ax)
    
    # 2. 画原始稀疏航点 (虚线带叉叉)
    # 注意这里传入了三个切片: [:, 0]是X, [:, 1]是Y, [:, 2]是Z
    ax.plot(path_to_draw[:, 0], path_to_draw[:, 1], path_to_draw[:, 2], 
            color='gray', linestyle='--', linewidth=2, marker='x', markersize=8, label='Original Waypoints')
            
    # 3. 画经过 B-Spline 平滑后的 3D 飞行曲线 (粉红色实线)
    ax.plot(smooth_path_to_draw[:, 0], smooth_path_to_draw[:, 1], smooth_path_to_draw[:, 2], 
            color='#FF007F', linestyle='-', linewidth=3, label='B-Spline Smooth Path')
            
    ax.set_title("3D UAV Path Visualization (B-Spline)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()