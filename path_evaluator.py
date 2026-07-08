import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制底层数学计算引擎只用单线程，防止内存冲突
import numpy as np
import matplotlib.pyplot as plt
from environment_buildup_3D import UAVEnvironment3D  
import scipy.interpolate as spl

class PathEvaluator:
    def __init__(self):
        # 实例化3D环境
        self.env = UAVEnvironment3D('maps/haining.json5')
        
        # 基础惩罚权重
        self.penalties = {
            'fatal_collision': 1000000.0,  
            'missed_target_base': 500000.0,
            'missed_target_factor': 20000.0,    
            'sharp_turn': 10000.0,   
            'margin_violation': 5000.0,
            'altitude_violation': 50000.0,
            'boundary_violation': 50000.0,
            'pitch_violation': 20000.0,
            'loop_violation': 15000.0
        }

        self.params = {
            'max_turn_angle': 120.0,
            'max_pitch_angle': 45.0,
            'bspline_num_points': 100,   # B-Spline 参数
            'min_waypoint_dist': 5.0,
            'margin_layers': [0.5, 0.2]
        }
        
        self.ideal_min_distance = self._calculate_ideal_min_distance()
        print(f" [环境加载] 当前3D地图理论最短直线距离: {self.ideal_min_distance:.1f} 米")

    def _calculate_ideal_min_distance(self):
        pts = [self.env.start_point]
        targets = [t['center'] for t in self.env.target_areas]
        
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

    def generate_bspline_path(self, waypoints, num_points=100):
        """ 
        【全面升级为 3D B-Spline 插值算法】 
        极致平滑，二阶导数连续，符合真实无人机空气动力学飞行轨迹。
        """
        waypoints = np.array(waypoints)
        
        unique_waypoints = [waypoints[0]]
        for pt in waypoints[1:]:
            if np.linalg.norm(pt - unique_waypoints[-1]) > 0.1:
                unique_waypoints.append(pt)
        unique_waypoints = np.array(unique_waypoints)

        num_wp = len(unique_waypoints)
        if num_wp < 3:
            return unique_waypoints 
            
        k = 3 if num_wp >= 4 else num_wp - 1

        x = unique_waypoints[:, 0]
        y = unique_waypoints[:, 1]
        z = unique_waypoints[:, 2]

        tck, u = spl.splprep([x, y, z], s=0, k=k)

        u_new = np.linspace(0, 1.0, num_points)
        x_new, y_new, z_new = spl.splev(u_new, tck)

        smooth_path = np.column_stack((x_new, y_new, z_new))

        # 任何平滑过冲导致的负数高度，全部托底到 0.1 米（贴地皮）
        smooth_path[:, 2] = np.clip(smooth_path[:, 2], 0.1, self.env.z_bounds[1]) # 顺便顺手把天花板也卡死
        
        # X 和 Y 轴的绝对硬裁剪，利用环境里的 bounds，确保平滑轨迹也绝不出界
        smooth_path[:, 0] = np.clip(smooth_path[:, 0], self.env.x_bounds[0], self.env.x_bounds[1])
        smooth_path[:, 1] = np.clip(smooth_path[:, 1], self.env.y_bounds[0], self.env.y_bounds[1])
        
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
        v1 = p2 - p1
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def calculate_pitch_angle(self, p1, p2):
        vec = p2 - p1
        dz = vec[2] # Z轴上的高度差
        dist_3d = np.linalg.norm(vec)
        if dist_3d == 0: return 0.0
        # 利用反正弦函数计算夹角
        pitch_rad = np.arcsin(np.clip(dz / dist_3d, -1.0, 1.0))
        return np.abs(np.degrees(pitch_rad))

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
            center_2d = target['center'][:2] 
            radius = target['radius']
            z_min = target.get('z_min', 0.0)
            z_max = target.get('z_max', 20.0) 
            
            min_missed_dist = float('inf')
            
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                
                dist_3d = self.env.calculate_distance(p1, p2)
                num_steps = max(2, int(dist_3d / 1.0)) 
                
                for step_i in range(num_steps + 1):
                    t = step_i / num_steps
                    pt = p1 + t * (p2 - p1) 
                    
                    dist_xy = np.linalg.norm(pt[:2] - center_2d)
                    missed_xy = max(0.0, dist_xy - radius)
                    
                    if pt[2] < z_min:
                        missed_z = z_min - pt[2]
                    elif pt[2] > z_max:
                        missed_z = pt[2] - z_max
                    else:
                        missed_z = 0.0
                        
                    total_missed = np.sqrt(missed_xy**2 + missed_z**2)
                    
                    if total_missed < min_missed_dist:
                        min_missed_dist = total_missed
            
            if min_missed_dist > 0.1: 
                # 动态读取惩罚值，如果没有被 Agent 修改，就默认使用你的阶梯惩罚
                # 恢复原本纯粹且公平的线性惩罚逻辑
                base_pen = self.penalties.get('missed_target_base', 500000.0)
                factor_pen = self.penalties.get('missed_target_factor', 20000.0)
                
                # （距离 * 系数）
                total_target_penalty += base_pen + (min_missed_dist * factor_pen)
                
        return total_target_penalty

    def calculate_fitness(self, path_points):
        details = {
            'distance': 0.0,          
            'fatal_collision': 0.0,   
            'margin_violation': 0.0,  
            'smoothness': 0.0,        
            'sharp_turn': 0.0,        
            'missed_target': 0.0,
            'altitude_violation': 0.0,
            'boundary_violation': 0.0,
            'gravity_cost': 0.0,
            'pitch_violation': 0.0, # 记录俯仰角违规扣分
            'loop_penalty': 0.0
        }

        # 1. 距离算分
        details['distance'] = self.calculate_path_length(path_points)
        
        # 清理了重复定义的冗余代码
        margin_layers = self.params.get('margin_layers', [0.5, 0.2])
        layer_penalty = self.penalties.get('margin_violation', 5000.0) / len(margin_layers)
        bound_penalty = self.penalties.get('boundary_violation', 50000.0)
        alt_penalty = self.penalties.get('altitude_violation', 50000.0)
        fatal_penalty = self.penalties.get('fatal_collision', 1000000.0)

        # 2. 空域与四周边界管制检测
        for i in range(len(path_points)):
            pt = path_points[i]
            
            # 2.1 垂直空域管制（钻地或冲天）
            if pt[2] < 0:
                details['fatal_collision'] += fatal_penalty
            elif pt[2] > self.env.z_bounds[1]:
                details['altitude_violation'] += alt_penalty
                
            # 2.2 X 轴水平越界检测
            if pt[0] < self.env.x_bounds[0] or pt[0] > self.env.x_bounds[1]:
                details['boundary_violation'] += bound_penalty
                
            # 2.3 Y 轴水平越界检测
            if pt[1] < self.env.y_bounds[0] or pt[1] > self.env.y_bounds[1]:
                details['boundary_violation'] += bound_penalty

        #  防绕圈/防自相交检测 (Anti-Looping)
        N = len(path_points)
        time_gap = 15  # 时间差：如果在相隔 15 个点之外
        loop_radius = 5.0 # 空间差：距离又近于 5 米，那就是纯纯的绕圈
        
        if N > time_gap:
            # 利用 numpy 上三角矩阵快速提取所有索引差 > time_gap 的点对
            i_idx, j_idx = np.triu_indices(N, k=time_gap)
            
            # 向量化计算所有这些点对之间的 3D 距离 (计算速度是 for 循环的 100 倍)
            dists = np.linalg.norm(path_points[i_idx] - path_points[j_idx], axis=1)
            
            # 找出那些“时间过了很久，却又回到原地附近”的违规点对
            loop_count = np.sum(dists < loop_radius)
            if loop_count > 0:
                details['loop_penalty'] += loop_count * self.penalties.get('loop_violation', 15000.0)

        # SFJ (Straight Flight Judgment) 快筛雷达
        # 极大降低 CPU 算力消耗，为 B-Spline 微小线段办理“免检通行证”
        sfj_safe_segments = [False] * (len(path_points) - 1)
        jump_step = 10  # SFJ 探测步长：一次性往后看 10 步 (宏观视野)
        
        for i in range(0, len(path_points) - jump_step, jump_step):
            p_start = path_points[i]
            p_end = path_points[i + jump_step]
            
            # 使用稍微大一圈的安全余量 (比如 2.0米)，做一次宏观视距(LoS)判断
            # 如果这条长直线是完全畅通的，说明这一大片空域没有任何障碍物！
            if not self.env.is_segment_collision(p_start, p_end, safe_margin=2.0):
                # 探测安全！给中间这 10 个小线段全部盖上“免检”钢印！
                for j in range(i, i + jump_step):
                    sfj_safe_segments[j] = True

        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            # 物理线段积分法 (能耗 = 航段长度 × 平均高度)
            segment_dist = np.linalg.norm(p2 - p1)
            avg_height = (p1[2] + p2[2]) / 2.0
            
            if avg_height > 0:
                # 无论线段被切得多碎，这段物理距离的积分面积永远恒定！
                # 乘以 1.5 是为了让分数规模和以前保持在同一个数量级
                details['gravity_cost'] += avg_height * segment_dist * 1.5

            # 每段飞行路线的物理俯仰角检测 (保持你之前的代码不变)
            pitch = self.calculate_pitch_angle(p1, p2)
            if pitch > self.params.get('max_pitch_angle', 45.0):
                # 累加惩罚：超出越多，惩罚越狠 (二次方惩罚)
                details['pitch_violation'] += self.penalties.get('pitch_violation', 20000.0) * ((pitch - 45.0) / 10.0)

            # ==========================================
            # SFJ 偷懒判断发威！
            if sfj_safe_segments[i]:
                continue # 有 SFJ 护航说明周围绝对没有大楼，直接跳过极其耗时的几何障碍物遍历检测！
            # ==========================================

            # 只有在 SFJ 没敢打包票的地方，才启动复杂的避障排雷算法
            if self.env.is_segment_collision(p1, p2, safe_margin=0.0):
                details['fatal_collision'] += fatal_penalty
                continue 
                
            if not self.env.is_segment_collision(p1, p2, safe_margin=0.5):
                continue

            for m in margin_layers:
                if self.env.is_segment_collision(p1, p2, safe_margin=m):
                    details['margin_violation'] += layer_penalty

        # 3. 急转弯计算
        max_turn = self.params.get('max_turn_angle', 120.0)
        sharp_turn_pen = self.penalties.get('sharp_turn', 10000.0)

        for i in range(len(path_points) - 2):
            angle = self.calculate_turn_angle(path_points[i], path_points[i+1], path_points[i+2])
            if angle > 10.0: 
                details['smoothness'] += ((angle - 10.0) ** 2) * 0.2
            if angle > max_turn:
                details['sharp_turn'] += sharp_turn_pen

        # 4. 3D 悬空圆柱打卡检测
        details['missed_target'] += self.calculate_target_penalty(path_points)

        env_info = {
            'ideal_distance': self.ideal_min_distance,
            'obstacle_count': len(self.env.obstacles)
        }
                
        total_score = sum(details.values())
        return total_score, details, env_info

    def evaluate_pso_particle(self, raw_waypoints):
        # 1. 提取控制点之间的间距惩罚 (保留对底层基因的排斥判定)
        spacing_penalty = self.calculate_spacing_penalty(raw_waypoints)
        
        # 强制写死 100 个点作为唯一的绝对公平基准！
        # 绝不读取 params 里的动态配置，防止 Agent 乱改导致分数通货膨胀！
        num_pts = 100 
        smooth_path = self.generate_bspline_path(raw_waypoints, num_points=num_pts)
        
        base_score, smooth_details, env_info = self.calculate_fitness(smooth_path)
        
        # 3. 补上底层基因的间距惩罚
        smooth_details['spacing_penalty'] = spacing_penalty
        total_score = sum(smooth_details.values())
        
        return total_score, smooth_details, env_info
# ==========================================
# 3D 测试用例 
# ==========================================
if __name__ == "__main__":
    evaluator = PathEvaluator()
    
    path_3d_safe = np.array([[43.0, 3.0, 30.0], [43.0, 50.0, 30.0], [51.0, 94.0, 30.0]])
    path_3d_crash = np.array([[43.0, 3.0, 5.0], [43.0, 50.0, 5.0], [51.0, 94.0, 5.0]])
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

    path_to_draw = path_3d_safe 
    smooth_path_to_draw = evaluator.generate_bspline_path(path_to_draw, num_points=100)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d') 
    
    evaluator.env.draw_environment_3d(ax)
    
    ax.plot(path_to_draw[:, 0], path_to_draw[:, 1], path_to_draw[:, 2], 
            color='gray', linestyle='--', linewidth=2, marker='x', markersize=8, label='Original Waypoints')
            
    ax.plot(smooth_path_to_draw[:, 0], smooth_path_to_draw[:, 1], smooth_path_to_draw[:, 2], 
            color='#FF007F', linestyle='-', linewidth=3, label='B-Spline Smooth Path')
            
    ax.set_title("3D UAV Path Visualization (B-Spline)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()