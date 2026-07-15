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
            'loop_violation': 15000.0,
            # 机动变化功率 (Change Power) 的超标惩罚乘数
            'change_power_penalty': 8000.0 
        }

        self.params = {
            'max_turn_angle': 120.0,
            'max_pitch_angle': 45.0,
            'bspline_num_points': 100,   
            'min_waypoint_dist': 5.0,
            'margin_layers': [0.5, 0.2],
            # 能耗物理学参数
            'v_cruise': 13.17,         # 假设最佳巡航速度为 13.17m/s
            'accel_threshold': 2.0    # 允许的最大合理加速度 2.0m/s^2 ，超过即视为急刹车/急加速
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

        smooth_path[:, 2] = np.clip(smooth_path[:, 2], 0.1, self.env.z_bounds[1]) 
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
        dz = vec[2] 
        dist_3d = np.linalg.norm(vec)
        if dist_3d == 0: return 0.0
        pitch_rad = np.arcsin(np.clip(dz / dist_3d, -1.0, 1.0))
        return np.abs(np.degrees(pitch_rad))

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
                base_pen = self.penalties.get('missed_target_base', 500000.0)
                factor_pen = self.penalties.get('missed_target_factor', 20000.0)
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
            'pitch_violation': 0.0,
            'loop_penalty': 0.0,
            'base_energy_cost': 0.0,   # 出生自带的固定能耗
            'change_power_pen': 0.0    # 机动变化功率(急停急加)超标罚分
        }

        # 先算总距离
        total_dist = self.calculate_path_length(path_points)
        details['distance'] = total_dist
        
        # ==========================================
        # 大招四：PECM 广义推进能耗几何映射模型
        # ==========================================
        v = self.params.get('v_cruise', 13.17)
        
        # 1. 常规基础能耗 (起降悬停 + 寄生阻力 + 诱导阻力)
        # 根据公式：寄生正比于 v^3, 诱导反比于 v。此处用常数系数代替空气动力学实参。
        c_parasite = 0.1
        c_induced = 150.0
        P_cruise = c_parasite * (v**3) + c_induced * (1/v)
        
        time_flight = total_dist / v
        E_base = 2000.0  # 起降通讯等固定开销
        E_cruise = P_cruise * time_flight
        details['base_energy_cost'] = E_base + E_cruise
        
        # 2. 机动变化功率 (Change Power) 罚分 
        accel_threshold = self.params.get('accel_threshold', 2.5)
        change_power_multiplier = self.penalties.get('change_power_penalty', 8000.0)
        
        # ==========================================

        margin_layers = self.params.get('margin_layers', [0.5, 0.2])
        layer_penalty = self.penalties.get('margin_violation', 5000.0) / len(margin_layers)
        bound_penalty = self.penalties.get('boundary_violation', 50000.0)
        alt_penalty = self.penalties.get('altitude_violation', 50000.0)
        fatal_penalty = self.penalties.get('fatal_collision', 1000000.0)

        # 1. 空域与四周边界管制检测
        for i in range(len(path_points)):
            pt = path_points[i]
            if pt[2] < 0:
                details['fatal_collision'] += fatal_penalty
            elif pt[2] > self.env.z_bounds[1]:
                details['altitude_violation'] += alt_penalty
            if pt[0] < self.env.x_bounds[0] or pt[0] > self.env.x_bounds[1]:
                details['boundary_violation'] += bound_penalty
            if pt[1] < self.env.y_bounds[0] or pt[1] > self.env.y_bounds[1]:
                details['boundary_violation'] += bound_penalty

        # 绕圈/防自相交检测 (Anti-Looping)
        N = len(path_points)
        time_gap = 15  
        loop_radius = 5.0 
        
        if N > time_gap:
            i_idx, j_idx = np.triu_indices(N, k=time_gap)
            dists = np.linalg.norm(path_points[i_idx] - path_points[j_idx], axis=1)
            loop_count = np.sum(dists < loop_radius)
            if loop_count > 0:
                details['loop_penalty'] += loop_count * self.penalties.get('loop_violation', 15000.0)

        # SFJ 快筛雷达
        sfj_safe_segments = [False] * (len(path_points) - 1)
        jump_step = 10  
        
        for i in range(0, len(path_points) - jump_step, jump_step):
            p_start = path_points[i]
            p_end = path_points[i + jump_step]
            if not self.env.is_segment_collision(p_start, p_end, safe_margin=2.0):
                for j in range(i, i + jump_step):
                    sfj_safe_segments[j] = True

        # 逐段精细计算
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            segment_dist = np.linalg.norm(p2 - p1)
            avg_height = (p1[2] + p2[2]) / 2.0
            if avg_height > 0:
                details['gravity_cost'] += avg_height * segment_dist * 1.5

            pitch = self.calculate_pitch_angle(p1, p2)
            if pitch > self.params.get('max_pitch_angle', 45.0):
                details['pitch_violation'] += self.penalties.get('pitch_violation', 20000.0) * ((pitch - 45.0) / 10.0)

            if sfj_safe_segments[i]:
                continue 

            if self.env.is_segment_collision(p1, p2, safe_margin=0.0):
                details['fatal_collision'] += fatal_penalty
                continue 
                
            if not self.env.is_segment_collision(p1, p2, safe_margin=0.5):
                continue

            for m in margin_layers:
                if self.env.is_segment_collision(p1, p2, safe_margin=m):
                    details['margin_violation'] += layer_penalty

        # 急转弯计算 & 机动变化功率(Change Power)计算
        max_turn = self.params.get('max_turn_angle', 120.0)
        sharp_turn_pen = self.penalties.get('sharp_turn', 10000.0)

        for i in range(len(path_points) - 2):
            p_prev = path_points[i]
            p_curr = path_points[i+1]
            p_next = path_points[i+2]
            
            angle = self.calculate_turn_angle(p_prev, p_curr, p_next)
            if angle > 10.0: 
                details['smoothness'] += ((angle - 10.0) ** 2) * 0.2
            if angle > max_turn:
                details['sharp_turn'] += sharp_turn_pen
                
            # PECM: 几何向运动学的神级映射
            # 1. 用转角算出速度矢量的改变 (delta_v ≈ v * 弧度)
            delta_v = v * np.radians(angle)
            
            # 2. 算这段时间 (dt ≈ 段距离 / v)
            dist_seg = np.linalg.norm(p_curr - p_prev)
            dt = dist_seg / v if dist_seg > 0 else 0.1
            
            # 3. 算出该航段所逼出的向心加速度 / 减速加速度
            accel = delta_v / dt
            
            # 4. 如果加速度在安全阈值内，视为正常平滑机动，耗电极少不予扣分；
            #    如果超出阈值，暴增的电机电流 (P_change) 与加速度的平方成正比！
            if accel > accel_threshold:
                excess_accel = accel - accel_threshold
                details['change_power_pen'] += (excess_accel ** 2) * change_power_multiplier

        # 3D 悬空圆柱打卡检测
        details['missed_target'] += self.calculate_target_penalty(path_points)

        env_info = {
            'ideal_distance': self.ideal_min_distance,
            'obstacle_count': len(self.env.obstacles)
        }
                
        total_score = sum(details.values())
        return total_score, details, env_info

    def evaluate_pso_particle(self, raw_waypoints):
        spacing_penalty = self.calculate_spacing_penalty(raw_waypoints)
        
        num_pts = 100 
        smooth_path = self.generate_bspline_path(raw_waypoints, num_points=num_pts)
        
        base_score, smooth_details, env_info = self.calculate_fitness(smooth_path)
        smooth_details['spacing_penalty'] = spacing_penalty
        total_score = sum(smooth_details.values())
        
        return total_score, smooth_details, env_info