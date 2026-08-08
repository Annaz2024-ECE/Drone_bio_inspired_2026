import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制底层数学计算引擎只用单线程，防止内存冲突
import numpy as np
import matplotlib.pyplot as plt
from environment_buildup_3D import UAVEnvironment3D, RandomMapGenerator  
import scipy.interpolate as spl

class PathEvaluator:
    def __init__(self, randomize_targets=True):
        # 实例化3D环境
        self.env = UAVEnvironment3D('maps/hard_map.json5') # 假设你现在跑紫金港
        
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
            # 机动变化功率的惩罚系数 (已大幅下调，配合新版逻辑)
            'change_power_penalty': 10.0,
            # 时间惩罚权重 (每飞行1秒扣 10 分)
            'time_penalty_factor': 10.0
        }

        self.params = {
            'max_turn_angle': 120.0,
            'max_pitch_angle': 45.0,
            'bspline_num_points': 100,   
            'min_waypoint_dist': 5.0,
            'margin_layers': [0.5, 0.2],
            # 能耗物理学参数
            'v_cruise': 13.17,        # 巡航最佳速度 m/s
            'v_inspection': 5.0,      # 巡检打卡时的低速模式
            'accel_threshold': 2.0    # 最大合理加速度 2.0m/s^2
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

    # ==========================================
    # 用于接收智能体的目标裁剪指令并刷新全局参数
    # ==========================================
    def update_env_targets(self, new_targets):
        """ 
        动态目标裁剪同步接口：更新环境目标，并重新计算理论最短距离 
        """
        self.env.target_areas = new_targets
        # 重新计算剔除目标后的最短距离
        self.ideal_min_distance = self._calculate_ideal_min_distance()
        print(f" [环境同步] 气象裁剪完成！当前3D地图理论最短直线距离已刷新为: \033[92m{self.ideal_min_distance:.1f} 米\033[0m")

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

    # ==========================================
    # 感知环境，自动进行限速降挡
    # ==========================================
    def _get_local_speed(self, pt):
        v_cruise = self.params.get('v_cruise', 13.17)
        v_inspection = self.params.get('v_inspection', 5.0)
        radius_multiplier = 3.0  # 减速影响半径倍数
        
        # 1. 计算当前点到所有目标点的距离 (XY平面)
        min_dist_to_target = float('inf')
        for target in self.env.target_areas:
            center_2d = target['center'][:2]
            radius = target['radius']
            # 只考虑 XY 平面距离
            dist_xy = np.linalg.norm(pt[:2] - center_2d)
            # 计算“到目标边缘的相对距离”，越小表示越靠近目标
            relative_dist = dist_xy / radius
            if relative_dist < min_dist_to_target:
                min_dist_to_target = relative_dist

        # 2. 如果没有目标，直接返回巡航速度
        if min_dist_to_target == float('inf'):
            return v_cruise

        # 3. 核心平滑公式：根据距离比例，线性插值速度
        # 当 relative_dist <= 1.0 (在目标内部) → 速度 = 5.0
        # 当 relative_dist >= 3.0 (远离目标) → 速度 = 13.17
        # 在 1.0 ~ 3.0 之间 → 速度线性变化
        if min_dist_to_target <= 1.0:
            return v_inspection
        elif min_dist_to_target >= radius_multiplier:
            return v_cruise
        else:
            # 线性插值：越靠近目标，速度越接近 5.0
            t = (min_dist_to_target - 1.0) / (radius_multiplier - 1.0)  # t: 0~1
            speed = v_inspection + (v_cruise - v_inspection) * t
            return speed


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
            'base_energy_cost': 0.0,   
            'change_power_pen': 0.0,
            'time_cost': 0.0   
        }

        # 先算总距离
        total_dist = self.calculate_path_length(path_points)
        details['distance'] = total_dist
        
        v_cruise = self.params.get('v_cruise', 13.17)
        
        # 1. 稳态基础能耗 + 任务通信能耗
        c_parasite = 0.1
        c_induced = 150.0
        P_cruise = c_parasite * (v_cruise**3) + c_induced * (1/v_cruise)
        time_flight_est = total_dist / v_cruise

        time_factor = self.penalties.get('time_penalty_factor', 10.0)
        details['time_cost'] = time_flight_est * time_factor 
        
        E_base = 2000.0 
        E_task = len(self.env.target_areas) * 500.0 # 每个打卡点的并发通信耗电
        E_cruise = P_cruise * time_flight_est
        
        details['base_energy_cost'] = E_base + E_task + E_cruise
        
        accel_threshold = self.params.get('accel_threshold', 2.0)
        change_power_multiplier = self.penalties.get('change_power_penalty', 10.0)
        
        margin_layers = self.params.get('margin_layers', [0.5, 0.2])
        layer_penalty = self.penalties.get('margin_violation', 5000.0) / len(margin_layers)
        bound_penalty = self.penalties.get('boundary_violation', 50000.0)
        alt_penalty = self.penalties.get('altitude_violation', 50000.0)
        fatal_penalty = self.penalties.get('fatal_collision', 1000000.0)

        # 边界与底线安全
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

        # 防绕圈
        N = len(path_points)
        time_gap = 15  
        loop_radius = 5.0 
        if N > time_gap:
            i_idx, j_idx = np.triu_indices(N, k=time_gap)
            dists = np.linalg.norm(path_points[i_idx] - path_points[j_idx], axis=1)
            loop_count = np.sum(dists < loop_radius)
            if loop_count > 0:
                details['loop_penalty'] += loop_count * self.penalties.get('loop_violation', 15000.0)

        # 雷达快筛
        sfj_safe_segments = [False] * (len(path_points) - 1)
        jump_step = 10  
        for i in range(0, len(path_points) - jump_step, jump_step):
            p_start = path_points[i]
            p_end = path_points[i + jump_step]
            if not self.env.is_segment_collision(p_start, p_end, safe_margin=2.0):
                for j in range(i, i + jump_step):
                    sfj_safe_segments[j] = True

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

            if sfj_safe_segments[i]: continue 

            # 致命碰撞检测 (safe_margin=0.0)
            if self.env.is_segment_collision(p1, p2, safe_margin=0.0):
                details['fatal_collision'] += fatal_penalty
                continue 
            # 安全裕度分层检测 (margin_layers = [0.5, 0.2])
            if not self.env.is_segment_collision(p1, p2, safe_margin=0.5): continue
            for m in margin_layers:
                if self.env.is_segment_collision(p1, p2, safe_margin=m):
                    details['margin_violation'] += layer_penalty

        # ==========================================
        # 急转弯 & 动态限速能耗计算 (双轴加速度)
        # ==========================================
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
                
            # ----------------------------------------------------
            # 获取两点的速度差，计算双轴加速度
            # ----------------------------------------------------
            # 1. 分别获取上一段和当前段的速度限制
            v_prev_local = self._get_local_speed(p_prev)
            v_curr_local = self._get_local_speed(p_curr)
            
            dist_seg = np.linalg.norm(p_curr - p_prev)*10 # 比例尺
            dist_seg = max(dist_seg, 0.5)  # 防爆锁，防止除以 0
            
            # 2. 使用平均速度计算通过这段航段的真实时间 dt
            v_avg = (v_prev_local + v_curr_local) / 2.0
            dt = dist_seg / v_avg
            
            # 3. 计算双轴加速度
            # a. 向心加速度 (转弯带来的横向过载)
            delta_v_turn = v_curr_local * np.radians(angle)
            accel_turn = delta_v_turn / dt
            
            # b. 线性/纵向加速度 (切换速度状态时的急刹车/急加速)
            accel_linear = abs(v_curr_local - v_prev_local) / dt
            
            # c. 总机动加速度 (勾股定理合成物理矢量)
            accel = np.sqrt(accel_turn**2 + accel_linear**2)
            # ----------------------------------------------------
            
            # 新版双层惩罚机制 (无缝对接原版代码，变量名沿用 accel)
            if accel > accel_threshold:
                # 严重超标：平方级暴击惩罚 (急加急刹)
                excess_accel = accel - accel_threshold
                details['change_power_pen'] += (excess_accel ** 2) * change_power_multiplier
            elif accel > 0.1:
                # 没超标但有加速度：极小的线性惩罚
                details['change_power_pen'] += accel * (change_power_multiplier * 0.05)

        details['missed_target'] += self.calculate_target_penalty(path_points)

        env_info = {
            'ideal_distance': self.ideal_min_distance,
            'obstacle_count': len(self.env.obstacles)
        }
                 
        total_score = sum(details.values())
        return total_score, details, env_info

    def evaluate_particle(self, raw_waypoints):
        spacing_penalty = self.calculate_spacing_penalty(raw_waypoints)
        num_pts = 100 
        smooth_path = self.generate_bspline_path(raw_waypoints, num_points=num_pts)
        
        base_score, smooth_details, env_info = self.calculate_fitness(smooth_path)
        smooth_details['spacing_penalty'] = spacing_penalty
        total_score = sum(smooth_details.values())
        
        return total_score, smooth_details, env_info