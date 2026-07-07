class CoordinatorAgent:
    def __init__(self):
        # 1. 算法共性参数管理 (算力预算)
        self.algo_params = {'pop_size': 50, 'max_iter': 100}
        
        # 2. 评价器物理参数管理 (物理规则)
        self.eval_params = {
            'bspline_num_points': 100, 
            'min_waypoint_dist': 5.0,
            'max_turn_angle': 120.0  # 🔥 新增：把转弯限制纳入老中医的调控范围
        }

        # 3. 核心监控指标
        self.meta_iteration = 0
        self.stuck_counter = 0
        self.last_score = float('inf')


    def analyze_and_act(self, total_score, details, env_info, current_algo):
        """
        六大算法全解锁的【终极 3D 调参大脑】(物理启发式升级版)
        """
        self.meta_iteration += 1
        actions_taken = []
        specific_params = {} 
        is_finished = False 
        
        # 全局物理行为指令 (Universal Directives)
        # 任何算法只要探测到这些指令为 True，就可以执行对应的物理动作
        specific_params['emergency_escape'] = False 
        specific_params['radar_guidance'] = False
        specific_params['press_down'] = False       # 向下试探 (省电)
        specific_params['lift_up'] = False          # 向上拉升 (避障)
        
        print(f"\n[调参] 第 {self.meta_iteration} 轮诊断中... (负责压榨 {current_algo} 的极限)")

        # ==========================================
        # 0. 收敛极限与早停判定 
        # ==========================================
        improvement = self.last_score - total_score
        if self.last_score == float('inf'):
            improvement_rate = 0.05 #可搭建桥梁让llm/仿生优化智能体调节  
        else:
            improvement_rate = max(0, improvement) / (self.last_score + 1e-8) 
            
        self.last_score = total_score 

        is_perfectly_safe = (details.get('fatal_collision', 0) == 0 and 
                             details.get('missed_target', 0) == 0 and
                             details.get('sharp_turn', 0) == 0 and
                             details.get('altitude_violation', 0) == 0)
        
        if is_perfectly_safe and (self.meta_iteration > 1) and (improvement_rate < 0.005):
            print(f"   [全局通知] 3D 航线已绝对安全，且收敛至极限(进步率 < 0.5%)，申请提前结束")
            is_finished = True
            return self.algo_params, self.eval_params, specific_params, is_finished

        ideal_dist = env_info.get('ideal_distance', 100.0)
        obs_count = env_info.get('obstacle_count', 0)
        dynamic_tolerance = 1.0 + (obs_count * 0.005)
        max_allowed_dist = ideal_dist * dynamic_tolerance
        
        if is_perfectly_safe and details.get('distance', 0) > max_allowed_dist:
            print(f"  [警告] 路线安全，但总航程 {details.get('distance'):.1f}m 超过动态底线，存在绕路")
            if current_algo == "PSO":
                specific_params['c2'] = 1.0 
            elif current_algo in ["ACO", "DSACO"]:
                specific_params['beta'] = 6.0 
            elif current_algo == "SSA":
                specific_params['ST'] = 0.9 
            actions_taken.append(f"TUNE_{current_algo}: 降低探索欲，强行拉直航线以缩短距离")

        # 判定卡壳
        is_failing = (details.get('fatal_collision', 0) > 0 or 
                      details.get('missed_target', 0) > 0 or
                      details.get('altitude_violation', 0) > 0 or
                      details.get('boundary_violation', 0) > 0)
                      
        if is_failing and improvement < 1000:
            self.stuck_counter += 1
            print(f"  [警告] 算法在 3D 空间陷入瓶颈！累计卡壳: {self.stuck_counter} 次")
        else:
            self.stuck_counter = 0

        # ==========================================
        #  1. 通用物理维度全局引导 (Universal Physical Guidance)
        # ==========================================
        
        # 1.1 撞墙避障逻辑：尝试拉升高度 + 放宽转弯限制
        if details.get('fatal_collision', 0) > 0:
            specific_params['lift_up'] = True
            self.eval_params['max_turn_angle'] = 150.0 # 放宽物理规则：允许150度急转弯来躲避大楼
            actions_taken.append("UNIVERSAL: 遭遇建筑碰撞！下达全局 [紧急拉升] 指令，并放宽最大转弯角至 150° 以利于规避")
        else:
            self.eval_params['max_turn_angle'] = 120.0 # 没撞墙就恢复正常平滑转弯

        # 1.2 高度省电逻辑：如果绝对安全，但飞得太高，引导下压
        is_safe_but_high = is_perfectly_safe and (details.get('gravity_cost', 0) > 1500)
        if is_safe_but_high:
            specific_params['press_down'] = True
            actions_taken.append("UNIVERSAL: 航线绝对安全但能耗过高，下达全局 [贴地压低] 指令，试探安全底线")

        # 1.3 漏打卡逻辑：如果漏打卡，直接全局广播雷达空投引导
        if details.get('missed_target', 0) > 0:
            specific_params['radar_guidance'] = True 
            actions_taken.append("UNIVERSAL: 偏离打卡点！激活全系统 [雷达空投] 机制，引导敢死队向目标跃迁")

        # 2. 算法专属参数微操 (仅调整算法自身的数学参数)
        if current_algo in ["ACO", "DSACO"]:
            if self.stuck_counter >= 2:
                specific_params['rho'] = 0.5 
                actions_taken.append(f"TUNE_{current_algo}: 提高挥发率 rho=0.5, 迫使其遗忘烂路重搜")

        elif current_algo == "PSO":
            if details.get('smoothness', 0) > 2000:
                specific_params['c1'] = 2.2  
                specific_params['c2'] = 1.0
                actions_taken.append("TUNE_PSO: 调高 c1 调低 c2, 使其注重个体轨迹自适应平滑")

        elif current_algo == "SSA":
            if self.stuck_counter >= 1:
                specific_params['ST'] = 0.6  
                actions_taken.append("TUNE_SSA: 降低安全阈值 ST=0.6, 强制打散局部僵局")

        elif current_algo == "GWO":
            if self.stuck_counter >= 1:
                specific_params['stagnation_max'] = 12  
                actions_taken.append("TUNE_GWO: 降低停滞阈值至 12 代，加速触发大爆炸")

        elif current_algo == "WOA":
            if details.get('smoothness', 0) > 3000:
                specific_params['b'] = 0.4  
                actions_taken.append("TUNE_WOA: 减小对数螺旋系数 b=0.4, 收紧 3D 气泡网")

        # 混合算法专属调参逻辑
        elif current_algo == "HybridPSOGWO":
            if self.stuck_counter >= 1 or details.get('missed_target', 0) > 0:
                specific_params['pso_ratio'] = 0.5  
                actions_taken.append("TUNE_HYBRID: 延长 PSO 探路阶段比例至 50%，加强大范围视野")

        # 3. 共性参数宏观调控 (Macro-management)
        if is_failing:
            if self.algo_params['pop_size'] < 200:
                self.algo_params['pop_size'] += 20
                actions_taken.append("MACRO: INCREASE_POP_SIZE (增派搜索兵力)")
            if details.get('fatal_collision', 0) > 0 and self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                actions_taken.append("MACRO: INCREASE_MAX_ITER (延长规避计算工期)")

        if details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000:
            if self.eval_params.get('bspline_num_points', 100) < 150:
                self.eval_params['bspline_num_points'] += 10
                actions_taken.append("MACRO: ENHANCE_SMOOTHNESS (增加 B样条插值点数以柔化急弯)")

        if not actions_taken:
            actions_taken.append("MAINTAIN (当前状态极佳，维持原方)")

        for action in actions_taken:
            print(f"  └── 药方: \033[93m{action}\033[0m")

        return self.algo_params, self.eval_params, specific_params, is_finished