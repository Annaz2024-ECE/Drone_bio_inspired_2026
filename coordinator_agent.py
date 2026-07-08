class CoordinatorAgent:
    def __init__(self):
        self.algo_params = {'pop_size': 50, 'max_iter': 100}
        self.eval_params = {
            'bspline_num_points': 100, 
            'min_waypoint_dist': 5.0,
            'max_turn_angle': 120.0  
        }
        self.meta_iteration = 0
        self.stuck_counter = 0
        self.last_score = float('inf')

    def analyze_and_act(self, total_score, details, env_info, current_algo):
        self.meta_iteration += 1
        actions_taken = []
        specific_params = {} 
        is_finished = False 
        
        # 物理指令保留（因为这涉及复杂的 3D 坐标操作，需要底层的雷达/下压接收器配合）
        specific_params['emergency_escape'] = False 
        specific_params['radar_guidance'] = False
        specific_params['press_down'] = False       
        specific_params['lift_up'] = False          
        
        print(f"\n[调参] 第 {self.meta_iteration} 轮诊断中... (负责压榨 {current_algo} 的极限)")

        improvement = self.last_score - total_score
        if self.last_score == float('inf'):
            improvement_rate = 0.05  
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
        # 通用物理避障/雷达/压低指令 
        # ==========================================
        if details.get('fatal_collision', 0) > 0:
            specific_params['lift_up'] = True
            self.eval_params['max_turn_angle'] = 150.0 
            actions_taken.append("UNIVERSAL: 遭遇建筑碰撞！下达全局 [紧急拉升] 指令，放宽转弯限制")
        else:
            self.eval_params['max_turn_angle'] = 120.0 

        is_safe_but_high = is_perfectly_safe and (details.get('gravity_cost', 0) > 1500)
        if is_safe_but_high:
            specific_params['press_down'] = True
            actions_taken.append("UNIVERSAL: 航线安全但能耗高，下达全局 [贴地压低] 指令")

        if details.get('missed_target', 0) > 0:
            specific_params['radar_guidance'] = True 
            actions_taken.append("UNIVERSAL: 偏离打卡点！激活全系统 [雷达空投] 机制")

        # ==========================================
        # 核心：六大算法原生参数暴力接管区 (无需修改底层代码)
        # ==========================================
        needs_smooth = (details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000 or details.get('loop_penalty', 0) > 0)

        if current_algo == "PSO":
            # 只要把名字写对 (跟 PSOPlanner 里 self.xxx 名字一致)，底下就自动生效！
            if needs_smooth:
                specific_params['w_max'] = 0.4  # 直接压制惯性
                specific_params['c1'] = 2.0     
                specific_params['c2'] = 0.5     
                actions_taken.append("TUNE_PSO: 强行修改底层参数 (w_max=0.4, c2=0.5)，限制粒子冲刺以打磨平滑度！")
            elif self.stuck_counter >= 1:
                specific_params['w_max'] = 1.2  
                actions_taken.append("TUNE_PSO: 卡壳！直接篡改底层 (w_max=1.2)，强制惯性超载冲出瓶颈！")

        elif current_algo == "WOA":
            if needs_smooth:
                specific_params['b'] = 0.2      # 直接收缩气泡网
                actions_taken.append("TUNE_WOA: 强行修改底层参数 (b=0.2)，收紧螺旋圈以打磨平滑度！")
            elif self.stuck_counter >= 1:
                specific_params['b'] = 2.0      
                actions_taken.append("TUNE_WOA: 卡壳！直接篡改底层 (b=2.0)，强行放大螺旋网扩大搜索！")

        elif current_algo == "SSA":
            if needs_smooth:
                specific_params['ST'] = 0.95    # 直接拉满安全感
                actions_taken.append("TUNE_SSA: 强行修改底层参数 (ST=0.95)，让麻雀停止大跳跃以打磨平滑度！")
            elif self.stuck_counter >= 1:
                specific_params['ST'] = 0.4     
                actions_taken.append("TUNE_SSA: 卡壳！直接篡改底层 (ST=0.4)，制造恐慌打散麻雀群！")

        elif current_algo in ["ACO", "DSACO"]:
            if needs_smooth:
                specific_params['beta'] = 5.0   
                specific_params['alpha'] = 0.5  
                actions_taken.append(f"TUNE_{current_algo}: 强行修改底层参数 (beta=5.0)，增大终点牵引拉直路线！")
            elif self.stuck_counter >= 2:
                specific_params['rho'] = 0.6    
                actions_taken.append(f"TUNE_{current_algo}: 卡壳！直接篡改底层 (rho=0.6)，加快挥发遗忘死胡同！")

        elif current_algo == "GWO":
            if self.stuck_counter >= 1:
                specific_params['stagnation_max'] = 12  
                actions_taken.append("TUNE_GWO: 卡壳！强行修改停滞阈值 stagnation_max=12，加速大爆炸！")
                
        elif current_algo == "HybridPSOGWO":
            if needs_smooth and not is_failing:
                specific_params['pso_ratio'] = 0.1 
                actions_taken.append("TUNE_HYBRID: 强行修改底层分配 (pso_ratio=0.1)，砍掉探路，交给GWO精修！")
            elif self.stuck_counter >= 1 or details.get('missed_target', 0) > 0:
                specific_params['pso_ratio'] = 0.5 
                actions_taken.append("TUNE_HYBRID: 卡壳！直接拉高 PSO 探路比例至 50% 加强大范围突围！")

        # ==========================================
        # 宏观算力调配
        # ==========================================
        if is_failing:
            if self.algo_params['pop_size'] < 200:
                self.algo_params['pop_size'] += 20
                actions_taken.append("MACRO: INCREASE_POP_SIZE (增派搜索兵力)")
            if details.get('fatal_collision', 0) > 0 and self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                actions_taken.append("MACRO: INCREASE_MAX_ITER (延长规避计算工期)")

        if not actions_taken:
            actions_taken.append("MAINTAIN (当前状态极佳，维持原方)")

        for action in actions_taken:
            print(f"  └── 药方: \033[93m{action}\033[0m")

        return self.algo_params, self.eval_params, specific_params, is_finished