class CoordinatorAgent:
    def __init__(self):
        # 1. 算法共性参数管理 (算力预算)
        self.algo_params = {'pop_size': 50, 'max_iter': 100}
        
        # 2. 评价器物理参数管理 (物理规则)
        # 将过时的 chaikin 移除，替换为 B-Spline 采样点数
        self.eval_params = {'bspline_num_points': 100, 'min_waypoint_dist': 5.0}

        # 3. 核心监控指标
        self.meta_iteration = 0
        self.stuck_counter = 0
        self.last_score = float('inf')


    def analyze_and_act(self, total_score, details, env_info, current_algo):
        """
        六大算法全解锁的【终极 3D 调参大脑】
        """
        self.meta_iteration += 1
        actions_taken = []
        specific_params = {} 
        is_finished = False 
        specific_params['emergency_escape'] = False # 默认关闭逃逸模式
        
        print(f"\n[调参] 第 {self.meta_iteration} 轮诊断中... (负责压榨 {current_algo} 的极限)")

        # ==========================================
        # 0. 收敛极限与早停判定 (榨汁机逻辑)
        # ==========================================
        improvement = self.last_score - total_score
        if self.last_score == float('inf'):
            improvement_rate = 1.0  
        else:
            improvement_rate = max(0, improvement) / (self.last_score + 1e-8) 
            
        self.last_score = total_score 

        # 【修改2：加入 3D 高度违规检测】
        is_perfectly_safe = (details.get('fatal_collision', 0) == 0 and 
                             details.get('missed_target', 0) == 0 and
                             details.get('sharp_turn', 0) == 0 and
                             details.get('altitude_violation', 0) == 0) # 必须不能遁地或冲天
        
        if is_perfectly_safe and (self.meta_iteration > 1) and (improvement_rate < 0.01):
            print(f"   [全局通知] 3D 航线已绝对安全，且收敛至极限(进步率 < 1%)，申请提前结束")
            is_finished = True
            return self.algo_params, self.eval_params, specific_params, is_finished

        ideal_dist = env_info.get('ideal_distance', 100.0)
        obs_count = env_info.get('obstacle_count', 0)
        
        dynamic_tolerance = 1.0 + (obs_count * 0.005)
        max_allowed_dist = ideal_dist * dynamic_tolerance
        
        if is_perfectly_safe and details.get('distance', 0) > max_allowed_dist:
            print(f"  [警告] 路线安全，但总航程 {details.get('distance'):.1f}m 超过动态底线 {max_allowed_dist:.1f}m (容忍度:{dynamic_tolerance:.2f}x)，存在绕路")
            if current_algo == "PSO":
                specific_params['c2'] = 1.0 
                actions_taken.append("TUNE_PSO: 降低社会认知 c2, 减少绕路甩尾，强行拉直航线")
            elif current_algo in ["ACO", "DSACO"]:
                specific_params['beta'] = 6.0 
                actions_taken.append(f"TUNE_{current_algo}: 极度强化目标牵引 beta=6.0, 强行拉直路线")
            elif current_algo == "SSA":
                specific_params['ST'] = 0.9 
                actions_taken.append("TUNE_SSA: 提高安全阈值 ST=0.9, 减少麻雀乱跳, 多走直线")

        # 判定卡壳
        is_failing = (details.get('fatal_collision', 0) > 0 or 
                      details.get('missed_target', 0) > 0 or
                      details.get('altitude_violation', 0) > 0 or
                      details.get('boundary_violation', 0) > 0)
                      
        if is_failing and improvement < 1000:
            self.stuck_counter += 1
            print(f"  [警告] 算法在 3D 空间陷入瓶颈！累计卡壳: {self.stuck_counter} 次")
            
            # 如果卡壳是因为撞墙导致的，立刻下达“激进逃逸指令”
            if details.get('fatal_collision', 0) > 0 and self.stuck_counter >= 2:
                print("  [紧急状态] 连续撞击建筑物！启动 Z 轴激进拉升预案！")
                
                if current_algo == "PSO":
                    specific_params['w_max'] = 1.5 # 极大增加惯性权重，冲破局部极小值
                    specific_params['c1'] = 2.5    # 鼓励粒子相信自己变异出的高空基因
                    actions_taken.append("EMERGENCY_PSO: 惯性超载，强制粒子全向乱窜寻找空中缺口")
                
                elif current_algo == "SSA":
                    specific_params['ST'] = 0.4    # 极度降低安全感，逼迫所有麻雀起飞逃亡
                    actions_taken.append("EMERGENCY_SSA: 触发恐慌机制，逼迫麻雀群体向上方大范围跳跃")
                    
                elif current_algo == "WOA":
                    specific_params['a'] = 2.0     # 强制重置鲸鱼的攻击圈大小
                    actions_taken.append("EMERGENCY_WOA: 扩大搜寻气泡网，寻找上方突围路径")

                elif current_algo == "GWO":
                    specific_params['emergency_escape'] = True
                    actions_taken.append("EMERGENCY_GWO: 激活紧急逃逸！向敢死队注入强力上升气流与转向基因！")
                    
        else:
            self.stuck_counter = 0

        # ==========================================
        # 1. 算法专属参数微操 (维持原样即可，数学逻辑通用)
        # ==========================================
        
        if current_algo in ["ACO", "DSACO"]:
            if self.stuck_counter >= 2:
                specific_params['rho'] = 0.5 
                actions_taken.append(f"TUNE_{current_algo}: 提高挥发率 rho=0.5, 迫使其遗忘烂路重搜")
            if details.get('sharp_turn', 0) > 0:
                specific_params['beta'] = 5.0
                actions_taken.append(f"TUNE_{current_algo}: 提高启发因子 beta=5.0, 增强终点目标吸引")

        elif current_algo == "PSO":
            if details.get('missed_target', 0) > 0:
                specific_params['w_max'] = 0.99
                actions_taken.append("TUNE_PSO: 提高最大惯性权重 w_max=0.99, 强制粒子向外乱窜探索高度")
            if details.get('smoothness', 0) > 2000:
                specific_params['c1'] = 2.2  
                specific_params['c2'] = 1.0
                actions_taken.append("TUNE_PSO: 调高 c1 调低 c2, 使其注重个体轨迹自适应平滑")

        elif current_algo == "SSA":
            if details.get('missed_target', 0) > 0:
                pop_size = self.algo_params['pop_size']
                specific_params['PD'] = 0.4  
                specific_params['num_producers'] = int(pop_size * 0.4) 
                actions_taken.append("TUNE_SSA: 提高发现者比例 PD=0.4, 扩大 3D 搜索网")
            if self.stuck_counter >= 1:
                specific_params['ST'] = 0.6  
                actions_taken.append("TUNE_SSA: 降低安全阈值 ST=0.6, 强制打散局部僵局")

        elif current_algo == "GWO":
            if self.stuck_counter >= 1:
                specific_params['stagnation_max'] = 12  
                actions_taken.append("TUNE_GWO: 降低停滞阈值至 12 代，加速触发全图重置大爆炸")

        elif current_algo == "WOA":
            if details.get('smoothness', 0) > 3000:
                specific_params['b'] = 0.4  
                actions_taken.append("TUNE_WOA: 减小对数螺旋系数 b=0.4, 收紧 3D 气泡网")
            if details.get('spacing_penalty', 0) > 100:
                specific_params['b'] = 2.0  
                actions_taken.append("TUNE_WOA: 增大螺旋系数 b=2.0, 扩张气泡网以打散聚集航点")
                if self.eval_params.get('min_waypoint_dist', 5.0) > 2.0:
                    self.eval_params['min_waypoint_dist'] -= 0.5
                    actions_taken.append("MACRO: 针对鲸鱼聚集特性，稍微下调物理最小航点排斥距离")
            if details.get('missed_target', 0) > 0:
                actions_taken.append("TUNE_WOA: 激活外环资源，加派鲸鱼数量寻找目标")

        # 混合算法专属调参逻辑
        elif current_algo == "HybridPSOGWO":
            if self.stuck_counter >= 1 or details.get('missed_target', 0) > 0:
                specific_params['pso_ratio'] = 0.5  # 增加 PSO 探路比例
                actions_taken.append("TUNE_HYBRID: 延长 PSO 探路阶段比例至 50%，加强全局 3D 视野突围搜索")
            if details.get('smoothness', 0) > 2000 and details.get('fatal_collision', 0) == 0:
                specific_params['pso_ratio'] = 0.1  # 减少探路，增加平滑打磨
                actions_taken.append("TUNE_HYBRID: 缩短探路，将 90% 的算力交给 GWO 灰狼进行极致平滑包围")

        # ==========================================
        # 2. 共性参数宏观调控 (Macro-management)
        # ==========================================
        # 针对 3D 难度，加大了派兵力度
        if is_failing:
            if self.algo_params['pop_size'] < 200:
                self.algo_params['pop_size'] += 20
                actions_taken.append("MACRO: INCREASE_POP_SIZE (增派 3D 搜索兵力)")
            if details.get('fatal_collision', 0) > 0 and self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                actions_taken.append("MACRO: INCREASE_MAX_ITER (延长规避计算工期)")

        # 将 Chaikin 平滑调控改为 B-Spline 点数调控
        if details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000:
            if self.eval_params.get('bspline_num_points', 100) < 150:
                self.eval_params['bspline_num_points'] += 10
                actions_taken.append("MACRO: ENHANCE_SMOOTHNESS (增加 B样条插值点数以柔化 3D 急弯)")

        if not actions_taken:
            actions_taken.append("MAINTAIN (当前状态极佳，维持原方)")

        for action in actions_taken:
            print(f"  └── 药方: \033[93m{action}\033[0m")

        return self.algo_params, self.eval_params, specific_params, is_finished