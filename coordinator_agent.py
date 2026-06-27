class CoordinatorAgent:
    def __init__(self):
        # 1. 算法共性参数管理 (算力预算)
        self.algo_params = {'pop_size': 50, 'max_iter': 100}
        
        # 2. 评价器物理参数管理 (物理规则)
        self.eval_params = {'chaikin_iterations': 3, 'min_waypoint_dist': 5.0}

        # 3. 核心监控指标
        self.meta_iteration = 0
        self.stuck_counter = 0
        self.last_score = float('inf')

    def analyze_and_act(self, total_score, details, current_algo):
        """
        六大算法全解锁的【终极调参大脑】
        """
        self.meta_iteration += 1
        actions_taken = []
        specific_params = {} 
        is_finished = False 
        
        print(f"\n[调参老中医] 第 {self.meta_iteration} 轮诊断中... (负责压榨 {current_algo} 的极限)")

        # ==========================================
        # 0. 收敛极限与早停判定 (榨汁机逻辑)
        # ==========================================
        improvement = self.last_score - total_score
        if self.last_score == float('inf'):
            improvement_rate = 1.0  # 第一轮算作 100% 进步
        else:
            improvement_rate = max(0, improvement) / (self.last_score + 1e-8) 
            
        self.last_score = total_score 

        is_perfectly_safe = (details.get('fatal_collision', 0) == 0 and 
                             details.get('missed_target', 0) == 0 and
                             details.get('sharp_turn', 0) == 0)
        
        if is_perfectly_safe and (self.meta_iteration > 1) and (improvement_rate < 0.01):
            print(f"   [全局通知] 路线已绝对安全，且收敛至极限(进步率 < 1%)，申请提前交卷！")
            is_finished = True
            return self.algo_params, self.eval_params, specific_params, is_finished

        ideal_dist = details.get('ideal_distance', 100.0)
        max_allowed_dist = ideal_dist * 1.15  
        
        # 绕路诊断
        if is_perfectly_safe and details.get('distance', 0) > max_allowed_dist:
            print(f"  [警告] 路线已安全，但总航程 {details.get('distance'):.1f}m 超过了动态底线 {max_allowed_dist:.1f}m，存在绕路！")
            
            if current_algo == "PSO":
                specific_params['c2'] = 1.0 
                actions_taken.append("TUNE_PSO: 降低社会认知 c2, 减少绕路甩尾，强行拉直航线")
                
            elif current_algo in ["ACO", "DSACO"]:
                specific_params['beta'] = 6.0 
                actions_taken.append(f"TUNE_{current_algo}: 极度强化目标牵引 beta=6.0, 强行拉直路线")
                
            elif current_algo == "SSA":
                specific_params['ST'] = 0.9 
                actions_taken.append("TUNE_SSA: 提高安全阈值 ST=0.9, 让麻雀安心走直线少乱跳")

        # 判定卡壳状态
        is_failing = details.get('fatal_collision', 0) > 0 or details.get('missed_target', 0) > 0
        if is_failing and improvement < 1000:
            self.stuck_counter += 1
            print(f"  [警告] 算法陷入瓶颈！累计卡壳: {self.stuck_counter} 次")
        else:
            self.stuck_counter = 0 

        # ==========================================
        # 1. 算法专属参数微操 (Micro-management)
        # ==========================================
        
        # ---- 【A. 蚁群系统 (ACO / DSACO)】 ----
        if current_algo in ["ACO", "DSACO"]:
            if self.stuck_counter >= 2:
                specific_params['rho'] = 0.5 
                actions_taken.append(f"TUNE_{current_algo}: 提高挥发率 rho=0.5, 迫使其遗忘烂路重搜")
            if details.get('sharp_turn', 0) > 0:
                specific_params['beta'] = 5.0
                actions_taken.append(f"TUNE_{current_algo}: 提高启发因子 beta=5.0, 增强终点目标吸引")

        # ---- 【B. 粒子群系统 (PSO)】 ----
        elif current_algo == "PSO":
            if details.get('missed_target', 0) > 0:
                specific_params['w_max'] = 0.99
                actions_taken.append("TUNE_PSO: 提高最大惯性权重 w_max=0.99, 强制粒子群向外乱窜探索")
            if details.get('smoothness', 0) > 2000:
                specific_params['c1'] = 2.2  # 增大个体认知，减少盲从头鸟导致的剧烈摆动
                specific_params['c2'] = 1.0
                actions_taken.append("TUNE_PSO: 调高 c1 调低 c2, 使其注重个体轨迹自适应平滑")

        # ---- 【C. 麻雀搜索系统 (SSA)】 ----
        elif current_algo == "SSA":
            if details.get('missed_target', 0) > 0:
                # 漏打卡：说明先锋部队不够。由于 num_producers 在 init 计算，我们直接远程重算并注入它！
                pop_size = self.algo_params['pop_size']
                specific_params['PD'] = 0.4  
                specific_params['num_producers'] = int(pop_size * 0.4) # 双重注入，强行生效！
                actions_taken.append("TUNE_SSA: 提高发现者比例 PD=0.4 并刷新先锋数量，扩大搜索网")
            if self.stuck_counter >= 1:
                # 卡壳：说明安全阈值卡得太死，麻雀不敢飞
                specific_params['ST'] = 0.6  # 降低安全阈值，逼迫麻雀产生危机感，大范围飞离
                actions_taken.append("TUNE_SSA: 降低安全阈值 ST=0.6，强制打散局部僵局")

        # ---- 【D. 灰狼优化系统 (GWO)】 ----
        elif current_algo == "GWO":
            if self.stuck_counter >= 1:
                # 灰狼卡壳：说明头三只狼被障碍物夹死了，带错路了
                # 药方：与其等 30 代再爆炸，不如让大脑缩短忍耐度，立刻触发大爆炸
                specific_params['stagnation_max'] = 12  
                actions_taken.append("TUNE_GWO: 降低停滞阈值至 12 代，加速触发全图重置大爆炸")

        # ---- 【E. 鲸鱼优化系统 (WOA)】 ----
        elif current_algo == "WOA":
            if details.get('smoothness', 0) > 3000:
                # 鲸鱼路线太崎岖：说明螺旋攻击范围过大甩尾严重
                specific_params['b'] = 0.4  # 减小对数螺旋线系数，收紧气泡网
                actions_taken.append("TUNE_WOA: 减小对数螺旋系数 b=0.4, 收紧气泡网以细腻局部轨迹")
            if details.get('missed_target', 0) > 0:
                # 漏打卡：说明全局围捕半径太大，错过了猎物
                actions_taken.append("TUNE_WOA: 激活外环资源，通过共性参数加派鲸鱼数量")

        # ==========================================
        # 2. 共性参数宏观调控 (Macro-management)
        # ==========================================
        if details.get('fatal_collision', 0) > 0 or details.get('missed_target', 0) > 100000:
            if self.algo_params['pop_size'] < 150:
                self.algo_params['pop_size'] += 20
                actions_taken.append("MACRO: INCREASE_POP_SIZE (增派兵力)")
            if details.get('fatal_collision', 0) > 0 and self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                actions_taken.append("MACRO: INCREASE_MAX_ITER (延长工期)")

        if details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000:
            if self.eval_params['chaikin_iterations'] < 6:
                self.eval_params['chaikin_iterations'] += 1
                actions_taken.append("MACRO: ENHANCE_SMOOTHNESS (加强物理割角平滑)")

        if not actions_taken:
            actions_taken.append("MAINTAIN (当前状态极佳，维持原方)")

        for action in actions_taken:
            print(f"  └── 药方: \033[93m{action}\033[0m")

        return self.algo_params, self.eval_params, specific_params, is_finished