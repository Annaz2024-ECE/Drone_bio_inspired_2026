import importlib

class CoordinatorAgent:
    def __init__(self):
        # 宏观参数
        self.algo_params = {'pop_size': 50, 'max_iter': 100}
        self.eval_params = {
            'bspline_num_points': 100, 
            'min_waypoint_dist': 5.0,
            'max_turn_angle': 120.0,
            'v_cruise': 13.17
        }
        self.meta_iteration = 0
        self.stuck_counter = 0
        self.last_score = float('inf')

        # 专门用来记忆上一次下发的微观算法参数
        self.last_specific_params = {}

    def analyze_and_act(self, total_score, details, env_info, current_algo):
        # ==========================================
        # 调参前，先给所有当前参数“拍照存档”
        # ==========================================
        old_algo_params = self.algo_params.copy()
        old_eval_params = self.eval_params.copy()

        self.meta_iteration += 1
        actions_taken = []
        specific_params = {} 
        is_finished = False 
        
        # 物理指令默认状态 (全部由 BasePlanner 统一执行)
        specific_params['emergency_escape'] = False 
        specific_params['radar_guidance'] = False
        specific_params['press_down'] = False       
        specific_params['lift_up'] = False          
        specific_params['apply_laplacian'] = False  
        specific_params['apply_repulsion'] = False  
        
        print(f"\n[调参] 第 {self.meta_iteration} 轮诊断中... (当前执行算法: {current_algo})")

        improvement = self.last_score - total_score
        if self.last_score == float('inf'):
            improvement_rate = 0.05  
        else:
            improvement_rate = max(0, improvement) / (self.last_score + 1e-8) 
            self.last_score = total_score 

        is_perfectly_safe = (details.get('fatal_collision', 0) == 0 and 
                     details.get('missed_target_base', 0) == 0 and  
                     details.get('sharp_turn', 0) == 0 and
                     details.get('altitude_violation', 0) == 0)
        
        if is_perfectly_safe and (self.meta_iteration > 1) and (improvement_rate < 0.005):
            print(f"   [全局通知] 3D 航线已绝对安全，且收敛至极限(进步率 < 0.5%)，申请提前结束")
            is_finished = True
            return self.algo_params, self.eval_params, specific_params, is_finished, []

        is_failing = (details.get('fatal_collision', 0) > 0 or 
                    details.get('missed_target_base', 0) > 0 or 
                    details.get('altitude_violation', 0) > 0 or
                    details.get('boundary_violation', 0) > 0)
                            
        if is_failing and improvement < 1000:
            self.stuck_counter += 1
            print(f"  [警告] 算法在 3D 空间陷入瓶颈！累计卡壳: {self.stuck_counter} 次")
        else:
            self.stuck_counter = 0

        # ==========================================
        # 宏观调控一：通用物理避障与空间引导 (Physics Directives)
        # 适用于所有继承了 BasePlanner 的算法
        # ==========================================
        if details.get('fatal_collision', 0) > 0:
            specific_params['lift_up'] = True
            self.eval_params['max_turn_angle'] = 150.0 
            actions_taken.append("MACRO (物理): 遭遇撞击！下达全局 [紧急拉升] 指令，放宽转弯限制！")
        else:
            self.eval_params['max_turn_angle'] = 120.0 # 危机解除，回归120度，进一步优化

        if details.get('margin_violation', 0) > 0 and details.get('fatal_collision', 0) == 0:
            specific_params['apply_repulsion'] = True
            actions_taken.append("MACRO (物理): 航线极度擦墙！启动 [侧向斥力算子]，强行推离危险边缘！")

        is_safe_but_high = is_perfectly_safe and (details.get('gravity_cost', 0) > 1500)
        if is_safe_but_high:
            specific_params['press_down'] = True
            actions_taken.append("MACRO (物理): 航线安全但能耗高，下达全局 [贴地压低] 指令！")

        if details.get('missed_target_base', 0) > 0:
            specific_params['radar_guidance'] = True 
            actions_taken.append("MACRO (物理): 偏离打卡点！激活全系统 [雷达空投靶向] 机制！")

        # 既然拉普拉斯平滑也是基类的绝招，把它也归入宏观物理调控！
        needs_smooth = (details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000 or details.get('loop_penalty', 0) > 0)
        if needs_smooth:
            specific_params['apply_laplacian'] = True
            actions_taken.append("MACRO (物理): 路线不平滑或绕圈！下达 [拉普拉斯平滑算子] 橡皮筋拉直指令！")

        # ==========================================
        # 宏观调控二：宏观算力/预算调配 (Budget Allocation)
        # ==========================================
        if is_failing:
            if self.algo_params['pop_size'] < 200:
                self.algo_params['pop_size'] += 20
                actions_taken.append("MACRO (算力): INCREASE_POP_SIZE (大本营增派搜索兵力)")
            if details.get('fatal_collision', 0) > 0 and self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                actions_taken.append("MACRO (算力): INCREASE_MAX_ITER (延长规避计算工期)")

        # ==========================================
        # 宏观调控三：能耗与时间的“自动换挡”博弈 (Auto-Gearbox)
        # ==========================================
        change_power = details.get('change_power_pen', 0)
        current_v = self.eval_params['v_cruise']
        
        # 降挡逻辑：如果机动耗电爆炸，且速度还有下降空间（底线保底为 8.0m/s 防止不动）
        if change_power > 50000 and current_v > 8.0:
            self.eval_params['v_cruise'] = max(8.0, current_v - 1.5) # 每次踩 1.5 的刹车
            actions_taken.append(f"MACRO (能耗): 机动耗电爆炸！全局降速至 {self.eval_params['v_cruise']:.2f} m/s 缓解转弯压力！")
            
        # 升挡逻辑：如果机动耗电很小（证明路线已平滑），为了降低时间惩罚，踩油门恢复速度！
        elif change_power < 15000 and current_v < 13.17:
            self.eval_params['v_cruise'] = min(13.17, current_v + 1.0) # 每次踩 1.0 的油门
            actions_taken.append(f"MACRO (能耗): 路线已平滑，加速至 {self.eval_params['v_cruise']:.2f} m/s 减少飞行时间惩罚！")

        # ==========================================
        # 微观调控：动态加载算法专属的内部参数特工 (Algorithm Internal Tuning)
        # ==========================================
        module_name = f"agent_{current_algo.lower()}"
        try:
            # 动态导入对应的专属文件，例如 'agent_ga', 'agent_pso'
            micro_agent = importlib.import_module(module_name)
            # 呼叫专属特工看病，并获取药方
            micro_params, micro_actions = micro_agent.tune(details, self.stuck_counter, is_failing, needs_smooth)
            # 将微观参数和日志合并进总字典
            specific_params.update(micro_params)
            actions_taken.extend(micro_actions)
        except ImportError:
            # 如果该算法没有专属文件，就忽略微操，仅靠强大的物理引擎兜底
            pass

        if not actions_taken:
            actions_taken.append("MAINTAIN (当前状态极佳，全军保持原方推进)")

        # ==========================================
        # 🔥 【新增】：精准的状态比对引擎 (State Diff)
        # ==========================================
        param_changes = []
        
        # 1. 抓取【宏观算力参数】的数值突变
        for k, v in self.algo_params.items():
            if old_algo_params.get(k) != v:
                param_changes.append(f"{k}: {old_algo_params.get(k)}->{v}")
                
        # 2. 抓取【物理评价参数】的数值突变
        for k, v in self.eval_params.items():
            if old_eval_params.get(k) != v:
                param_changes.append(f"{k}: {old_eval_params.get(k)}->{v}")
                
        # 3. 抓取【微观算法参数】与【物理动作】的切变
        for k, v in specific_params.items():
            # 物理动作是布尔值触发器，单独处理
            if k in ['emergency_escape', 'radar_guidance', 'press_down', 'lift_up', 'apply_laplacian', 'apply_repulsion']:
                if v is True:
                    param_changes.append(f"Act: {k}")
                continue
            
            # 数值型微观参数比对
            old_val = self.last_specific_params.get(k, "Init")
            if old_val != v:
                param_changes.append(f"{k}: {old_val}->{v}")
        
        # 刷新记忆中枢，供下一轮比对使用
        self.last_specific_params.update(specific_params)

        for action in actions_taken:
            print(f"  └── 药方: \033[93m{action}\033[0m")

        # 【修改返回值】：把精准的 param_changes 列表扔给主程序
        return self.algo_params, self.eval_params, specific_params, is_finished, param_changes