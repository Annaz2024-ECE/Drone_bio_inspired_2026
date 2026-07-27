import importlib

class CoordinatorAgent:
    def __init__(self):
        # 宏观参数
        self.algo_params = {'pop_size': 55, 'max_iter': 100}
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

        # ==========================================
        # 物理干预强度记忆体 (0.0 表示关闭，1.0 表示满功率)
        # ==========================================
        self.intensities = {
            'laplacian': 0.0,
            'repulsion': 0.0,
            'lift_up': 0.0,
            'press_down': 0.0
        }
        # 渐进因子：每轮最多增加或减少 20% 的强度，给种群留出 5 轮的适应缓冲期
        self.gradient_step = 0.2

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
        specific_params['shattering_kick'] = False
        
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

        # ==========================================
        # 听从建议，化繁为简的“两极管”诊断系统
        # ==========================================
        # 只要没有撞墙，也没有漏打卡，那就是“存活状态”
        is_failing = (details.get('fatal_collision', 0) > 0 or details.get('missed_target', 0) > 0)
        needs_smooth = False 
        
        if is_failing and improvement < 1000:
            self.stuck_counter += 1
            print(f"  [警告] 算法在 3D 空间陷入生死瓶颈！累计卡壳: {self.stuck_counter} 次")
        else:
            self.stuck_counter = 0

        # 提前计算 needs_smooth，供两阶段共用
        needs_smooth = (details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000)
        has_loops = details.get('loop_penalty', 0) > 0 # 将绕圈判定升级为独立的“结构性危机”

        # ------------------------------------------
        # 阶段 1 (生死底线)：撞墙、漏打卡
        # ------------------------------------------
        if is_failing:
            print("  [状态] 当前处于危机状态：优先保命与打卡，屏蔽一切高阶优化！")
            self.eval_params['max_turn_angle'] = 150.0 # 放开转弯限制，允许急转弯逃生
            
            if details.get('fatal_collision', 0) > 0:
                specific_params['lift_up'] = True
                actions_taken.append("MACRO [绝对底线]: 遭遇撞击！下达 [紧急拉升] 指令！")
                
            if details.get('missed_target', 0) > 0:
                specific_params['radar_guidance'] = True 
                actions_taken.append("MACRO [绝对底线]: 偏离打卡点！激活全系统 [雷达空投靶向] 机制！")

            # 只有遇到生死危机，才延长计算时间
            if self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                
            # 在危机状态下，强行缓慢撤销高级平滑和斥力干预，把自由度还给保命动作
            self.intensities['laplacian'] = max(0.0, self.intensities['laplacian'] - self.gradient_step)
            self.intensities['repulsion'] = max(0.0, self.intensities['repulsion'] - self.gradient_step)
                
        # ------------------------------------------
        # 阶段 2 (全面精修优化)：合规、平滑、能耗、时间
        # ------------------------------------------
        else:
            self.eval_params['max_turn_angle'] = 120.0
            print("  [状态] 危机解除，进入全面精修优化阶段 (合规、平滑、能耗)...")

            # 算力弹性释放逻辑 
            if self.algo_params['max_iter'] > 100:
                old_iter = self.algo_params['max_iter']
                self.algo_params['max_iter'] = 100
                actions_taken.append(f"MACRO [算力释放]: 生死危机解除！撤销抢救算力，max_iter 从 {old_iter} 恢复至常态 100，加速精修！")
            # 重拳治乱！优先解决绕圈死结，再谈精修
            if has_loops:
                specific_params['shattering_kick'] = True
                # 强行给底层算法下发超高探索特权，暴力破局
                specific_params['mutation_rate'] = 0.8     # 80%的狼强制变异
                specific_params['mutation_scale'] = 5.0    # 允许产生最大 5 米的大跨步跳跃
                # 当航线打结时，必须立刻刹车，降低平滑强度，不让拉普拉斯帮倒忙
                self.intensities['laplacian'] = max(0.0, self.intensities['laplacian'] - self.gradient_step * 2)
                actions_taken.append("MACRO [拓扑破局]: 检测到严重的航线绕圈死结！激活 [高能扰动算子]，压制平滑，强制狼群炸开探索！")
            else:
                # 只有不绕圈时，拉普拉斯平滑才允许正常渐进
                if needs_smooth:
                    self.intensities['laplacian'] = min(1.0, self.intensities['laplacian'] + self.gradient_step)
                    actions_taken.append(f"MACRO [优化-平滑]: 路线曲折，拉普拉斯平滑渐进至 {self.intensities['laplacian']*100:.0f}%")
                else:
                    self.intensities['laplacian'] = max(0.0, self.intensities['laplacian'] - self.gradient_step)

            # 越界与擦墙优化 (Boundary & Margin)
            if details.get('boundary_violation', 0) > 0 or details.get('margin_violation', 0) > 0:
                self.intensities['repulsion'] = min(1.0, self.intensities['repulsion'] + self.gradient_step)
                actions_taken.append(f"MACRO [优化-安全]: 航线越界，侧向斥力渐进至 {self.intensities['repulsion']*100:.0f}%")
            else:
                self.intensities['repulsion'] = max(0.0, self.intensities['repulsion'] - self.gradient_step)

            # --- 互斥防爆锁 ---
            if self.intensities['laplacian'] > 0.5 and self.intensities['repulsion'] > 0.5:
                self.intensities['laplacian'] *= 0.7
                self.intensities['repulsion'] *= 0.7
                actions_taken.append(" [冲突抑制]: 平滑与斥力同时高强度触发，启动消解因子防畸变")

            # 超高与重力势能优化 (Altitude & Gravity)
            if details.get('altitude_violation', 0) > 0 or details.get('gravity_cost', 0) > 1500:
                specific_params['press_down'] = True
                actions_taken.append("MACRO [优化-高度]: 路线超高或重力能耗大，下达 [贴地压低] 指令！")

            # 4. 自动变速箱 (Speed/Time) 优化 - 【顺便调优：降低阈值至 40000 让你刚才的 4.2万 能够成功触发降速】
            change_power = details.get('change_power_pen', 0)
            current_v = self.eval_params['v_cruise']
            
            if change_power > 40000 and current_v > 8.0: # 阈值从5w调到4w
                self.eval_params['v_cruise'] = max(8.0, current_v - 1.5)
                actions_taken.append(f"MACRO [优化-机动]: 机动耗电过高！降速至 {self.eval_params['v_cruise']:.2f} m/s 缓解急弯！")
            elif change_power < 15000 and current_v < 13.17 and not has_loops: # 绕圈时不瞎提速
                self.eval_params['v_cruise'] = min(13.17, current_v + 1.0)
                actions_taken.append(f"MACRO [优化-时间]: 路线已丝滑，提速至 {self.eval_params['v_cruise']:.2f} m/s 缩短飞行时间！")

        # ==========================================
        # 统一装载物理干预强度并下发给 Planner
        # ==========================================
        self.intensities['laplacian'] = round(self.intensities['laplacian'], 3)
        self.intensities['repulsion'] = round(self.intensities['repulsion'], 3)

        specific_params['laplacian_intensity'] = self.intensities['laplacian']
        specific_params['repulsion_intensity'] = self.intensities['repulsion']
        
        specific_params['apply_laplacian'] = self.intensities['laplacian'] > 0
        specific_params['apply_repulsion'] = self.intensities['repulsion'] > 0

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
        # 精准的状态比对引擎 (State Diff)
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