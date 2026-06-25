class CoordinatorAgent:
    def __init__(self):
        """
        初始化智能体，设定各参数的默认初始状态与安全边界
        """
        # 1. 算法通用参数 (适用于 GWO, PSO, SSA, ACO, WOA)
        self.algo_params = {
            'pop_size': 50,       # 种群规模 (对于ACO就是 num_ants，对于PSO就是 num_particles)
            'max_iter': 100       # 最大迭代次数
        }
        
        # 2. 评价器(环境/物理)通用参数
        self.eval_params = {
            'chaikin_iterations': 3,
            'min_waypoint_dist': 5.0
        }

        # 记录调整历史
        self.meta_iteration = 0

        self.stuck_counter = 0            # 记录连续失败的次数
        self.last_score = float('inf')    # 记录上一轮的分数

    def analyze_and_act(self, total_score, details, current_algo):
        """
        核心决策大脑：读取体检报告，输出下一轮的调参指令，甚至更换算法
        :param total_score: 最终总分
        :param details: 惩罚项明细字典
        :param current_algo: 当前正在运行的算法名字 (如 "ACO")
        :return: (算法参数, 评价器参数, 下一轮要用的算法, 动作日志)
        """
        self.meta_iteration += 1
        actions_taken = []
        next_algo = current_algo  # 默认不换算法，继续用当前的
        
        print(f"\n[决策大脑] 第 {self.meta_iteration} 轮分析中... (当前算法: {current_algo})")

        # ==========================================
        # 核心新增：卡壳诊断与动态换人
        # ==========================================
        # 1. 计算本轮进步幅度
        improvement = self.last_score - total_score
        self.last_score = total_score

        # 2. 诊断是否陷入死局：如果还在撞墙或漏打卡，且分数下降不到 1000 分
        is_failing = details.get('fatal_collision', 0) > 0 or details.get('missed_target', 0) > 0
        if is_failing and improvement < 1000:
            self.stuck_counter += 1
            print(f"  [警告] 当前算法 {current_algo} 陷入瓶颈！累计卡壳: {self.stuck_counter} 次")
        else:
            self.stuck_counter = 0 # 只要有明显进步，或者已经跑通了，耐心值就清零

        # 3. 耐心值耗尽：触发终极绝招动态换人！
        if self.stuck_counter >= 2:
            print("  🚨 触发【动态算法切换】机制！")
            if current_algo == "ACO":
                next_algo = "GWO"
                actions_taken.append(f"SWITCH_ALGO (ACO卡壳，切换为 {next_algo})")
            elif current_algo == "GWO":
                next_algo = "PSO"
                actions_taken.append(f"SWITCH_ALGO (GWO卡壳，切换为 {next_algo})")
            elif current_algo == "PSO":
                next_algo = "ACO"
                actions_taken.append(f"SWITCH_ALGO (PSO卡壳，切换回 {next_algo})")
            
            self.stuck_counter = 0  # 换人后，耐心值重新计算

        # ==========================================
        # 以下保留你原有的参数微调逻辑
        # ==========================================
        # 诊断 1：极其恶劣的失败 (撞墙 或 严重漏打卡)
        if details.get('fatal_collision', 0) > 0 or details.get('missed_target', 0) > 100000:
            if self.algo_params['pop_size'] < 150:
                self.algo_params['pop_size'] += 20
                actions_taken.append("INCREASE_POP_SIZE (增加种群以逃离局部最优)")
            if details.get('fatal_collision', 0) > 0 and self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                actions_taken.append("INCREASE_MAX_ITER (增加迭代时间用于避障)")

        # 诊断 2：路线安全，但像无头苍蝇一直急转弯
        if details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000:
            if self.eval_params['chaikin_iterations'] < 6:
                self.eval_params['chaikin_iterations'] += 1
                actions_taken.append("ENHANCE_SMOOTHNESS (增加Chaikin割角次数)")
        if details.get('spacing_penalty', 0) > 1000:
            if self.eval_params['min_waypoint_dist'] < 10.0:
                self.eval_params['min_waypoint_dist'] += 1.0
                actions_taken.append("INCREASE_SPACING (增加航点排斥力)")

        # 诊断 3：完美收敛 (无致命伤，分数极低)
        is_perfect = details.get('fatal_collision', 0) == 0 and \
                     details.get('missed_target', 0) == 0 and \
                     details.get('sharp_turn', 0) == 0
        if is_perfect and total_score < 5000:
            if self.algo_params['pop_size'] > 30:
                self.algo_params['pop_size'] -= 10
                actions_taken.append("DECREASE_POP_SIZE (已找到最优解，裁员优化速度)")
            if self.algo_params['max_iter'] > 80:
                self.algo_params['max_iter'] -= 20
                actions_taken.append("DECREASE_MAX_ITER (减少不必要的迭代时间)")

        # 如果无事发生
        if not actions_taken:
            actions_taken.append("MAINTAIN (当前参数表现良好，维持现状)")

        # 打印动作日志
        for action in actions_taken:
            print(f"  └── 动作指令: \033[93m{action}\033[0m")

        # 注意这里返回了 next_algo
        return self.algo_params, self.eval_params, next_algo