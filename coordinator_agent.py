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

    def analyze_and_act(self, total_score, details):
        """
        核心决策大脑：读取体检报告，输出下一轮的调参指令
        :param total_score: 最终总分
        :param details: 惩罚项明细字典
        :return: (更新后的算法参数, 更新后的评价器参数, 智能体执行的动作日志)
        """
        self.meta_iteration += 1
        actions_taken = []
        
        print(f"\n[决策大脑] 第 {self.meta_iteration} 轮分析中...")

        # ==========================================
        # 诊断 1：极其恶劣的失败 (撞墙 或 严重漏打卡)
        # 病因：算法完全陷入死胡同，或者搜索范围太小
        # 处方：暴增种群规模，加大全局探索力度
        # ==========================================
        if details.get('fatal_collision', 0) > 0 or details.get('missed_target', 0) > 100000:
            # 种群规模增加 20，但设一个上限防止内存撑爆
            if self.algo_params['pop_size'] < 150:
                self.algo_params['pop_size'] += 20
                actions_taken.append("INCREASE_POP_SIZE (增加种群以逃离局部最优)")
            
            # 如果撞墙严重，增加迭代次数让它慢慢试
            if details.get('fatal_collision', 0) > 0 and self.algo_params['max_iter'] < 500:
                self.algo_params['max_iter'] += 50
                actions_taken.append("INCREASE_MAX_ITER (增加迭代时间用于避障)")

        # ==========================================
        # 诊断 2：路线安全，但像无头苍蝇一直急转弯
        # 病因：环境走廊狭窄，或者航点靠得太近
        # 处方：增加 Chaikin 物理平滑次数，适当推开航点距离
        # ==========================================
        if details.get('sharp_turn', 0) > 0 or details.get('smoothness', 0) > 2000:
            if self.eval_params['chaikin_iterations'] < 6:
                self.eval_params['chaikin_iterations'] += 1
                actions_taken.append("ENHANCE_SMOOTHNESS (增加Chaikin割角次数)")
                
        if details.get('spacing_penalty', 0) > 1000:
            if self.eval_params['min_waypoint_dist'] < 10.0:
                self.eval_params['min_waypoint_dist'] += 1.0
                actions_taken.append("INCREASE_SPACING (增加航点排斥力)")

        # ==========================================
        # 诊断 3：完美收敛 (无致命伤，分数极低)
        # 病因：算法已经找到很好的解了
        # 处方：尝试“裁员”或者“减少迭代时间”，压榨计算效率
        # ==========================================
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

        return self.algo_params, self.eval_params