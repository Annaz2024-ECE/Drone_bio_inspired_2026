def tune(details, stuck_counter, is_failing, needs_smooth):
    """ SSA (麻雀算法) 专属内部参数调优特工 """
    params = {}
    actions = []

    # 提取物理惩罚状态（与 GA/PSO 统一）
    fatal_col = details.get('fatal_collision', 0.0) > 0
    missed_tgt = (details.get('missed_target', 0.0) + details.get('missed_target_base', 0.0)) > 0
    has_loops = details.get('loop_penalty', 0.0) > 0 or details.get('shattering_kick', False)

    # ---------- 优先级 1：整体失败 ----------
    if is_failing:
        if fatal_col:
            # 致命碰撞 → 紧急逃生，压低安全阈值，大幅提升发现者/侦察者比例和突变率
            params['ST'] = 0.3
            params['PD'] = 0.4          # 更多发现者负责探索
            params['SD'] = 0.2          # 更多侦察者感知危险
            params['mutation_rate'] = 0.4
            params['emergency_escape'] = True
            actions.append("MICRO [SSA]: 致命碰撞！紧急逃生 (emergency_escape) 并降低ST=0.3，提高PD=0.4，突变率0.4，激活莱维跳跃。")
        elif missed_tgt:
            # 漏打卡 → 雷达空投，适度提高发现者，保持中等安全阈值
            params['ST'] = 0.7
            params['PD'] = 0.3
            params['SD'] = 0.15
            params['mutation_rate'] = 0.2
            params['radar_guidance'] = True
            actions.append("MICRO [SSA]: 漏打卡！启动雷达空投 (radar_guidance) 并调整PD=0.3，SD=0.15，突变率0.2，强化全局寻靶。")

    # ---------- 优先级 2：演化卡壳 ----------
    elif stuck_counter >= 1:
        # 动态降低安全阈值，提升突变率和侦察者数量，开启紧急逃生
        target_st = max(0.2, 0.6 - stuck_counter * 0.1)   # 随卡壳次数递减
        target_mutation = min(0.3 + stuck_counter * 0.1, 0.6)
        params['ST'] = target_st
        params['PD'] = 0.35
        params['SD'] = 0.2
        params['mutation_rate'] = target_mutation
        params['emergency_escape'] = True
        actions.append(f"MICRO [SSA]: 演化卡壳 (x{stuck_counter})！降低ST={target_st:.2f}，提升突变率={target_mutation:.2f}，开启紧急逃生。")

    # ---------- 优先级 3：航线绕圈 ----------
    elif has_loops:
        # 绕圈死循环 → 碎环击，大幅提高发现者和突变率，压低安全阈值打散群体
        params['ST'] = 0.4
        params['PD'] = 0.5
        params['SD'] = 0.25
        params['mutation_rate'] = 0.5
        params['shattering_kick'] = True
        actions.append("MICRO [SSA]: 航线绕圈！触发碎环击 (shattering_kick) 并降低ST=0.4，提高PD=0.5，突变率0.5，强行重置麻雀群。")

    # ---------- 优先级 4：需要平滑 ----------
    elif needs_smooth:
        # 安全平滑 → 高安全阈值，降低发现者和突变率，维持稳定雕刻
        params['ST'] = 0.95
        params['PD'] = 0.15
        params['SD'] = 0.05
        params['mutation_rate'] = 0.05
        actions.append("MICRO [SSA]: 航线安全！拉高ST=0.95，降低PD=0.15和突变率=0.05，配合物理雕刻。")

    return params, actions