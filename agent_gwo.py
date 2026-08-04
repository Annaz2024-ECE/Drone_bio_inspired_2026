def tune(details, stuck_counter, is_failing, needs_smooth):
    """ GWO (灰狼算法) 专属内部参数调优特工 (WOA 同款架构标准版) """
    params = {}
    actions = []
    
    # ==========================================
    # 1. 统一初始化默认参数 (保底基因)
    # ==========================================
    mutation_rate = 0.2      # 默认变异率
    mutation_scale = 0.5     # 默认变异步长比例
    levy_beta = 1.5          # 默认莱维分布指数 (1.5 为温和分布)
    stagnation_max = 20      # 默认头狼最大容忍停滞代数 (20代不进步就换狼)

    # ==========================================
    # 2. 提取状态标志位 
    # ==========================================
    fatal_col = details.get('fatal_collision', 0.0) > 0
    missed_tgt = details.get('missed_target', 0.0) > 0
    has_loops = details.get('loop_penalty', 0.0) > 0 or details.get('shattering_kick', False)
    alt_violation = details.get('altitude_violation', 0.0) > 0
    
    # ==========================================
    # 3. 优先级诊断树
    # ==========================================
    if is_failing:
        if fatal_col:
            mutation_rate = 0.6  
            mutation_scale = 2.0 
            levy_beta = 1.2      
            actions.append("MICRO [GWO]: 严重碰撞！激发 60% 变异率与 [高能莱维瞬移] 强行逃生。")
            
        elif missed_tgt:
            mutation_rate = 0.1
            mutation_scale = 0.2
            actions.append("MICRO [GWO]: 目标捕获不全！降低变异率 (0.1)，要求狼群配合雷达靶向集中！")

    elif stuck_counter >= 1:
        # 学习 WOA：随卡壳轮数指数级递增变异率，并逐步缩短头狼任期逼迫种群换血
        mutation_rate = min(0.8, 0.2 + (0.1 * stuck_counter))
        mutation_scale = min(3.0, 0.5 + (0.4 * stuck_counter))
        stagnation_max = max(5, 20 - (3 * stuck_counter))
        actions.append(f"MICRO [GWO]: 连续卡壳 {stuck_counter} 轮！渐进提高变异率 ({mutation_rate:.2f})，缩短头狼任期 ({stagnation_max})。")

    elif has_loops:
        mutation_rate = 0.7
        mutation_scale = 2.5
        levy_beta = 1.2
        actions.append("MICRO [GWO]: 检测到航线绕圈死结！大幅提升变异率与步长进行拓扑破局。")

    elif alt_violation:
        mutation_rate = 0.3
        mutation_scale = 0.1
        actions.append("MICRO [GWO]: 配合贴地压低，开启中度下探变异。")

    elif needs_smooth:
        mutation_rate = 0.1   
        mutation_scale = 0.05 
        levy_beta = 1.5      
        actions.append("MICRO [GWO]: 路径过于曲折/抖动！注入镇定剂 (限制狼群乱跑)，专心打磨局部细节。")

    # ==========================================
    # 4. 统一打包参数返回
    # ==========================================
    params['mutation_rate'] = mutation_rate
    params['mutation_scale'] = mutation_scale
    params['levy_beta'] = levy_beta
    params['stagnation_max'] = stagnation_max

    return params, actions