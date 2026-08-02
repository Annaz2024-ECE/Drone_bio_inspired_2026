def tune(details, stuck_counter, is_failing, needs_smooth):
    params = {}
    actions = []
    b_val = 1.0              # 默认对数螺旋形状参数
    levy_scale = 0.05        # 默认莱维大跳跃步长比例
    levy_beta = 1.5
    p_threshold = 0.5        # 默认包围与螺旋运动的分界线
    a_decay_power = 1.0      # 默认收敛因子衰减指数

    fatal_col = details.get('fatal_collision', 0.0) > 0
    missed_tgt = details.get('missed_target', 0.0) > 0
    has_loops = details.get('loop_penalty', 0.0) > 0 or details.get('shattering_kick', False)
    
    if is_failing:
        if fatal_col:
            p_threshold = 0.65  
            a_decay_power = 0.7  
            actions.append("MICRO [WOA]: 严重碰撞！启动大步长莱维扰动 (scale=0.12)，偏向全局包围逃生。")
        elif missed_tgt:
            b_val = 1.4
            p_threshold = 0.4
            actions.append("MICRO [WOA]: 目标捕获不全！小幅拓宽螺旋范围 (b=1.4) 诱导个体打卡。")

    elif stuck_counter >= 1:
        # 随卡壳轮数指数级递增 b 的大小，逐步拓宽猎食搜索的范围
        b_val = 1.0 + (0.4 * stuck_counter)
        #levy_scale = min(0.1, 0.04 + 0.02 * stuck_counter)
        actions.append(f"MICRO [WOA]: 连续卡壳 {stuck_counter} 轮！渐进拓宽对数螺旋幅度 (b={b_val:.2f})，增强突变率。")
    elif has_loops:
            b_val = 2.0
            p_threshold = 0.7        # 提高包围/全局探索概率
            a_decay_power = 0.5
            actions.append("MICRO [WOA]: 检测到航线绕圈！大幅拓宽螺旋并提高全局探索率。")
    elif needs_smooth:
        b_val = 0.35
        p_threshold = 0.35  
        actions.append("MICRO [WOA]: 路径过于曲折/抖动！收紧对数螺旋圈。")

    
    params['b'] = b_val
    params['levy_scale'] = levy_scale
    params['levy_beta'] = levy_beta
    params['p_threshold'] = p_threshold
    params['a_decay_power'] = a_decay_power

    return params, actions