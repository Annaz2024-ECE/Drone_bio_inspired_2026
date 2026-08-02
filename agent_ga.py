def tune(details, stuck_counter, is_failing, needs_smooth):
    """ GA (遗传算法) 专属内部参数调优特工 """
    params = {}
    actions = []
    # 获取具体的物理惩罚状态
    fatal_col = details.get('fatal_collision', 0.0) > 0
    missed_tgt = (details.get('missed_target', 0.0) + details.get('missed_target_base', 0.0)) > 0
    has_loops = details.get('loop_penalty', 0.0) > 0 or details.get('shattering_kick', False)

    if is_failing:
        if fatal_col:
            params['pm'], params['pc'] = 0.30, 0.75
            params['emergency_escape'] = True
            actions.append("MICRO [GA]: 致命碰撞！提变异(pm=0.35)激活莱维大范围跳跃逃生。")
        elif missed_tgt:
            params['pm'], params['pc'] = 0.20, 0.80
            actions.append("MICRO [GA]: 漏打卡！适度变异(pm=0.20)配合全局雷达空投寻靶。")
    
    elif stuck_counter >= 1:
        target_pm = min(0.15 + stuck_counter * 0.1, 0.6)
        params['pm'], params['pc'] = target_pm, 0.70
        params['emergency_escape'] = True
        actions.append(f"MICRO [GA]: 演化卡壳(x{stuck_counter})！提升变异 pm={target_pm:.2f} 并开启莱维突变。")
    elif has_loops:
        params['pm'], params['pc'] = 0.40, 0.60
        params['emergency_escape'] = True
        actions.append("MICRO [GA]: 航线绕圈！高变异(pm=0.40)强行打碎死结基因。")
    elif needs_smooth:
        params['pm'], params['pc'] = 0.10, 0.95
        actions.append("MICRO [GA]: 航线安全！降变异(pm=0.10)升交叉(pc=0.95)配合物理雕刻。")
    return params, actions