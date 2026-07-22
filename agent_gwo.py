def tune(details, stuck_counter, is_failing, needs_smooth):
    """ GWO (灰狼算法) 专属内部参数调优特工 (完全体) """
    params = {}
    actions = []
    
    # 获取各种危机标志位
    has_loops = details.get('loop_penalty', 0) > 0
    fatal_collision = details.get('fatal_collision', 0) > 0
    missed_target = details.get('missed_target', 0) > 0
    alt_violation = details.get('altitude_violation', 0) > 0
    
    # 1. 第一优先级：生死与拓扑危机 (Panic & Structure Mode)
    if fatal_collision or has_loops:
        # 撞墙逃生或死结破局，需要极高的变异率和步长！
        params['mutation_rate'] = 0.6  
        params['mutation_scale'] = 2.0 
        params['levy_beta'] = 1.2      # 激发莱维飞行的长距离瞬移
        actions.append("MICRO [GWO]: 遭遇致命撞击或死结！激发 60% 变异率与 [高能莱维瞬移] 强行破局！")
        
    elif missed_target:
        # 雷达靶向牵引，需要低变异率，防止狼群乱跑不听指挥
        params['mutation_rate'] = 0.1
        params['mutation_scale'] = 0.2
        actions.append("MICRO [GWO]: 配合雷达靶向，降低变异率 (0.1)，要求狼群向目标点集中！")
        
    # 2. 第二优先级：停滞卡壳 (Stagnation)
    elif stuck_counter >= 1:
        params['stagnation_max'] = 12  # 加速头狼更迭
        params['mutation_rate'] = 0.4
        actions.append("MICRO [GWO]: 狼群陷入局部死胡同！缩短头狼任期，增加变异探索度！")
        
    # 3. 第三优先级：路线压低与精修阶段 (Refinement)
    elif alt_violation:
        # 贴地压低需要中等变异率
        params['mutation_rate'] = 0.3
        params['mutation_scale'] = 0.1
        actions.append("MICRO [GWO]: 配合贴地压低，开启中度下探变异！")
        
    elif needs_smooth:
        # 配合拉普拉斯平滑，注入镇定剂
        params['mutation_rate'] = 0.1   
        params['mutation_scale'] = 0.05 
        params['levy_beta'] = 1.5      # 恢复温和的莱维分布
        actions.append("MICRO [GWO]: 配合物理平滑，注入镇定剂 (限制狼群乱跑)，专心打磨局部细节！")
        
    return params, actions