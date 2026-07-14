def tune(details, stuck_counter, is_failing, needs_smooth):
    """ SSA (麻雀算法) 专属内部参数调优特工 """
    params = {}
    actions = []
    
    if needs_smooth:
        params['ST'] = 0.95    
        actions.append("MICRO [SSA]: 配合物理平滑，强行拉高安全感 (ST=0.95)，阻止麻雀惊飞乱跳！")
    elif stuck_counter >= 1:
        params['ST'] = 0.4     
        actions.append("MICRO [SSA]: 卡壳！压低安全阈值 (ST=0.4) 制造恐慌，打散麻雀群！")
        
    return params, actions