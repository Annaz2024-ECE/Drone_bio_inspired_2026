def tune(details, stuck_counter, is_failing, needs_smooth):
    """ PSO (粒子群算法) 专属内部参数调优特工 """
    params = {}
    actions = []
    
    if needs_smooth:
        params['w_max'] = 0.4  
        params['c1'] = 2.0     
        params['c2'] = 0.5   
        actions.append("MICRO [PSO]: 配合物理平滑，压低粒子惯性 (w_max=0.4, c2=0.5)，限制其盲目冲刺！")
    elif stuck_counter >= 1:
        params['w_max'] = 1.2  
        actions.append("MICRO [PSO]: 卡壳！直接篡改底层 (w_max=1.2)，强制惯性超载冲出瓶颈！")
        
    return params, actions