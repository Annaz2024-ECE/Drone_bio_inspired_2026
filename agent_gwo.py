def tune(details, stuck_counter, is_failing, needs_smooth):
    """ GWO (灰狼算法) 专属内部参数调优特工 """
    params = {}
    actions = []
    
    if stuck_counter >= 1:
        params['stagnation_max'] = 12  
        actions.append("MICRO [GWO]: 卡壳！修改内部停滞阈值 stagnation_max=12，加速头狼更新交替！")
        
    if needs_smooth:
        params['mutation_rate'] = 0.1   
        params['mutation_scale'] = 0.05 
        actions.append("MICRO [GWO]: 配合物理平滑，限制狼群乱跑 (mutation_rate=0.1)，专心打磨局部细节！")
        
    return params, actions