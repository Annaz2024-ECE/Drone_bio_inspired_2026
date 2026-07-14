def tune(details, stuck_counter, is_failing, needs_smooth):
    """ GA (遗传算法) 专属内部参数调优特工 """
    params = {}
    actions = []
    
    if needs_smooth:
        params['pm'] = 0.05    # 降低突变，防止好不容易平滑的线被切断
        params['pc'] = 0.95    # 极高交叉率，保留优良几何特征
        actions.append("MICRO [GA]: 配合物理平滑，修改底层变异率 pm=0.05, 交叉率 pc=0.95 保留优良基因")
    elif stuck_counter >= 1:
        params['pm'] = 0.6     # 基因大爆炸
        actions.append("MICRO [GA]: 卡壳！强制触发基因大爆炸 (pm=0.6)，跳出局部最优陷阱！")
        
    return params, actions