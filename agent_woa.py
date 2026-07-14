def tune(details, stuck_counter, is_failing, needs_smooth):
    """ WOA (鲸鱼算法) 专属内部参数调优特工 """
    params = {}
    actions = []
    
    if needs_smooth:
        params['b'] = 0.2      
        actions.append("MICRO [WOA]: 配合物理平滑，收紧底层螺旋圈参数 (b=0.2)！")
    elif stuck_counter >= 1:
        params['b'] = 2.0      
        actions.append("MICRO [WOA]: 卡壳！强行放大对数螺旋参数 (b=2.0)，扩大猎食范围！")
        
    return params, actions