def tune(details, stuck_counter, is_failing, needs_smooth):
    """
    WOA (鲸鱼算法) 专属深度自适应调优特工
    
    输入参数：
    - details: 包含评价细项的字典，如 'fatal_collision', 'sharp_turn', 'missed_target' 等
    - stuck_counter: 连续没有找到更好解的卡壳轮数
    - is_failing: 当前解是否极差（如发生大面积撞楼、严重漏点等）
    - needs_smooth: 外部宏观特工是否要求强制物理平滑
    """
    params = {}
    actions = []

    # ==========================================
    # 1. 核心参数默认初始化
    # ==========================================
    b_val = 1.0              # 默认对数螺旋形状参数
    levy_scale = 0.05        # 默认莱维大跳跃步长比例
    levy_beta = 1.5
    p_threshold = 0.5        # 默认包围与螺旋运动的分界线
    a_decay_power = 1.0      # 默认收敛因子衰减指数

    # 获取具体的惩罚项数值（防 Key 不存在）
    fatal_col = details.get('fatal_collision', 0.0)
    sharp_turn = details.get('sharp_turn', 0.0)
    missed_tgt = details.get('missed_target', 0.0)
    power_pen = details.get('change_power_pen', 0.0)

    # ==========================================
    # 2. 靶向诊疗逻辑 (Targeted Therapy)
    # ==========================================
    
    # 【病状 A】：撞墙严重 (Fatal Collision) 或处于绝境故障 (is_failing)
    if fatal_col > 100000.0 or is_failing:
        # 1. 激发大跨度莱维飞行，跳出死胡同
        levy_scale = 0.12
        # 2. 偏向于进行“全局探索和包围”，限制螺旋运动概率
        p_threshold = 0.65  
        # 3. 延缓收敛因子的衰减，给算法更多的时间进行大范围探索
        a_decay_power = 0.7  
        actions.append("MICRO [WOA]: 发现严重碰撞/环境受阻！启动大步长莱维扰动 (levy_scale=0.12)，延缓收敛。")

    # 【病状 B】：卡壳不收敛 (Stuck)
    elif stuck_counter >= 1:
        # 随卡壳轮数指数级递增 b 的大小，逐步拓宽猎食搜索的范围
        b_val = 1.0 + (0.4 * stuck_counter)
        levy_scale = min(0.1, 0.04 + 0.02 * stuck_counter)
        actions.append(f"MICRO [WOA]: 连续卡壳 {stuck_counter} 轮！渐进拓宽对数螺旋幅度 (b={b_val:.2f})，增强突变率。")

    # 【病状 C】：路径严重抖动，需要强制平滑 (needs_smooth 或转弯与变化功率惩罚过高)
    elif needs_smooth or sharp_turn > 20000.0 or power_pen > 5000.0:
        # 1. 收敛对数螺旋，迫使局部寻优向直线或圆滑弧线靠拢
        b_val = 0.25
        # 2. 降低大跳跃频率，避免由于莱维大跨度造成折线和锐角
        levy_scale = 0.01  
        # 3. 增加螺旋搜索的占比，依靠连续弧线过渡
        p_threshold = 0.35  
        actions.append("MICRO [WOA]: 路径过于曲折/抖动！收紧对数螺旋圈 (b=0.25)，弱化突变步长以追求平滑。")

    # 【病状 D】：漏打卡严重 (Missed Target)
    elif missed_tgt > 100000.0:
        # 略微膨胀螺旋探测范围，增大个体贴近并捕获航点目标的期望概率
        b_val = 1.4
        p_threshold = 0.4
        actions.append("MICRO [WOA]: 巡检目标捕获不全！小幅拓宽螺旋猎食范围 (b=1.4) 诱导个体打卡。")

    # ==========================================
    # 3. 参数输出打包
    # ==========================================
    params['b'] = b_val
    params['levy_scale'] = levy_scale
    params['levy_beta'] = levy_beta
    params['p_threshold'] = p_threshold
    params['a_decay_power'] = a_decay_power

    return params, actions