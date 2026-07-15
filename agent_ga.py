def tune(details, stuck_counter, is_failing, needs_smooth):
    """ GA (遗传算法) 专属内部参数调优特工 """
    params = {}
    actions = []
    # 获取具体的物理惩罚状态
    fatal_collision = details.get('fatal_collision', 0.0)
    missed_target = details.get('missed_target_base', 0.0) + details.get('missed_target_factor', 0.0)
    sharp_turn = details.get('sharp_turn', 0.0)
    margin_violation = details.get('margin_violation', 0.0)
    
    # 初始化默认的物理开关 (防止底层 Planner 找不到属性)
    params['apply_laplacian'] = False
    params['apply_repulsion'] = False
    # ==========================================================
    # 场景 1：航线基本安全，需要极致的几何平滑度与业务性能打磨
    # ==========================================================
    if needs_smooth and fatal_collision == 0:
        params['pm'] = 0.02    # 压低变异率，进入“静默精修”状态
        params['pc'] = 0.95    # 极高交叉率，通过算术线性插值融合出更顺滑的中间曲线
        params['apply_laplacian'] = True  # 开启拉普拉斯物理平滑，强行拉直折线
        actions.append("MICRO [GA]: 航线安全！开启拉普拉斯物理拉直，降变异(pm=0.02)升交叉(pc=0.95)进行静默雕刻。")

    # ==========================================================
    # 场景 2：发生严重卡壳（陷入局部最优/死胡同）
    # ==========================================================
    elif stuck_counter >= 1:
        # 渐进式基因大爆炸：随着卡壳代数增加，逐步增大变异率
        # 不再采用一刀切的 0.6，防止优秀骨架瞬间稀碎
        target_pm = min(0.1 + stuck_counter * 0.1, 0.6) 
        params['pm'] = target_pm
        params['pc'] = 0.7  # 略微调低交叉率，给突变留出探索空间
        
        actions.append(f"MICRO [GA]: 诊断为局部卡壳(x{stuck_counter})！渐进提升变异率至 pm={target_pm:.2f} 触发突变破局。")
        
        # 如果卡壳时伴随着撞楼，开启大楼排斥力场，用外力强行把航线推离危险区
        if fatal_collision > 0:
            params['apply_repulsion'] = True
            actions.append("MICRO [GA-PHYSICS]: 检测到卡壳且撞楼，紧急开启【大楼斥力势场】物理推搡！")

    # ==========================================================
    # 场景 3：严重物理病态（大范围撞楼）
    # ==========================================================
    elif fatal_collision > 0:
        # 撞楼时，我们需要莱维飞行产生大范围跳跃，所以调高变异率
        params['pm'] = 0.35
        params['pc'] = 0.80
        params['apply_repulsion'] = True # 开启斥力场
        actions.append("MICRO [GA]: 检测到严重碰撞！上调变异率 pm=0.35 激活莱维大范围空间传送，并开启物理斥力场。")

    # ==========================================================
    # 场景 4：业务病态（漏掉巡检点）
    # ==========================================================
    elif missed_target > 0:
        # 漏靶时，说明航线因为过度变异“脱轨”了。
        # 我们应当调低变异率，强化巡检骨架引导，并且关闭拉普拉斯平滑（防止平滑把踩点航段切掉）
        params['pm'] = 0.05
        params['pc'] = 0.90
        params['apply_laplacian'] = False 
        actions.append("MICRO [GA]: 发生漏打卡！降低变异率 pm=0.05 保护踩点基因，关闭物理拉直防止切角漏点。")

    # ==========================================================
    # 场景 5：飞行动力学不达标（急转弯太多）
    # ==========================================================
    elif sharp_turn > 0 or margin_violation > 0:
        params['apply_laplacian'] = True
        params['pm'] = 0.08
        actions.append("MICRO [GA-PHYSICS]: 转弯角度过载！开启拉普拉斯航线去褶皱物理修正。")

    return params, actions