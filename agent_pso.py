def tune(details, stuck_counter, is_failing, needs_smooth):
    """ PSO (粒子群算法) 专属内部参数调优特工 """
    params = {}
    actions = []

    # 提取物理惩罚状态（与 GA 保持一致）
    fatal_col = details.get('fatal_collision', 0.0) > 0
    missed_tgt = (details.get('missed_target', 0.0) + details.get('missed_target_base', 0.0)) > 0
    has_loops = details.get('loop_penalty', 0.0) > 0 or details.get('shattering_kick', False)

    # ---------- 优先级 1：整体失败 ----------
    if is_failing:
        if fatal_col:
            # 致命碰撞 → 强制大规模跳跃 + 高探索惯性
            params['w_max'] = 1.2
            params['c1'] = 2.5          # 增强个体认知，鼓励脱离局部
            params['c2'] = 1.5
            params['disturb_ratio'] = 0.35
            params['emergency_escape'] = True
            actions.append("MICRO [PSO]: 致命碰撞！惯性超载 (w_max=1.2) 并激活紧急莱维逃生 (disturb_ratio=0.35)。")
        elif missed_tgt:
            # 漏打卡 → 雷达引导 + 社会学习强化
            params['w_max'] = 0.7
            params['c1'] = 1.0
            params['c2'] = 2.5          # 强社会性，快速向历史最优靠拢
            params['radar_guidance'] = True
            actions.append("MICRO [PSO]: 漏打卡！启动雷达空投 (radar_guidance) 并增强社会学习 (c2=2.8) 全局寻靶。")

    # ---------- 优先级 2：演化卡壳 ----------
    elif stuck_counter >= 1:
        # 动态提升扰动比例和惯性，并开启紧急逃生
        target_disturb = min(0.2 + stuck_counter * 0.1, 0.6)
        params['w_max'] = 1.2
        params['w_min'] = 0.6           # 调高最小惯性，保持持续探索
        params['c1'] = 2.0
        params['c2'] = 0.8              # 降低社会性，鼓励个体自主突围
        params['disturb_ratio'] = target_disturb
        params['emergency_escape'] = True
        actions.append(f"MICRO [PSO]: 演化卡壳 (x{stuck_counter})！提升扰动至 {target_disturb:.2f}，惯性 (w_max=1.2) 并开启莱维跳跃。")

    # ---------- 优先级 3：航线绕圈 ----------
    elif has_loops:
        # 绕圈死循环 → 打碎路径结构，增加社会随机性
        params['w_max'] = 0.9
        params['c1'] = 0.8
        params['c2'] = 2.5
        params['disturb_ratio'] = 0.4
        params['shattering_kick'] = True
        actions.append("MICRO [PSO]: 航线绕圈！高扰动 (disturb_ratio=0.4) 配合碎环击 (shattering_kick) 强行重置粒子。")

    # ---------- 优先级 4：需要平滑 ----------
    elif needs_smooth:
        # 安全平滑阶段 → 低惯性、强个体认知，限制盲目冲刺
        params['w_max'] = 0.4
        params['w_min'] = 0.2
        params['c1'] = 2.0
        params['c2'] = 0.5
        actions.append("MICRO [PSO]: 航线安全！压低惯性 (w_max=0.4, w_min=0.2) 并降低社会学习 (c2=0.5)，配合物理雕刻。")

    return params, actions