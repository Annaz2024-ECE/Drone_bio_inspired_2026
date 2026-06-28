import numpy as np
import matplotlib.pyplot as plt
from environment_buildup import UAVEnvironment2D
import scipy.interpolate as spl

class PathEvaluator:
    def __init__(self):
        self.env = UAVEnvironment2D()
        
        # 基础惩罚权重 (外环智能体可以直接修改这些值)
        self.penalties = {
            'fatal_collision': 1000000.0,  # 撞墙 
            'missed_target': 1000000.0,     # 漏掉目标区
            'sharp_turn': 10000.0,         # 死亡急转弯
            'margin_violation': 5000.0     # 侵入安全边距
        }

        # 算法通用调节参数 (供智能体动态调优)
        self.params = {
            'max_turn_angle': 90.0,
            'chaikin_iterations': 3,     # 狭窄地带可能需要增加平滑迭代
            'min_waypoint_dist': 5.0,    # 航点弹簧排斥距离
            'margin_layers': [0.25, 0.16, 0.12, 0.08, 0.04] # 遇台风天可由智能体动态加宽
        }
        
        self.ideal_min_distance = self._calculate_ideal_min_distance()
        print(f" [环境加载] 当前地图理论最短直线距离: {self.ideal_min_distance:.1f} 米")

    def _calculate_ideal_min_distance(self):
        """ 计算 起点 -> 所有打卡点 -> 终点 的橡皮筋直线距离 """
        pts = [self.env.start_point]
        
        # 提取所有打卡点的圆心，并按 X 坐标排序（模拟顺路飞行的逻辑）
        targets = [t['center'] for t in self.env.target_areas]
        targets.sort(key=lambda p: p[0])
        
        pts.extend(targets)
        pts.append(self.env.end_point)

        dist = 0.0
        for i in range(len(pts) - 1):
            dist += self.env.calculate_distance(pts[i], pts[i+1])
        return dist

    def update_params(self, new_penalties=None, new_params=None):
        """
        供协调决策智能体调用的统一接口：动态刷新权重和参数
        """
        if new_penalties:
            self.penalties.update(new_penalties)
        if new_params:
            self.params.update(new_params)

    #画直线段的算法
    # def apply_string_pulling(self, path_points):
    #     """
    #     拉线平滑算法：大刀阔斧地砍掉所有不必要的中间航点，强制拉直路线！
    #     """
    #     if len(path_points) <= 2:
    #         return path_points

    #     optimized_path = [path_points[0]]  # 始终保留起点
    #     current_index = 0
        
    #     # 只要还没走到终点，就一直往后看
    #     while current_index < len(path_points) - 1:
    #         furthest_visible_index = current_index + 1
            
    #         # 从终点倒着往前看，寻找能一眼看到的“最远的那个点”
    #         for next_index in range(len(path_points) - 1, current_index, -1):
    #             p1 = path_points[current_index]
    #             p2 = path_points[next_index]
                
    #             # 视线检测：用你的大边距 (1.0) 测试这条直连线会不会撞墙
    #             if not self.env.is_segment_collision(p1, p2, safe_margin=1.0):
    #                 furthest_visible_index = next_index
    #                 break  # 找到了能直达的最远点，跳出循环！
                    
    #         # 把找到的这个最远点加入最终路线
    #         optimized_path.append(path_points[furthest_visible_index])
            
    #         # 瞬移到这个最远点，继续往后看
    #         current_index = furthest_visible_index
            
    #     return np.array(optimized_path)


    # B样条——但是太过于丝滑，狭窄地区不如线段。
    # def generate_bspline_path(self, waypoints, num_points=100):
    #     """
    #     将稀疏的关键点(waypoints)转换为密集的平滑曲线点
    #     """
    #     waypoints = np.array(waypoints)
        
    #     # 1. 剔除极其接近的重复点，防止插值算法除以0报错
    #     unique_waypoints = [waypoints[0]]
    #     for pt in waypoints[1:]:
    #         if np.linalg.norm(pt - unique_waypoints[-1]) > 0.1:
    #             unique_waypoints.append(pt)
    #     unique_waypoints = np.array(unique_waypoints)

    #     # 2. 检查点数。B样条默认需要至少 4 个点 (阶数 k=3)
    #     num_wp = len(unique_waypoints)
    #     if num_wp < 3:
    #         return unique_waypoints # 点太少，画不出平滑曲线，直接返回原线
            
    #     k = 3 if num_wp >= 4 else num_wp - 1

    #     # 3. 提取 X 和 Y
    #     x = unique_waypoints[:, 0]
    #     y = unique_waypoints[:, 1]

    #     # 4. 计算 B样条参数 
    #     # s=0 表示强制曲线精确穿过你给定的每一个控制点
    #     tck, u = spl.splprep([x, y], s=0, k=k)

    #     # 5. 生成 100 个均匀分布的新参数，并计算出 100 个平滑点
    #     u_new = np.linspace(0, 1.0, num_points)
    #     x_new, y_new = spl.splev(u_new, tck)

    #     # 6. 把 X 和 Y 重新拼成 [[x1,y1], [x2,y2]...] 的格式
    #     smooth_path = np.column_stack((x_new, y_new))
    #     return smooth_path

    def generate_chaikin_path(self, waypoints, iterations=4):
        """
        Chaikin 割角算法：专为狭窄走廊设计的局部平滑算法。
        只切内角绝不向外膨胀甩尾, 100% 继承原折线的安全性！
        把每个转弯角，变的圆润，减少转弯角带来的惩罚
        """
        pts = np.array(waypoints)
        # 迭代次数越多，拐角处越圆滑
        for _ in range(iterations):
            new_pts = [pts[0]] # 始终保留起点
            for i in range(len(pts) - 1):
                p0 = pts[i]
                p1 = pts[i+1]
                # 在线段的 25% 和 75% 处打两个新点，把尖角“切掉”
                Q = 0.75 * p0 + 0.25 * p1
                R = 0.25 * p0 + 0.75 * p1
                new_pts.extend([Q, R])
            new_pts.append(pts[-1]) # 始终保留终点
            pts = np.array(new_pts)
        return pts

    # 计算原始航点的距离，推荐离得远一点，不要太近。
    def calculate_spacing_penalty(self, raw_waypoints):
        penalty = 0.0
        # 遍历所有原始点，算两两之间的距离
        min_dist = self.params.get('min_waypoint_dist', 5.0)
        penalty = 0.0
        for i in range(len(raw_waypoints) - 1):
            dist = np.linalg.norm(raw_waypoints[i+1] - raw_waypoints[i])
            if dist < min_dist:
                penalty += ((min_dist - dist) ** 2) * 5.0
        return penalty

    def calculate_path_length(self, path_points):
        """ 计算路径总长度 """
        total_length = 0.0
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            total_length += self.env.calculate_distance(p1, p2)
        return total_length

    def calculate_turn_angle(self, p1, p2, p3):
        """ 计算转弯角度 """
        v1 = p2 - p1
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    # 计算一个点到一条线段的最短距离
    def point_to_segment_distance(self, point, seg_a, seg_b):
        line_vec = seg_b - seg_a
        point_vec = point - seg_a
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.linalg.norm(point - seg_a)
        
        t = max(0.0, min(1.0, np.dot(point_vec, line_vec) / line_len_sq))
        projection = seg_a + t * line_vec
        return np.linalg.norm(point - projection)

    # 检查路径漏掉了几个目标区域
    def calculate_target_penalty(self, path_points):
        total_target_penalty = 0.0
        
        for target in self.env.target_areas:
            center = target['center']
            radius = target['radius']
            min_dist_to_target = float('inf')
            
            # 找到整条航线离这个目标区圆心最近的距离
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                dist = self.point_to_segment_distance(center, p1, p2)
                if dist < min_dist_to_target:
                    min_dist_to_target = dist
            
           # 1. 如果还在圈外：启动“核武级”惩罚！
            if min_dist_to_target > radius:
                missed_distance = min_dist_to_target - radius
                # 基础惩罚从 50万 提升到 100万
                # 距离偏离梯度的惩罚从 1万 提升到 5万/米！
                total_target_penalty += 1000000.0 + (missed_distance * 50000.0)
                
        return total_target_penalty

    def calculate_fitness(self, path_points):
        # 初始化一个“体检报告单” (字典)，把所有可能扣分的项目都列出来
        details = {
            'distance': 0.0,          # 路径基础长度
            'fatal_collision': 0.0,   # 物理撞墙扣分
            'margin_violation': 0.0,  # 侵入安全区扣分
            'smoothness': 0.0,        # 连续小弯折扣分
            'sharp_turn': 0.0,        # 死亡急转弯扣分
            'missed_target': 0.0      # 漏打卡扣分
        }

        # 基础距离
        raw_distance = self.calculate_path_length(path_points)
        distance_weight = 1.0  # 权重：1米 = 1分
        details['distance'] = raw_distance * distance_weight
        # details['ideal_distance'] = self.ideal_min_distance
        
        # 动态读取参数
        # 洋葱皮线性惩罚模型：把 0.25 的安全区切成 5 层
        # 距离越近，穿透的层数越多，累计惩罚越大
        #台风天改长一点
        margin_layers = self.params.get('margin_layers', [0.25, 0.16, 0.12, 0.08, 0.04])

        # 每穿透一层，扣除总软惩罚的 1/5 (即 1000 分)
        layer_penalty = self.penalties.get('margin_violation', 5000.0) / len(margin_layers)
        fatal_penalty = self.penalties.get('fatal_collision', 1000000.0)

        # 1. 【线性升级】检查物理碰撞和安全边距
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            # 先查最严重的：真实物理撞墙 (边距为0)
            if self.env.is_segment_collision(p1, p2, safe_margin=0.0):
                details['fatal_collision'] += fatal_penalty
                continue # 已经撞毁了，没必要算外围的软边距了
                

            # 先查最大的 1.0 圈
            if not self.env.is_segment_collision(p1, p2, safe_margin=1.0):
                continue# 如果最外围都是安全的，直接跳过当前线段

            for m in margin_layers:
                if self.env.is_segment_collision(p1, p2, safe_margin=m):
                    details['margin_violation'] += layer_penalty


        # 动态读取角度参数
        max_turn = self.params.get('max_turn_angle', 90.0)
        sharp_turn_pen = self.penalties.get('sharp_turn', 10000.0)

        # 2. 急转弯与平滑度连续惩罚
        for i in range(len(path_points) - 2):
            angle = self.calculate_turn_angle(path_points[i], path_points[i+1], path_points[i+2])
            
            # 设定死区：小于 5 度的微小弯（B样条的平滑点）不扣分
            if angle > 5.0:
                # 非线性惩罚：角度越大，惩罚成指数级暴增！
                # 减去 5 是为了让惩罚在突破死区时是平滑过渡的
                details['smoothness'] += ((angle - 5.0) ** 2) * 0.5
                
            # 依然保留死亡急转弯的底线（比如超过 90 度直接判死刑）
            if angle > max_turn:
                details['sharp_turn'] += sharp_turn_pen

        # 3. 调用引力梯度目标惩罚
        details['missed_target'] += self.calculate_target_penalty(path_points)

        # 专门的情报字典
        env_info = {
            'ideal_distance': self.ideal_min_distance,
            'obstacle_count': len(self.env.obstacles)
        }
                
        #算总分
        total_score = sum(details.values())
        
        # 返回 3 个值：总分、扣分账本、情报字典
        return total_score, details, env_info

    def evaluate_pso_particle(self, raw_waypoints):
        """
        最后结果输出 (现返回：总分, 记账明细)
        """
        # 1. 算排斥力
        spacing_penalty = self.calculate_spacing_penalty(raw_waypoints)
        
        # 2. 动态读取平滑迭代次数
        iters = self.params.get('chaikin_iterations', 3)
        smooth_path = self.generate_chaikin_path(raw_waypoints, iterations=iters)
        
        # 3. 算分与明细
        base_score, details, env_info = self.calculate_fitness(smooth_path)
        
        # 将排斥力加入账本
        details['spacing_penalty'] = spacing_penalty
        
        # 重新核算最终总分 (因为加入了 spacing_penalty)
        total_score = sum(details.values())
        
        return total_score, details, env_info

# syc自测用；debug用
if __name__ == "__main__":
    evaluator = PathEvaluator()
    
    # 保持你原来的测试用例不变
    path_bspline_perfect = np.array([[43.0,  3.0], [43.0, 25.0], [43.0, 28.0], [41.0, 29.0], [29.0, 29.0], [27.0, 31.0], [27.0, 35.0], [27.0, 65.0], [27.0, 69.0], [29.0, 71.0], [71.0, 71.0], [76.0, 71.0], [78.0, 73.0], [78.0, 88.0], [78.0, 92.0], [75.0, 94.0], [51.0, 94.0]])
    path_lazy_straight = np.array([[43.0,  3.0], [51.0, 94.0]])
    path_margin_graze = np.array([[43.0,  3.0], [43.0, 27.0], [41.0, 29.0], [24.5, 29.0], [24.5, 71.0], [76.0, 71.0], [78.0, 73.0], [78.0, 90.0], [75.0, 93.0], [51.0, 94.0]])
    path_sharp_turns = np.array([[43.0,  3.0], [43.0, 40.0], [22.0, 40.0], [78.0, 40.0], [78.0, 70.0], [30.0, 70.0], [51.0, 94.0]])
    path_missed_target = np.array([[43.0,  3.0], [43.0, 27.0], [45.0, 29.0], [78.0, 29.0], [78.0, 70.0], [78.0, 90.0], [75.0, 93.0], [51.0, 94.0]])

    test_cases = [
        ("完美标杆路线", path_bspline_perfect),
        ("莽夫直线 (全撞+漏打卡)", path_lazy_straight),
        ("擦边狂魔 (危险边缘试探)", path_margin_graze),
        ("死亡折返跑 (全是急转弯)", path_sharp_turns),
        ("漏打卡路线 (安全但忘做任务)", path_missed_target)
    ]

    print("=" * 50)
    print("开始执行 终极黑盒API 压力测试...")
    print("=" * 50)

    for name, raw_path in test_cases:
        # 【修改点】：同时接收 score 和 details
        score, details, env_info = evaluator.evaluate_pso_particle(raw_path)
        
        print(f"【{name}】")
        if score > 5000:
            print(f"   最终得分: \033[91m{score:,.2f}\033[0m")
        else:
            print(f"   最终得分: \033[92m{score:,.2f}\033[0m (极致丝滑安全的完美分数！)")
            
        # 【新增：打印智能体最需要的体检明细】
        print("   >>> 状态明细(State):")
        for k, v in details.items():
            if v > 0: # 只打印扣分的项目，看起来更清爽
                color = "\033[91m" if v > 1000 else "\033[0m" # 严重扣分标红
                print(f"       - {k}: {color}{v:,.2f}\033[0m")
        print("-" * 50)

    # 画图验证 (对比原始控制点和 Chaikin 平滑曲线)
    path_to_draw = path_bspline_perfect 
    smooth_path_to_draw = evaluator.generate_chaikin_path(path_to_draw, iterations=4)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    evaluator.env.draw_environment(ax)
    
    ax.plot(path_to_draw[:, 0], path_to_draw[:, 1], color='gray', linestyle='--', 
            linewidth=1.5, marker='x', markersize=8, label='PSO Control Waypoints')
    ax.plot(smooth_path_to_draw[:, 0], smooth_path_to_draw[:, 1], color='#FF007F', linestyle='-', 
            linewidth=3, label='Chaikin Flight Path')
            
    ax.set_title("Chaikin Corner-Cutting Path Visualization", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()