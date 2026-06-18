import numpy as np
import matplotlib.pyplot as plt
from environment_buildup import UAVEnvironment2D
import scipy.interpolate as spl

class PathEvaluator:
    def __init__(self):
        self.env = UAVEnvironment2D()
        
        # 阶梯式惩罚权重字典
        self.penalties = {
            'fatal_collision': 1000000.0,  # 撞墙 
            'missed_target': 500000.0,     # 漏掉目标区
            'sharp_turn': 10000.0,         # 死亡急转弯
            'margin_violation': 5000.0     # 侵入安全边距
        }
        # 定义最大允许转弯角度
        self.max_turn_angle = 90.0

    #B样条——但是太过于丝滑，狭窄地区不如线段。
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
    def count_missed_targets(self, path_points):
        missed_count = 0
        for target in self.env.target_areas:
            center = target['center']
            radius = target['radius']
            visited = False
            
            # 遍历路径的每一段，看看有没有穿过目标圆
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                dist = self.point_to_segment_distance(center, p1, p2)
                if dist <= radius:
                    visited = True
                    break # 这个区域查岗通过，看下一个区域
            
            if not visited:
                missed_count += 1
                print(f"警告: 任务失败！无人机未进入巡检区 [{target['name']}]")
                
        return missed_count

    def calculate_fitness(self, path_points):
        distance = self.calculate_path_length(path_points)
        penalty = 0.0
        
        # 1. 检查物理碰撞和安全边距
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            # 物理撞墙 (边距为0)
            if self.env.is_segment_collision(p1, p2, safe_margin=0.0):
                # print(f"撞墙")
                penalty += self.penalties['fatal_collision']
                
            # 是否侵入了安全缓冲带 (边距为1.0)
            elif self.env.is_segment_collision(p1, p2, safe_margin=1.0):
                # print(f"擦边.")
                penalty += self.penalties['margin_violation']

        # 2. 检查急转弯 （累加扣分）
        for i in range(len(path_points) - 2):
            angle = self.calculate_turn_angle(path_points[i], path_points[i+1], path_points[i+2])
            if angle > self.max_turn_angle:
                # print(f"急转弯警告")
                penalty += self.penalties['sharp_turn']

        # 3. 检查漏掉的巡检区域 
        missed_count = self.count_missed_targets(path_points)
        if missed_count > 0:
            penalty += missed_count * self.penalties['missed_target']
                
        # 最终得分 = 基础距离 + 所有的惩罚总和
        return distance + penalty


# syc自测用；debug用
if __name__ == "__main__":
    evaluator = PathEvaluator()
    
    # ==========================================
    # 🛸 测试用例库 (涵盖各种经典死法和完美路线)
    # ==========================================
    
    # 1. 【完美基准线】: 不撞墙、不擦边、不急转、全打卡。
    path_perfect = np.array([
        [43.0,  3.0], [43.0, 27.0], [41.0, 29.0], [28.0, 29.0], 
        [26.0, 31.0], [26.0, 69.0], [28.0, 71.0], [76.0, 71.0], 
        [78.0, 73.0], [78.0, 90.0], [75.0, 93.0], [51.0, 94.0]
    ])

    # 2. 【莽夫直线】: 只有起点和终点。直接穿透整个校区，无视建筑物和巡检区。
    # 预期: 触发多次致命撞击 (数百万分) + 漏掉巡检区 (百万分)。
    path_lazy_straight = np.array([
        [43.0,  3.0], 
        [51.0, 94.0]
    ])

    # 3. 【擦边狂魔】: 大体方向对，但在西侧走廊贴着墙皮飞 (X=24.5)。
    # 预期: 不会触发 fatal_collision，但会疯狂触发 margin_violation (每次 5000)。
    path_margin_graze = np.array([
        [43.0,  3.0], [43.0, 27.0], [41.0, 29.0], [24.5, 29.0], 
        [24.5, 71.0], [76.0, 71.0], [78.0, 73.0], [78.0, 90.0], 
        [75.0, 93.0], [51.0, 94.0]
    ])

    # 4. 【死亡折返跑】: 碰到了巡检区，但飞行轨迹像无头苍蝇，充满 90度和180度的锐角转弯。
    # 预期: 触发多次 sharp_turn 惩罚 (每次 10000)。
    path_sharp_turns = np.array([
        [43.0,  3.0], [43.0, 40.0], 
        [22.0, 40.0], # 直角拐向西区
        [78.0, 40.0], # 180度掉头横穿
        [78.0, 70.0], # 垂直打卡东区
        [30.0, 70.0], # 再次直角折返
        [51.0, 94.0]
    ])

    # 5. 【漏打卡路线】: 飞行极其安全平稳，但忘了去西区 (West Area)，直接去了东区。
    # 预期: 路线分很低，但会吃一个 missed_target 的 50万分死刑。
    path_missed_target = np.array([
        [43.0,  3.0], [43.0, 27.0], [45.0, 29.0], [78.0, 29.0], 
        [78.0, 70.0], [78.0, 90.0], [75.0, 93.0], [51.0, 94.0]
    ])

    # 将测试用例打包
    test_cases = [
        ("完美标杆路线", path_perfect),
        ("莽夫直线 (全撞+漏打卡)", path_lazy_straight),
        ("擦边狂魔 (危险边缘试探)", path_margin_graze),
        ("死亡折返跑 (全是急转弯)", path_sharp_turns),
        ("漏打卡路线 (安全但忘做任务)", path_missed_target)
    ]

    # 批量执行测试并打印报告
    print("=" * 50)
    print("开始执行 PSO 适应度函数压力测试...")
    print("=" * 50)

    for name, path in test_cases:
        score = evaluator.calculate_fitness(path)
        print(f"【{name}】")
        # 如果分数大于 1000，说明吃到了惩罚，用红色警告色打印
        if score > 1000:
            print(f"   最终得分: \033[91m{score:,.2f}\033[0m")
        else:
            print(f"   最终得分: \033[92m{score:,.2f}\033[0m (纯纯的距离分！)")
        print("-" * 50)

    # 选一个有代表性的画出来看看 
    path_to_draw = path_margin_graze
    
    fig, ax = plt.subplots(figsize=(10, 10))
    evaluator.env.draw_environment(ax)
    ax.plot(path_to_draw[:, 0], path_to_draw[:, 1], color='#FF5722', linestyle='-', 
            linewidth=3, marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax.set_title("Test Path Visualization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()