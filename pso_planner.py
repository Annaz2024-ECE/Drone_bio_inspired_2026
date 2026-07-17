import numpy as np
from base_planner import BasePlanner
import os

class PSOPlanner(BasePlanner):
    def __init__(self, evaluator=None, num_particles=100, max_iter=200, num_waypoints=30, disturb_ratio=0.15):
        """
        3D PSO 路径规划器 (已移植 SSA 的多目标锚固与高度约束技术)
        """
        # 【修复1】强制提升控制点数量以应对 3D 复杂度和双点锚固
       # num_waypoints = max(num_waypoints, 40)
        super().__init__(num_waypoints=num_waypoints, max_iter=max_iter, evaluator=evaluator)
        
        self.num_particles = num_particles
        self.disturb_ratio = disturb_ratio
        
        # PSO 核心参数
        self.w_max = 0.9  
        self.w_min = 0.4  
        self.c1 = 1.5 
        self.c2 = 1.5 
        self.v_max = 8.0 
        
        # 【新增】Z轴维度面具与速度限制
        self.z_mask = np.tile([1.0, 1.0, 0.2], self.num_waypoints)
        self.v_max_arr = self.v_max * self.z_mask  # 限制垂直最大速度，防止上下暴走
        
        # 初始化粒子位置和速度
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((self.num_particles, self.dim))
        
        # 记录个体最优和全局最优
        self.pbest_pos = np.copy(self.particles)
        self.pbest_scores = np.full(self.num_particles, np.inf)
        
        # 【核心修改】：废除原有的 gbest，使用 historical_best 完美对接老中医，实现跨轮次记忆！
        self.historical_best_pos = np.zeros(self.dim)
        self.historical_best_score = np.inf
    
    # def _decode_path(self, position):
    #     """
    #     [子类重写] 植入贪心最近邻排序 (Nearest Neighbor Sort)
    #     在每次评价前，强制理顺被打乱的麻雀基因，彻底消除 3D 航线打结与绕圈现象。
    #     """
    #     # 1. 将 1D 基因还原为 3D 坐标点阵
    #     waypoints = position.reshape((self.num_waypoints, 3))
        
    #     # 2. 贪心最近邻排序核心逻辑
    #     sorted_waypoints = []
    #     current_point = self.env.start_point 
    #     remaining_indices = list(range(self.num_waypoints))
        
    #     while remaining_indices:
    #         best_idx = -1
    #         min_dist = float('inf')
            
    #         # 遍历所有还没被连线的点，找离当前位置最近的
    #         for idx in remaining_indices:
    #             dist = np.linalg.norm(waypoints[idx] - current_point)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 best_idx = idx
                    
    #         # 把找到的最近点加入有序列表，并将“当前位置”推进到该点
    #         sorted_waypoints.append(waypoints[best_idx])
    #         current_point = waypoints[best_idx]
    #         remaining_indices.remove(best_idx)
            
    #     # 3. 拼接起终点返回
    #     sorted_waypoints = np.array(sorted_waypoints)
    #     full_path = np.vstack([self.env.start_point, sorted_waypoints, self.env.end_point])
    #     return full_path

    def _initialize_particles(self):
        """ 
        生成 3D 初始粒子群，直接调用基类的“TSP+三点锚固+拱门飞跃”超级骨架。
        """
        particles = np.zeros((self.num_particles, self.dim))
        
        # ==========================================
        # 1. 直接向父类索要“无敌拓扑骨架”
        # ==========================================
        super_skeleton = self._generate_heuristic_skeleton()
        
        # 2. 第 0 号粒子直接封神，拿走完美骨架，保底不撞楼且不漏打卡
        particles[0] = super_skeleton
        
        # 3. 剩下的粒子基于无敌骨架，配合 z_mask 限制垂直扰动，进行不同梯度的变异探索
        for i in range(1, self.num_particles):
            if i < int(self.num_particles * 0.4):
                noise = np.random.normal(0, 2.0, self.dim) * self.z_mask
            elif i < int(self.num_particles * 0.8):
                noise = np.random.normal(0, 6.0, self.dim) * self.z_mask
            else:
                noise = np.random.normal(0, 12.0, self.dim) * self.z_mask
                
            particles[i] = np.clip(super_skeleton + noise, self.lb, self.ub)
        
        return particles

    def optimize(self):
        print("开始 3D PSO 粒子群算法路径规划 (带多目标锚固、限速与历史最优保护)...")
        
        # 1. 初始评估 (适应调参大脑的多轮重入，强制刷新 pbest 的真实得分)
        for i in range(self.num_particles):
            # 如果不是第一轮，老中医可能修改了规则，过去的 pbest_score 已作废，必须按新规则重算！
            if self.pbest_scores[i] != np.inf:
                true_pbest_score, _, _ = self.evaluator.evaluate_particle(self._decode_path(self.pbest_pos[i]))
                self.pbest_scores[i] = true_pbest_score
                
            full_path = self._decode_path(self.particles[i])
            score, _, _ = self.evaluator.evaluate_particle(full_path)
            
            if score < self.pbest_scores[i]:
                self.pbest_scores[i] = score
                self.pbest_pos[i] = np.copy(self.particles[i])
                
            # 记录历史最优
            if score < self.historical_best_score:
                self.historical_best_score = score
                self.historical_best_pos = np.copy(self.particles[i])

        # 2.
        for iteration in range(self.max_iter):
            w_current = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            # ==========================================
            # 第一阶段：纯粹的物理位移 (只算公式，不打分)
            # ==========================================
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                social = self.c2 * r2 * (self.historical_best_pos - self.particles[i])
                
                self.velocities[i] = w_current * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max_arr, self.v_max_arr)
                
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

            # ==========================================
            # >>> 【完美缝隙】：触发全局拉普拉斯平滑 <<<
            # 桥接黑科技：把 self.particles 临时伪装成 self.positions 借给基类用
            self.positions = self.particles
            
            # 呼叫基类的物理引擎进行平滑
            self.execute_universal_physics_directives()
            
            # 把平滑后的结果拿回来
            self.particles = self.positions
            
            # 烧毁临时护照
            del self.positions
            # ==========================================

            # ==========================================
            # 第二阶段：纯粹的成绩评估与记录
            # ==========================================
            for i in range(self.num_particles):
                full_path = self._decode_path(self.particles[i])
                score, _, _ = self.evaluator.evaluate_particle(full_path)
                
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_pos[i] = np.copy(self.particles[i])
                if score < self.historical_best_score:
                    self.historical_best_score = score
                    self.historical_best_pos = np.copy(self.particles[i])

                    
            # ==========================================
            # 接收老中医的四大通用物理指令 (Universal API)
            # ==========================================
            is_radar = getattr(self, 'radar_guidance', False)
            is_emergency = getattr(self, 'emergency_escape', False)
            is_lift_up = getattr(self, 'lift_up', False)
            is_press_down = getattr(self, 'press_down', False)

            # 只要触发了任何一个全局动作，就启动底层基因干预
            if is_radar or is_emergency or is_lift_up or is_press_down:
                
                mutation_rate = 0.1
                if is_emergency: mutation_rate = 0.5
                elif is_lift_up: mutation_rate = 0.4
                elif is_press_down: mutation_rate = 0.3
                
                do_mutation = np.random.rand(self.num_particles) < mutation_rate
                
                # 绝对保护：保留历史最优粒子，留下革命火种
                best_idx = np.argmin(self.pbest_scores)
                do_mutation[best_idx] = False 
                
                # 1. 雷达空投逻辑
                if is_radar:
                    targets_3d = []
                    for t in self.env.target_areas:
                        z_mid = (t.get('z_min', 0.0) + t.get('z_max', 10.0)) / 2.0
                        targets_3d.append(np.array([t['center'][0], t['center'][1], z_mid]))
                    
                    for i in range(self.num_particles):
                        if do_mutation[i]:
                            new_pos = np.zeros((self.num_waypoints, 3))
                            if len(targets_3d) > 0:
                                for j in range(self.num_waypoints):
                                    new_pos[j] = targets_3d[j % len(targets_3d)]
                            
                            noise = np.random.randn(self.num_waypoints, 3) * 2.0
                            self.particles[i] = np.clip((new_pos + noise).flatten(), self.lb, self.ub)
                            
                            # 【核心斩断橡皮筋】：清零速度并强制洗脑 pbest
                            self.velocities[i] = 0.0 
                            full_path = self._decode_path(self.particles[i])
                            score, _, _ = self.evaluator.evaluate_particle(full_path)
                            self.pbest_scores[i] = score
                            self.pbest_pos[i] = np.copy(self.particles[i])
                
                # 2. 物理推力逻辑 (Z轴强制位移)
                else:
                    noise = np.zeros((self.num_particles, self.dim))
                    if is_emergency:
                        for d in range(2, self.dim, 3): noise[:, d] = 15.0 
                    elif is_lift_up:
                        for d in range(2, self.dim, 3): noise[:, d] = 8.0  
                    elif is_press_down:
                        for d in range(2, self.dim, 3): noise[:, d] = -3.0 
                    
                    for i in range(self.num_particles):
                        if do_mutation[i]:
                            self.particles[i] += noise[i]
                            self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)
                            
                            # 【核心斩断橡皮筋】：清零速度并强制洗脑 pbest
                            self.velocities[i] = 0.0 
                            full_path = self._decode_path(self.particles[i])
                            score, _, _ = self.evaluator.evaluate_particle(full_path)
                            self.pbest_scores[i] = score
                            self.pbest_pos[i] = np.copy(self.particles[i])

                # 3. 动作结束后，二次检查是否诞生了新的全局最优
                for i in range(self.num_particles):
                    if self.pbest_scores[i] < self.historical_best_score:
                        self.historical_best_score = self.pbest_scores[i]
                        self.historical_best_pos = np.copy(self.pbest_pos[i])

                # ==========================================
                # 【极其关键】：阅后即焚！
                # 执行完一次老中医的“冲量”干预后，必须立刻销毁指令，
                # 否则后续代数粒子会被无限次拔高飞入太空！
                # ==========================================
                self.radar_guidance = False
                self.emergency_escape = False
                self.lift_up = False
                self.press_down = False
            # ==========================================

            self.convergence_curve.append(self.historical_best_score)
            
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"  > 迭代 {iteration+1:03d}/{self.max_iter} | 历史最优得分: {self.historical_best_score:,.2f}")
                
        return self._decode_path(self.historical_best_pos), self.convergence_curve

# ===================== 修改后的主函数（10次循环保存） =====================
if __name__ == "__main__":
    save_dir = "PSO_new_path_evaluator"
    os.makedirs(save_dir, exist_ok=True)
    
    num_runs = 5
    all_final_scores = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*20} 第 {run_idx+1}/{num_runs} 次运行 {'='*20}")
        # 注意这里的 num_waypoints 被强制设定成了 30 以上来适应复杂的 3D 拐角
        planner = PSOPlanner(num_particles=100, max_iter=150, num_waypoints=45)
        best_path, history = planner.optimize()
        
        #planner.evaluator.debug_target_coverage(best_path)
        planner.plot_result(best_path, history, algo_name="PSO-3D", run_idx=run_idx, save_dir=save_dir)
        
        final_score = history[-1] if history else None
        all_final_scores.append(final_score)
        with open(os.path.join(save_dir, f"run_{run_idx:02d}_score.txt"), 'w') as f:
            f.write(f"Final score: {final_score:.2f}\n")
            
    # print("\n" + "="*50)
    # print("所有运行完成！结果保存在", save_dir)
    # print("各次最终得分:")
    # for i, score in enumerate(all_final_scores):
    #     print(f"  Run {i+1:02d}: {score:,.2f}")
    # if all_final_scores:
    #     avg = np.mean(all_final_scores)
    #     std = np.std(all_final_scores)
    #     print(f"\n平均得分: {avg:,.2f}  (±{std:,.2f})")
    # print("="*50)