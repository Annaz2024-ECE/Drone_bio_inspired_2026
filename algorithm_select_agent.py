import numpy as np
import matplotlib.pyplot as plt
import json

# 导入所有底层规划器
from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from woa_planner_fix import WOAPlanner
#from dsaco_planner import DSACOPlanner
from hybrid_pso_gwo import HybridPSOGWO

from path_evaluator import PathEvaluator

class Algorithm_Select_Agent:
    def __init__(self, evaluator, rainfall_mm=0.0, duration_hours=1.0, use_llm=False):
        """
        [LLM Ready] 仿生优化智能体
        :param use_llm: 布尔开关，决定是走本地 if-else 规则，还是走 LLM 决策链路
        """
        self.evaluator = evaluator
        self.env = evaluator.env
        self.rainfall_mm = rainfall_mm
        self.duration_hours = duration_hours
        self.use_llm = use_llm  # LLM 接入开关
        
        # 建立算法名称与类的映射字典，方便 LLM 字符串输出后直接映射
        self.algo_map = {
            "PSO": PSOPlanner,
            "SSA": SSAPlanner,
            "GWO": GWOPlanner,
            "WOA": WOAPlanner,
            #"DSACO": DSACOPlanner,
            "HybridPSOGWO": HybridPSOGWO
        }

    # ==========================================
    # 模块 A：LLM 专属准备层 (Prompt & Mock)
    # ==========================================
    def _build_llm_prompt(self):
        """
        【LLM 接口预备】收集当前环境的所有上下文，打包成 Prompt 发给 LLM
        """
        intensity = self.rainfall_mm / (self.duration_hours + 1e-5)
        
        # 提取地图目标点信息
        target_names = [t.get('name', 'unknown') for t in self.env.target_areas]
        
        prompt = f"""
        你是一个无人机路径规划的指挥大脑。请根据气象数据和地图信息进行决策。
        
        【气象数据】
        - 累计降雨量: {self.rainfall_mm} mm
        - 持续时间: {self.duration_hours} h
        - 降雨强度: {intensity:.2f} mm/h
        
        【地图数据】
        - 现有巡检目标清单: {target_names}
        - 障碍物数量: {len(self.env.obstacles)}
        
        【决策规则】
        1. 风险评估：根据降雨决定 Risk Level (High/Medium/Low)。
        2. 目标裁剪：如果非暴雨，请剔除 lake, river, playground, garage, pool 等低洼易积水地带以省电；如果暴雨，必须保留它们进行防涝巡检，并将它们的 z_min 提高 3.0 米。
        3. 算法分配：
           - High 风险选择 "HybridPSOGWO"
           - Medium 风险选择 "SSA"
           - Low 风险选择 "PSO"
           
        【输出格式要求】
        请严格输出为可解析的 JSON 格式，不要包含任何额外字符：
        {{
            "risk_level": "High/Medium/Low",
            "reasoning": "你的决策思考过程",
            "algorithm": "算法名称",
            "retained_targets": ["保留下来的目标name1", "name2", ...],
            "z_lift_required": true/false (是否需要整体拔高探查高度)
        }}
        """
        return prompt

    def _call_llm_mock(self, prompt):
        """
        【LLM 接口预备】模拟大语言模型的返回。
        未来只要把这个函数的内容替换成 openai.ChatCompletion 或你们自己部署模型的 API 即可！
        """
        # 提取降雨强度，模拟 LLM 的“思考逻辑”
        intensity = self.rainfall_mm / (self.duration_hours + 1e-5)
        
        # 模拟生成 JSON
        if self.rainfall_mm >= 50.0 or intensity >= 15.0:
            mock_response = {
                "risk_level": "High",
                "reasoning": "降雨量达暴雨级别，低洼地带积水溢流风险极高，必须全部巡检且拔高安全高度，地形恶劣启用 HybridPSOGWO。",
                "algorithm": "HybridPSOGWO",
                "retained_targets": [t.get('name') for t in self.env.target_areas],
                "z_lift_required": True
            }
        elif self.rainfall_mm >= 15.0 or intensity >= 5.0:
            mock_response = {
                "risk_level": "Medium",
                "reasoning": "中雨级别，建议启用 SSA 平稳避障，并针对低洼区域进行预防性检查。",
                "algorithm": "SSA",
                "retained_targets": [t.get('name') for t in self.env.target_areas],
                "z_lift_required": False
            }
        else:
            # 过滤掉低洼地带
            low_lying = ['lake', 'river', 'playground', 'garage', 'pool']
            retained = [t.get('name') for t in self.env.target_areas if not any(k in t.get('name', '').lower() for k in low_lying)]
            mock_response = {
                "risk_level": "Low",
                "reasoning": "常规天气，积水风险低，剔除湖泊/操场等低洼目标以节省算力，启用 PSO 快速收敛。",
                "algorithm": "PSO",
                "retained_targets": retained,
                "z_lift_required": False
            }
            
        # 返回一个 JSON 字符串，完美模拟真实的 LLM 响应体
        return json.dumps(mock_response, ensure_ascii=False, indent=2)


    # ==========================================
    # 模块 B：解析与执行层 (通用于 LLM 和 规则引擎)
    # ==========================================
    def _execute_decision(self, decision_json):
        """
        接收标准化的 JSON 决策（无论是由 LLM 生成，还是 Mock 生成），执行物理映射。
        """
        print("\n" + "=" * 65)
        print("[仿生优化智能体] 正在解析 大脑决策 (JSON Payload)...")
        print(f"   -> 思考逻辑: {decision_json['reasoning']}")
        print(f"   -> 评估风险: {decision_json['risk_level']}")
        
        # 1. 执行目标裁剪
        active_targets = []
        for t in self.env.target_areas:
            if t.get('name') in decision_json['retained_targets']:
                # 执行高度物理干预
                if decision_json['z_lift_required']:
                    t['z_min'] = t.get('z_min', 0.0) + 3.0
                active_targets.append(t)
                
        self.evaluator.update_env_targets(active_targets)
        num_targets = len(active_targets)
        num_obstacles = len(self.env.obstacles)
        
        print(f"   -> 地图重构: 过滤后需巡检目标 {num_targets} 个, 障碍物 {num_obstacles} 个")
        return decision_json['risk_level'], decision_json['algorithm'], num_targets, num_obstacles

    def _fine_tune_parameters(self, algo_name, num_targets, num_obstacles):
        """
        核心算力计算器：无论前端是 LLM 还是规则，底层算力的物理需求是不变的。
        """
        map_area = self.env.x_bounds[1] * self.env.y_bounds[1]
        is_large_map = map_area > 20000 
        
        base_pop = 120 if is_large_map else 80
        base_iter = 200 if is_large_map else 150
        base_waypoints = int(num_targets) 

        if algo_name in ["PSO", "SSA"]:
            base_waypoints = int(num_targets * 3.5) 
        elif algo_name in ["GWO", "WOA", "HybridPSOGWO"]:
            base_waypoints = int(num_targets * 1.8) 

        return base_pop, base_iter, base_waypoints

    # ==========================================
    # 模块 C：主控枢纽
    # ==========================================
    def make_decision(self):
        # 1. 产生决策 (路由选择)
        if self.use_llm:
            prompt = self._build_llm_prompt()
            # 【未来改这里】： response_str = requests.post("YOUR_LLM_API_URL", json={"prompt": prompt}).json()
            response_str = self._call_llm_mock(prompt) 
            decision_data = json.loads(response_str)
        else:
            # 兼容非 LLM 模式，自己调 Mock 直接生成数据字典
            response_str = self._call_llm_mock(self._build_llm_prompt())
            decision_data = json.loads(response_str)

        # 2. 解析与环境干预
        risk_level, algo_name, num_targets, num_obstacles = self._execute_decision(decision_data)
        
        # 3. 计算算力
        final_pop, final_iter, final_waypoints = self._fine_tune_parameters(algo_name, num_targets, num_obstacles)
        
        print(f"   -> 最终委派算法: \033[96m{algo_name}\033[0m")
        print(f"   -> 算力调度清单: 种群数 = {final_pop}, 迭代 = {final_iter}, 基因长度 = {final_waypoints}")
        print("=" * 65 + "\n")
        
        # 4. 统一打包实例化
        PlannerClass = self.algo_map.get(algo_name, PSOPlanner)
        kwargs = {'evaluator': self.evaluator, 'num_waypoints': final_waypoints, 'max_iter': final_iter}
        
        if algo_name in ["ACO", "DSACO"]: kwargs['num_ants'] = final_pop
        elif algo_name == "PSO": kwargs['num_particles'] = final_pop
        elif algo_name == "GWO": kwargs['num_wolves'] = final_pop
        elif algo_name == "SSA": kwargs['num_sparrows'] = final_pop
        elif algo_name == "WOA": kwargs['pop_size'] = final_pop
        elif algo_name == "HybridPSOGWO": kwargs['pop_size'] = final_pop

        return PlannerClass(**kwargs)