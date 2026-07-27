import numpy as np
import matplotlib.pyplot as plt
import json
import getpass  
from openai import OpenAI  
import os

from pso_planner import PSOPlanner
from ssa_planner import SSAPlanner
from gwo_planner import GWOPlanner
from woa_planner import WOAPlanner
from ga_planner import GAPlanner

from path_evaluator import PathEvaluator

class Algorithm_Select_Agent:
    def __init__(self, evaluator, prior_knowledge=None, rainfall_mm=0.0, duration_hours=1.0, use_llm=False):
        """
        [LLM Ready] 仿生优化智能体 (Cross-Domain Adaptation & Hyperparameter Inference Version)
        """
        self.evaluator = evaluator
        self.env = evaluator.env
        self.rainfall_mm = rainfall_mm
        self.duration_hours = duration_hours
        self.use_llm = use_llm  
        self.prior_knowledge = prior_knowledge or {} 
        
        self.algo_map = {
            "PSO": PSOPlanner,
            "SSA": SSAPlanner,
            "GWO": GWOPlanner,
            "WOA": WOAPlanner,
            "GA": GAPlanner
        }

    def _get_map_meta_features(self):
        """
        计算当前地图的物理元特征，用于 LLM 进行跨域类比推理
        """
        area = (self.env.x_bounds[1] - self.env.x_bounds[0]) * (self.env.y_bounds[1] - self.env.y_bounds[0])
        num_obstacles = len(self.env.obstacles)
        # 障碍物密度指标 (个数 / 面积万平米)
        density = (num_obstacles / area) * 10000 if area > 0 else 0
        
        return {
            "Area (sq meters)": area,
            "Number of Obstacles": num_obstacles,
            "Obstacle Density Index": round(density, 2)
        }

    # ==========================================
    # 模块 A：LLM 专属准备层 (Prompt & Mock)
    # ==========================================
    def _build_llm_prompt(self):
        intensity = self.rainfall_mm / (self.duration_hours + 1e-5)
        target_names = [t.get('name', 'unknown') for t in self.env.target_areas]
        
        # 获取当前应用地图的元特征
        meta_features = self._get_map_meta_features()
        meta_features_str = json.dumps(meta_features, indent=2)
        
        # 【关键】：直接将整个先验知识 JSON 转换为字符串塞入 Prompt
        full_prior_knowledge_str = json.dumps(self.prior_knowledge, ensure_ascii=False, indent=2)
        
        prompt = f"""
        你是一个严谨的数据驱动型无人机路径规划指挥大脑。
        你现在面对的是一个全新的【实际应用地图】（真实的校区），但你手中只有在【标准测试集】（Easy, Medium, Hard）上跑批得到的历史先验知识数据。
        你需要进行“跨域类比推理”并决定最佳的算法与超参数。
        
        【1. 当前气象数据】
        - 累计降雨量: {self.rainfall_mm} mm
        - 持续时间: {self.duration_hours} h
        - 降雨强度: {intensity:.2f} mm/h
        
        【2. 当前应用地图特征 (Application Map)】
        - 现有巡检目标清单: {target_names}
        - 地图元特征物理指标:
        {meta_features_str}
        
        【3. 测试集先验知识库 (Prior Knowledge from Benchmarks)】
        包含 5 种算法在不同难度地图上 30 次运行的统计。请特别注意它们使用的 `algorithm_params_used` 及其对应的 `success_rate_percent`。
        {full_prior_knowledge_str}
        
        【决策推理要求】
        步骤 1. 地图类比 (Analogical Reasoning)：分析【当前应用地图】的元特征，判断它等同于测试集中的哪个难度（Easy/Medium/Hard）？
        步骤 2. 算法与参数推断：基于你认定的难度，查阅先验知识库。挑选一个胜率最高且最稳定的算法。同时，提取或微调该算法在先验知识库中使用的超参数（pop_size, max_iter, num_waypoints），作为本次的初始参数。
        步骤 3. 风险与目标评估：非暴雨时，为省电应剔除部分目标；暴雨时，需保留所有目标，并决定是否要拔高安全高度。
           
        【输出格式要求】
        必须严格输出为以下可解析的 JSON 格式，不要有任何额外字符或 markdown 标记：
        {{
            "analogical_reasoning": "简述你如何将当前地图特征映射到测试集的某个难度的？",
            "algorithm_reasoning": "说明你为何选择该算法，以及你是如何确定 pop_size, max_iter, num_waypoints 的？",
            "algorithm": "算法名称",
            "pop_size": 种群数量 (整数, 例如 80),
            "max_iter": 迭代次数 (整数, 例如 150),
            "num_waypoints": 路径控制点数量 (整数, 例如 35),
            "retained_targets": ["保留下来的目标name1", "name2", ...],
            "z_lift_required": true/false
        }}
        """
        return prompt

    def _call_llm_mock(self, prompt):
        """ 本地离线兜底规则 """
        intensity = self.rainfall_mm / (self.duration_hours + 1e-5)
        
        if self.rainfall_mm >= 50.0 or intensity >= 15.0:
            mock_response = {
                "analogical_reasoning": "当前为本地降级模式，假设当前地图复杂度极高。",
                "algorithm_reasoning": "降雨量达暴雨级别。兜底调用破壁能力强的 SSA。",
                "algorithm": "SSA",
                "pop_size": 80,
                "max_iter": 150,
                "num_waypoints": int(len(self.env.target_areas) * 3.5),
                "retained_targets": [t.get('name') for t in self.env.target_areas],
                "z_lift_required": True
            }
        else:
            mock_response = {
                "analogical_reasoning": "当前为本地降级模式，假设当前地图复杂度中等。",
                "algorithm_reasoning": "常规天气。兜底调用收敛快的 PSO。",
                "algorithm": "PSO",
                "pop_size": 80,
                "max_iter": 150,
                "num_waypoints": int(len(self.env.target_areas) * 3.5),
                "retained_targets": [t.get('name') for t in self.env.target_areas][:2],
                "z_lift_required": False
            }
            
        return json.dumps(mock_response, ensure_ascii=False, indent=2)

    # ==========================================
    # 模块 B：解析与执行层 
    # ==========================================
    def _execute_decision(self, decision_json):
        print("\n" + "=" * 65)
        print("[仿生优化智能体] 正在解析 LLM 战术决策包...")
        print(f"   -> 跨域类比推断: \033[93m{decision_json.get('analogical_reasoning', '无')}\033[0m")
        print(f"   -> 战术与参数选择逻辑: \033[3m{decision_json.get('algorithm_reasoning', '无')}\033[0m")
        
        active_targets = []
        for t in self.env.target_areas:
            if t.get('name') in decision_json.get('retained_targets', []):
                if decision_json.get('z_lift_required', False):
                    t['z_min'] = t.get('z_min', 0.0) + 1.0
                active_targets.append(t)
                
        if len(active_targets) == 0:
             print("   -> [警告] LLM 裁减掉了所有目标，已强制兜底保留 2 个防崩溃！")
             active_targets = self.env.target_areas[:2]
                
        self.evaluator.update_env_targets(active_targets)
        
        # 从 JSON 中提取 LLM 决定的超参数
        algo_name = decision_json.get('algorithm', 'PSO')
        pop_size = int(decision_json.get('pop_size', 80))
        max_iter = int(decision_json.get('max_iter', 150))
        num_waypoints = int(decision_json.get('num_waypoints', len(active_targets) * 3))
        
        print(f"   -> 地图重构完成。当前目标: {len(active_targets)} 个, 障碍物: {len(self.env.obstacles)} 个")
        return algo_name, pop_size, max_iter, num_waypoints

    
    def get_fallback_algorithm(self, current_algo, failure_details=None):
        print(f"\n[仿生优化智能体] 收到前线溃败报告！正在向云端大脑请求 {current_algo} 的替补战术...")
        
        available_algos = [algo for algo in self.algo_map.keys() if algo != current_algo]
        
        if failure_details:
            failed_items = {k: v for k, v in failure_details.items() if v > 0}
            details_str = json.dumps(failed_items, ensure_ascii=False, indent=2)
        else:
            details_str = "无详细错误日志"
            
        meta_features_str = json.dumps(self._get_map_meta_features(), indent=2)
        full_prior_knowledge_str = json.dumps(self.prior_knowledge, ensure_ascii=False, indent=2)
        
        prompt = f"""
        你是一个无人机 3D 路径规划的最高指挥大脑。
        当前在应用地图中执行的 '{current_algo}' 已经连续 2 轮陷入死锁，无法找到安全航线。
        
        【实际应用地图特征】
        - 物理元特征: {meta_features_str}
        - 需巡检目标数: {len(self.env.target_areas)}
        - 可选替补算法池: {available_algos}
        
        【替补先验知识库 (Prior Knowledge)】
        {full_prior_knowledge_str}
        
        【失败者的体检诊断书 (Fitness Breakdown)】
        导致 {current_algo} 卡壳的具体扣分项明细：
        {details_str}
        
        【推理步骤】
        1. 重新映射当前应用地图到测试集难度。
        2. 查阅先验知识库，寻找能克服上述“诊断书”死穴的高胜率替补算法。
        3. 为这个替补算法设定合适的超参数。如果之前撞墙太多，可能需要适当增加 pop_size 或 num_waypoints 增加搜索维度。
        
        请严格输出 JSON 格式：
        {{
            "analogical_reasoning": "简述地图难度映射与分析",
            "reasoning": "你是如何结合胜率与诊断书，挑选出这个替补算法并设定其参数的？",
            "algorithm": "算法名称",
            "pop_size": 种群数量 (整数),
            "max_iter": 迭代次数 (整数),
            "num_waypoints": 控制点数量 (整数)
        }}
        """

        if self.use_llm:
            response_str = self._call_llm_api(prompt)
            
            # JSON 解析与容错保护
            try:
                decision_data = json.loads(response_str)
            except json.JSONDecodeError:
                print("   -> [解析异常] 大模型未返回合法 JSON，切回本地默认替补规则...")
                decision_data = {"algorithm": available_algos[0], "reasoning": "JSON 解析失败，触发降级顺序轮转。"}
        else:
            # 本地无 LLM 时的 Mock 降级兜底
            decision_data = {
                "algorithm": available_algos[0], 
                "reasoning": "未启用 LLM，触发本地默认轮转规则。"
            }

        next_algo = decision_data.get("algorithm", available_algos[0])
        if next_algo not in self.algo_map:
            next_algo = available_algos[0]

        pop_size = int(decision_data.get('pop_size', 80))
        max_iter = int(decision_data.get('max_iter', 150))
        num_waypoints = int(decision_data.get('num_waypoints', 30))

        print(f"   -> 跨域类比推断: \033[93m{decision_data.get('analogical_reasoning', '无')}\033[0m")
        print(f"   -> 换将理由与参数设计: \033[3m{decision_data.get('reasoning', '无')}\033[0m")
        print(f"   -> 云端战术变更: 替补算法 \033[96m{next_algo}\033[0m, 种群={pop_size}, 迭代={max_iter}, 控制点={num_waypoints}")
        
        return next_algo, pop_size, max_iter, num_waypoints

    # ==========================================
    # 真实的 LLM 调度接口
    # ==========================================
    def _call_llm_api(self, prompt):
        print("   -> [API 连线] 正在呼叫云端 LLM 大脑进行跨域类比与超参数推演...")
        
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        if not api_key:
            print("\n\033[93m[系统提示] 需要连接云端大脑，但未检测到 DEEPSEEK_API_KEY。\033[0m")
            api_key = getpass.getpass("请输入您的 DeepSeek API Key (屏幕不可见，按回车确认): ")
            os.environ["DEEPSEEK_API_KEY"] = api_key
            
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-v4-pro" 
        
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个严谨的智能体决策中枢。你必须强制输出合法的 JSON 格式数据，不要包含任何 Markdown 标记或多余文本。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.1, 
                timeout=20.0,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
            )
            
            result_str = response.choices[0].message.content
            print("   -> [API 响应] 云端参数推演成功！")
            return result_str
            
        except Exception as e:
            print(f"   -> [API 异常] 连线失败或超时: {e}")
            print("   -> [降级机制] 正在切回本地兜底规则...")
            return self._call_llm_mock(prompt)

    # ==========================================
    # 模块 C：主控枢纽
    # ==========================================
    def make_decision(self):
        if self.use_llm:
            prompt = self._build_llm_prompt()
            response_str = self._call_llm_api(prompt) 
            
            # JSON 解析与容错保护
            try:
                decision_data = json.loads(response_str)
            except json.JSONDecodeError:
                print("   -> [解析异常] 大模型未返回合法 JSON，切回本地默认替补规则...")
                response_str = self._call_llm_mock(prompt)
                decision_data = json.loads(response_str)
        else:
            # 本地无 LLM 时的 Mock 降级兜底
            response_str = self._call_llm_mock(self._build_llm_prompt())
            decision_data = json.loads(response_str)

        algo_name, final_pop, final_iter, final_waypoints = self._execute_decision(decision_data)
        
        if algo_name not in self.algo_map:
             print(f"   -> [幻觉拦截] LLM 选了不存在的算法 {algo_name}，强制修正为 PSO！")
             algo_name = "PSO"
             
        print(f"   -> 最终委派算法: \033[96m{algo_name}\033[0m")
        print(f"   -> 大模型下发的超参数: 种群数 = {final_pop}, 迭代 = {final_iter}, 基因长度 = {final_waypoints}")
        print("=" * 65 + "\n")
        
        PlannerClass = self.algo_map.get(algo_name, PSOPlanner)
        kwargs = {'evaluator': self.evaluator, 'num_waypoints': final_waypoints, 'max_iter': final_iter}
        
        if algo_name in ["ACO", "DSACO"]: kwargs['num_ants'] = final_pop
        elif algo_name == "PSO": kwargs['num_particles'] = final_pop
        elif algo_name == "GWO": kwargs['num_wolves'] = final_pop
        elif algo_name == "SSA": kwargs['num_sparrows'] = final_pop
        elif algo_name == "WOA": kwargs['pop_size'] = final_pop
        elif algo_name == "GA": kwargs['pop_size'] = final_pop
        elif algo_name == "HybridPSOGWO": kwargs['pop_size'] = final_pop

        return PlannerClass(**kwargs)
