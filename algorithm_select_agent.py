import numpy as np
import matplotlib.pyplot as plt
import json
import getpass  # 【新增】用于安全输入密码的模块
from openai import OpenAI  # 【新增】导入大模型 SDK
import os

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
        2. 目标裁剪：如果非暴雨，请剔除一些巡检区域用于省电；如果暴雨，必须保留它们进行防涝巡检，并考虑是否要将巡检高度提升一些。
        3. 算法分配：根据地图和降雨情况，从PSO，SSA，GWO，WOA四种算法里面选择一种用于路径规划
           
        【输出格式要求】
        请严格输出为可解析的 JSON 格式，不要包含任何额外字符：
        {{
            "risk_level": "High/Medium/Low",
            "reasoning": "你的决策思考过程，包括risk-level计算，algorithm、retained_targets、z_lift_required决策原因",
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
                    t['z_min'] = t.get('z_min', 0.0) + 1.0
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
    

    def get_fallback_algorithm(self, current_algo, failure_details=None):
        """
        【LLM 动态接管机制】
        当前线算法卡死时，将现场态势与具体的“失败诊断书”打包发送给 LLM，请求实时换将。
        """
        print(f"\n[仿生优化智能体] 收到前线溃败报告！正在向云端大脑请求 {current_algo} 的替补战术...")
        
        available_algos = [algo for algo in self.algo_map.keys() if algo != current_algo]
        
        # 将失败明细格式化为漂亮的 JSON 字符串，方便大模型阅读
        if failure_details:
            # 过滤掉得分为 0 的完美项，只给大模型看扣分项，节省 Token 和注意力
            failed_items = {k: v for k, v in failure_details.items() if v > 0}
            details_str = json.dumps(failed_items, ensure_ascii=False, indent=2)
        else:
            details_str = "无详细错误日志"
        
        # 2. 构建专属的“危机求助 Prompt” (加入诊断书)
        prompt = f"""
        你是一个无人机 3D 路径规划的最高指挥大脑。
        当前正在执行的算法 '{current_algo}' 已经连续 2 轮陷入局部最优死锁，无法找到安全的绕楼航线。
        
        【当前战场态势】
        - 气象环境: 降雨量 {self.rainfall_mm}mm (持续 {self.duration_hours}h)
        - 剩余需巡检目标: {len(self.env.target_areas)} 个
        - 地图障碍物数量: {len(self.env.obstacles)} 个
        - 可选替补算法池: {available_algos}
        
        【失败者的体检诊断书 (Fitness Breakdown)】
        以下是导致 {current_algo} 算法卡壳的具体扣分项明细：
        {details_str}
        
        请从备选池中挑选 1 个最适合针对性解决上述“体检诊断书”死穴的替补算法，并严格输出以下 JSON 格式：
        {{
            "algorithm": "算法名称",
            "reasoning": "你的战术推演过程：你是如何根据诊断书上的扣分项，判定这个替补算法能破局的？"
        }}
        """

        # 3. 呼叫云端大脑
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

        # 4. 提取决策，并加上一层“幻觉防御” (防止大模型胡编乱造了一个我们没有的算法名字)
        next_algo = decision_data.get("algorithm", available_algos[0])
        if next_algo not in self.algo_map:
            print(f"   -> [幻觉拦截] 大模型推荐了不存在的算法 '{next_algo}'，已强行修正！")
            next_algo = available_algos[0]

        print(f"   -> 云端战术变更: 决定派出替补算法 \033[96m{next_algo}\033[0m")
        print(f"   -> 换将理由: \033[3m{decision_data.get('reasoning', '无')}\033[0m")
        
        # 5. 极其关键：必须用新算法重新计算基因长度与算力！
        num_targets = len(self.env.target_areas)
        num_obstacles = len(self.env.obstacles)
        new_pop, new_iter, new_wp = self._fine_tune_parameters(next_algo, num_targets, num_obstacles)
        
        print(f"   -> 替补算力重配: 种群数 = {new_pop}, 迭代 = {new_iter}, 基因长度 = {new_wp}")
        
        return next_algo, new_pop, new_iter, new_wp

    # ==========================================
    # 【新增】真实的 LLM 调度接口
    # ==========================================
    def _call_llm_api(self, prompt):
        """
        调用真实的云端大模型 API。
        强制要求返回 JSON，并带有网络异常断开时的降级保护。
        """
        print("   -> [API 连线] 正在呼叫云端 LLM 大脑进行气象与地形分析...")
        
        # ==========================================
        # 【修改这里】：动态安全获取 API Key
        # ==========================================
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        # 如果环境变量里没有，就弹出安全输入提示
        if not api_key:
            print("\n\033[93m[系统提示] 需要连接云端大脑，但未检测到 DEEPSEEK_API_KEY。\033[0m")
            api_key = getpass.getpass("请输入您的 DeepSeek API Key (输入时屏幕不可见，按回车确认): ")
            
            # 存入本次运行的临时环境变量中
            # 这样如果系统在一个大循环里多次调用 LLM，就不会烦人地让你每次都输入了
            os.environ["DEEPSEEK_API_KEY"] = api_key
            
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-v4-flash"
        
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个严谨的具身智能无人机指挥系统。你必须强制输出合法的 JSON 格式数据，不要包含任何 Markdown 标记或多余的解释文本。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.1, 
                timeout=15.0,
                extra_body={"thinking": {"type": "disabled"}}
            )
            
            result_str = response.choices[0].message.content
            print("   -> [API 响应] 云端大脑决策接收成功！")

            # ==========================================
            # 【新增】：将 LLM 的原始输出打印到终端供你赏玩
            # ==========================================
            # print("\n" + "·" * 40)
            # print("\033[94m[LLM 原始 JSON 输出预览]\033[0m")
            # print(result_str)
            # print("·" * 40 + "\n")

            return result_str
            
        except Exception as e:
            print(f"   -> [API 异常] 连线失败或超时: {e}")
            print("   -> [降级机制] 正在切回本地紧急战术规则引擎 (Mock)...")
            return self._call_llm_mock(prompt)

    # ==========================================
    # 模块 C：主控枢纽
    # ==========================================
    def make_decision(self):
        # 1. 产生决策 (路由选择)
        if self.use_llm:
            prompt = self._build_llm_prompt()
            response_str = self._call_llm_api(prompt) 
            
            # 【新增 JSON 解析保护】防止大模型发疯输出无效格式
            try:
                decision_data = json.loads(response_str)
            except json.JSONDecodeError:
                print("   -> [解析异常] 大模型未返回合法 JSON，切回本地规则库...")
                response_str = self._call_llm_mock(prompt)
                decision_data = json.loads(response_str)
        else:
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