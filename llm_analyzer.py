"""
大模型分析模块 - 用于调用大模型API进行教育报告分析
支持多种大模型服务：OpenAI、智谱AI、DeepSeek、Moonshot等
"""

import aiohttp
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio


class LLMAnalyzer:
    """大模型分析器"""
    
    def __init__(self, base_url: str, api_key: str, model: str = "glm-4"):
        """
        初始化大模型分析器
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            model: 模型名称
        """
        self.base_url = base_url.rstrip('/')  # 移除末尾斜杠
        self.api_key = api_key
        self.model = model
        
        # 检测是否为智谱AI
        self.is_zhipu = "bigmodel" in base_url.lower() or "zhipu" in base_url.lower()
    
    async def analyze_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送数据到大模型进行分析
        
        Args:
            stats: 统计数据字典
            
        Returns:
            分析结果字典
        """
        try:
            # 1. 构造提示词
            prompt = self._construct_prompt(stats)
            
            # 2. 选择endpoint
            endpoint = self._get_endpoint()
            
            # 3. 构造请求体
            payload = self._build_payload(prompt)
            
            # 4. 发送请求
            response = await self._make_request(endpoint, payload)
            
            # 5. 解析响应
            return self._parse_response(response)
            
        except Exception as e:
            print(f"❌ 大模型分析失败: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _get_endpoint(self) -> str:
        """根据服务选择endpoint"""
        if self.is_zhipu:
            base = self.base_url.rstrip('/')
            # 智谱AI的正确路径处理
            if base.endswith('/api/paas/v4'):
                # 用户输入了正确的完整路径
                return f"{base}/chat/completions"
            elif base.endswith('/api'):
                # 用户输入了旧格式，自动转换为新格式
                return f"{base}/paas/v4/chat/completions"
            else:
                # 其他情况，尝试添加正确的路径
                return f"{base}/api/paas/v4/chat/completions"
        else:
            # OpenAI兼容格式
            base = self.base_url.rstrip('/')
            if base.endswith('/v1'):
                return f"{base}/chat/completions"
            else:
                return f"{base}/v1/chat/completions"
    
    def _build_payload(self, prompt: str) -> dict:
        """构造请求体"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": """你是一位专业的医学教育评估专家，擅长分析医学诊断练习数据。
你的任务是分析用户的诊断练习表现，并给出专业、客观、有建设性的评价。

请确保你的回复是有效的JSON格式，包含以下字段：
- overall_evaluation: 整体表现评价（字符串）
- diagnosis_ability: 诊断能力分析（字符串）
- ai_tool_usage: AI工具使用情况（字符串）
- improvement_suggestions: 改进建议（数组，包含3-5条具体建议）
- strength_points: 优势点（数组，包含2-3条）
- weakness_points: 不足点（数组，包含2-3条）"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7
        }
        
        # 智谱AI特有参数
        if self.is_zhipu:
            payload["top_p"] = 0.9
        
        return payload
    
    async def _make_request(self, endpoint: str, payload: dict) -> dict:
        """
        发送HTTP请求到大模型API
        
        Args:
            endpoint: API endpoint
            payload: 请求体
            
        Returns:
            响应字典
        """
        timeout = aiohttp.ClientTimeout(total=60)  # 60秒超时
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(endpoint, json=payload, headers=headers) as response:
                # 检查HTTP状态码
                if response.status != 200:
                    error_detail = await response.text()
                    raise Exception(f"API调用失败: {response.status} - {error_detail}")
                
                result = await response.json()
                return result
    
    def _construct_prompt(self, stats: Dict[str, Any]) -> str:
        """构造分析提示词"""
        
        # 添加背景说明
        background = """
【背景说明】
这是一个医学诊断教育培训系统，用于训练医师的诊断能力。
学员在判读过程中，系统会提供AI辅助诊断结果（包括置信度分数）作为参考。
        本次分析的是学员在使用AI辅助后的表现。

【四种依赖的定义】
1. **正确依赖 (Correct Reliance)** 
   - 定义：AI诊断正确，学员也诊断正确
   - 含义:学员正确利用了AI的辅助,做出了正确判断
   - 医学意义:展现了良好的AI辅助诊断能力

2. **依赖不足 (Insufficient Reliance)**}
   - 定义:AI诊断正确,但学员诊断错误
   - 含义:AI给出了正确建议，但学员未能正确理解或采纳
   - 医学意义:需要加强对AI诊断结果的理解和信任

3. **正确独立 (Correct Independence)**}
   - 定义:AI诊断错误，但学员诊断正确
   - 含义:学员依靠自己的专业知识做出了正确判断
   - 医学意义:展现了扎实的专业诊断能力

4. **过度依赖 (Over Reliance)**}
   - 定义:AI诊断错误，学员也诊断错误
   - 含义:学员过度依赖AI，未能发现AI的错误
   - 医学意义:需要培养批判性思维，不应盲目依赖AI
"""
        
        # 构建完整提示词
        prompt = f"""
{background}

请分析以下医学诊断练习数据，并给出专业评价：

【基础指标】
- 准确率: {stats['accuracy']*100:.1f}%
- 敏感度: {stats['sensitivity']*100:.1f}%
- 特异性: {stats['specificity']*100:.1f}%
- 总用时: {stats.get('formatted_duration', '未知')}

【病种统计】
{json.dumps(stats.get('category_stats', {}), ensure_ascii=False, indent=2)}

【时间分析】
{json.dumps(stats.get('time_analysis', {}), ensure_ascii=False, indent=2)}

【AI依赖性分析】
{json.dumps(stats.get('ai_dependency', {}), ensure_ascii=False, indent=2)}

【病例详细数据】
真实标签列表: {stats.get('ground_truth_labels', [])}
AI判读标签列表: {stats.get('ai_labels', [])}
医师判读标签列表: {stats.get('user_labels', [])}
判读使用时间列表(秒): {stats.get('view_times', [])}

请从以下几个方面给出评价：
1. 整体表现评价 - 综合准确率、敏感度、特异性等指标
2. 诊断能力分析 - 分析在不同病种上的表现，识别强项和弱项
3. AI工具使用情况 - 基于四种依赖的统计数据，分析学员对AI辅助的使用情况
4. 改进建议 - 给出具体、可操作的改进建议

请以JSON格式返回评价结果，包含以下字段：
{{
    "overall_evaluation": "整体评价文字",
    "diagnosis_ability": "诊断能力分析文字",
    "ai_tool_usage": "AI工具使用分析文字（请结合四种依赖的统计数据进行分析）",
    "improvement_suggestions": ["建议1", "建议2", "建议3"],
    "strength_points": ["优势1", "优势2"],
    "weakness_points": ["不足1", "不足2"]
}}
"""
        return prompt
    
    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """
        解析大模型响应（支持多种格式）
        
        Args:
            response: API响应字典
            
        Returns:
            解析后的分析结果
        """
        try:
            # OpenAI格式
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
            # 智谱AI格式
            elif "data" in response and "choices" in response["data"]:
                content = response["data"]["choices"][0]["content"]
            else:
                raise ValueError("无法识别的响应格式")
            
            # 尝试解析JSON（可能包含在markdown代码块中）
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(1))
            else:
                # 尝试直接解析JSON
                analysis = json.loads(content)
            
            # 验证必要字段
            required_fields = ["overall_evaluation", "diagnosis_ability", "ai_tool_usage", 
                             "improvement_suggestions", "strength_points", "weakness_points"]
            
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"缺少必要字段: {field}")
            
            return {
                "status": "completed",
                "analysis": analysis
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {
                "status": "failed",
                "error": "JSON解析失败",
                "raw_content": content if 'content' in locals() else str(response)
            }
        except Exception as e:
            print(f"解析大模型响应失败: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "raw_response": response
            }


async def test_connection(base_url: str, api_key: str, model: str = "glm-4") -> Dict[str, Any]:
    """
    测试大模型连接
    
    Args:
        base_url: API基础URL
        api_key: API密钥
        model: 模型名称
        
    Returns:
        测试结果
    """
    try:
        analyzer = LLMAnalyzer(base_url, api_key, model)
        
        # 发送简单测试请求
        test_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "你好，请回复'连接成功'"}
            ],
            "temperature": 0.7
        }
        
        if analyzer.is_zhipu:
            test_payload["top_p"] = 0.9
        
        endpoint = analyzer._get_endpoint()
        response = await analyzer._make_request(endpoint, test_payload)
        
        # 解析响应
        if "choices" in response:
            content = response["choices"][0]["message"]["content"]
        elif "data" in response and "choices" in response["data"]:
            content = response["data"]["choices"][0]["content"]
        else:
            content = "连接成功"
        
        return {
            "success": True,
            "message": content
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
