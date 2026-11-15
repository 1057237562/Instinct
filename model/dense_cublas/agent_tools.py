"""Original Agent RL simulated environment, with a bounded AST math evaluator.

Weather/time/exchange outputs are dataset fixtures, not live external services.
"""
import ast
import json
import math
import operator
import re
from typing import List, Optional


def safe_math(expression):
    if len(expression) > 256: raise ValueError('expression too long')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 64: raise ValueError('expression too complex')
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                 ast.Mod: operator.mod, ast.Pow: operator.pow}
    functions = {k: getattr(math, k) for k in ('sqrt','sin','cos','tan','log','log10','exp','fabs','ceil','floor')}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int,float): result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op,ast.USub) else 1)
        elif isinstance(node,ast.BinOp) and type(node.op) in operators:
            left,right=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Pow) and abs(right)>100: raise ValueError('exponent too large')
            result=operators[type(node.op)](left,right)
        elif isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='math' and node.attr in ('pi','e'):
            result=getattr(math,node.attr)
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='math' and node.func.attr in functions and not node.keywords:
            result=functions[node.func.attr](*[visit(a) for a in node.args])
        else: raise ValueError('unsupported expression')
        if not isinstance(result,(int,float)) or not math.isfinite(result) or abs(result)>1e100: raise ValueError('result out of range')
        return result
    return visit(tree.body)

def rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """计算文本的重复惩罚分数。

    将文本切分为 n-gram，重复 n-gram 占比越高惩罚越大，结果被 cap 截断；
    无有效 n-gram 时返回 0.0。用于抑制模型回复中的机械重复。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义 ========
# 暴露给模型的工具 schema 列表（OpenAI function calling 格式），注入到 system 提示中
TOOLS = [
    {"type": "function", "function": {
        "name": "calculate_math", "description": "计算数学表达式",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "unit_converter", "description": "单位换算",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
            "required": ["value", "from_unit", "to_unit"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_current_weather", "description": "获取天气",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    }},
    {"type": "function", "function": {
        "name": "get_current_time", "description": "获取时间",
        "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_exchange_rate", "description": "查询汇率",
        "parameters": {
            "type": "object",
            "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
            "required": ["from_currency", "to_currency"],
        },
    }},
    {"type": "function", "function": {
        "name": "translate_text", "description": "翻译文本",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"],
        },
    }},
]

# ======== 模拟数据 ========
WEATHER_DATA = {
    "北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"),
    "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"),
    "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"),
    "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗"),
}
TIME_DATA = {
    "Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00",
    "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00",
    "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00",
}
EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048,
    ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72,
}
TRANSLATE_DATA = {
    ("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
    ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程",
    ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐",
}
UNIT_DATA = {
    "km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592,
    "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556,
}

# ======== 模拟执行 ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math(
        str(args.get("expression", "0"))
        .replace("^", "**").replace("×", "*").replace("÷", "/")
        .replace("−", "-").replace("（", "(").replace("）", ")"),
    ))},
    "unit_converter": lambda args: {"result": round(
        float(args.get("value", 0))
        * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1),
        4,
    )},
    "get_current_weather": lambda args: (lambda w: {
        "city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1],
    })(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {
        "datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"),
        "timezone": args.get("timezone", "Asia/Shanghai"),
    },
    "get_exchange_rate": lambda args: {
        "from": args.get("from_currency"), "to": args.get("to_currency"),
        "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0),
    },
    "translate_text": lambda args: {
        "translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", "")),
    },
}

# ======== 参数校验 ========
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text: str) -> List[dict]:
    """从回复文本中解析 <tool_call>...</tool_call> 包裹的 JSON 工具调用。

    每个匹配片段按 JSON 解析为 {"name": ..., "arguments": ...} 格式，
    解析失败的片段直接跳过。
    """
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except: pass
    return calls

def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if fn is None or not isinstance(args, dict): return None
    try: return fn(args)
    except (ValueError, TypeError, ArithmeticError, KeyError): return None
