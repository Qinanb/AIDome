"""
AI 智能分析模块
调用 OpenAI 兼容接口对检测结果进行文字解读
"""
import os
import json
import base64
import requests


CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_config.json')


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def is_configured():
    cfg = load_config()
    return bool(cfg.get('api_base') and cfg.get('api_key') and cfg.get('model'))


def analyze_detection(detections, image_base64=None, conf_threshold=0.25):
    """
    对检测结果进行 AI 分析
    detections: [{'class': 'drone', 'confidence': 95.2, 'bbox': [...]}, ...]
    image_base64: 可选，图片的 base64 编码
    返回: AI 生成的分析文本
    """
    cfg = load_config()
    api_base = cfg.get('api_base', '').rstrip('/')
    api_key = cfg.get('api_key', '')
    model = cfg.get('model', '')

    if not all([api_base, api_key, model]):
        return None, "AI 未配置，请在设置页面填写 API 信息"

    # 统计信息
    drone_list = [d for d in detections if d['class'] == 'drone']
    bird_list = [d for d in detections if d['class'] == 'bird']

    stats_text = (
        f"检测置信度阈值: {conf_threshold}\n"
        f"总计检测到 {len(detections)} 个目标:\n"
        f"  - 无人机(drone): {len(drone_list)} 个\n"
        f"  - 鸟类(bird): {len(bird_list)} 个\n"
    )

    if drone_list:
        confs = [d['confidence'] for d in drone_list]
        stats_text += f"  - 无人机置信度范围: {min(confs):.1f}% ~ {max(confs):.1f}%\n"
    if bird_list:
        confs = [d['confidence'] for d in bird_list]
        stats_text += f"  - 鸟类置信度范围: {min(confs):.1f}% ~ {max(confs):.1f}%\n"

    # 构造 prompt
    system_prompt = (
        "你是一个专业的无人机检测分析助手。用户使用 YOLO 模型对图像进行了目标检测，"
        "检测类别为无人机(drone)和鸟类(bird)。请根据检测结果提供专业的中文分析报告。\n\n"
        "分析要求:\n"
        "1. 概述检测场景\n"
        "2. 分析检测到的目标情况（数量、分布、置信度）\n"
        "3. 如果检测到无人机，评估潜在风险/威胁等级\n"
        "4. 如果同时检测到鸟类，分析鸟类对无人机操作的潜在干扰\n"
        "5. 给出操作建议\n\n"
        "请用简洁专业的语言，分点列出，使用 Markdown 格式。"
    )

    user_content = []

    # 如果有图片，添加到消息中
    if image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })

    user_content.append({
        "type": "text",
        "text": f"以下是检测结果:\n\n{stats_text}\n请进行分析。"
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content if image_base64 else user_content[0]["text"]}
    ]

    # 调用 API
    url = f"{api_base}/v1/chat/completions" if '/v1' not in api_base else f"{api_base}/chat/completions"
    # 兼容不同 API 格式
    if api_base.endswith('/v1'):
        url = f"{api_base}/chat/completions"
    else:
        url = f"{api_base}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message']['content']
        return content, None
    except requests.exceptions.Timeout:
        return None, "AI 响应超时，请稍后重试"
    except requests.exceptions.ConnectionError:
        return None, f"无法连接到 API 地址: {api_base}"
    except requests.exceptions.HTTPError as e:
        return None, f"API 请求失败: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        return None, f"AI 分析失败: {str(e)}"
