"""读持仓截图（手机/PC通用）"""
import os, base64, json, sys

def read_screenshot(img_path, prompt_suffix=""):
    """调用智谱GLM-4V读取截图，返回文本结果"""
    zhipu_key = os.environ.get('ZHIPUAI_API_KEY', '')
    if not zhipu_key:
        return None, "ZHIPUAI_API_KEY not set"
    
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    import urllib.request
    url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {zhipu_key}'
    }
    
    data = {
        'model': 'glm-4v',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt_suffix or '请详细描述这张图片中的所有文字信息，包括每个数字和单位。'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}}
            ]
        }],
        'max_tokens': 1024,
        'temperature': 0.01
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
    res = urllib.request.urlopen(req, timeout=60)
    result = json.loads(res.read())
    text = result['choices'][0]['message']['content']
    return text, None


if __name__ == '__main__':
    # 用最新的inbound图片
    import glob
    files = sorted(glob.glob('/home/hermes/.hermes/openclaw-archive/media/inbound/*.jpg'), key=os.path.getmtime, reverse=True)
    
    prompt = """这张图片是手机截图，显示股票持仓信息。
请仔细识别图中的所有文字和数字，特别是：

1. 有哪些股票？
2. 每只股票的：名称、代码、持仓数量（股）、成本价（元）、最新价（元）、盈亏金额（元）
3. 如果有汇总信息（总资产、总市值、总盈亏等），也一起读出

注意：这是手机界面截图，数字可能比PC紧凑。认真识别每个数字和单位。
格式：每行一个股票，用 | 分隔 名称|代码|持仓数量|成本价|现价|盈亏金额。"""
    
    text, err = read_screenshot(files[0], prompt)
    if err:
        print(f"Error: {err}")
    else:
        print(text)
