# llm/qwen_api.py

from openai import OpenAI
import json
import os

class QwenNERAPI:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen-plus"

    def zero_shot_inference(self, text, entity_types):
        prompt = f"""你是命名实体识别专家。请从输入文本中识别以下实体类型：
{', '.join(entity_types)}。

输出必须为严格的JSON格式，键为实体类型，值为实体名称列表（无位置信息）。

文本：{text}"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)

    def few_shot_inference(self, text, entity_types, examples):
        """examples: [{"text": "...", "entities": {...}}, ...]"""
        example_text = "\n".join([
            f"文本：{ex['text']}\n输出：{json.dumps(ex['entities'], ensure_ascii=False)}"
            for ex in examples
        ])
        prompt = f"""你是命名实体识别专家。遵循以下示例的格式。

{example_text}

实体类型：{', '.join(entity_types)}

文本：{text}
输出："""
        # ... 调用API，代码同上