import asyncio
from openai import AsyncOpenAI  # 推荐使用异步客户端以获得更好性能
import time


async def call_multimodal_api():
    """
    调用运行在 192.168.100.15:8091 的 vLLM 多模态 API。
    此示例假设模型是 Qwen/Qwen3-VL-8B-Instruct，支持文本和图像输入。
    """
    # 1. 创建客户端实例
    # base_url 指向您的 vLLM 服务器地址
    # api_key 对于 vLLM 默认是 'EMPTY'，除非您自己设置了密钥
    client = AsyncOpenAI(base_url="http://192.168.110.217:8091/v1", api_key="EMPTY")

    # 2. 准备输入内容
    #    Qwen3-VL 支持文本和图像。图像需要转换为 URL 或 Base64 编码。
    #    示例：发送一张图片和一段文字提示词

    # --- 示例 1: 使用网络图片 URL ---
    # image_url = "https://example.com/path/to/your/image.jpg"

    # --- 示例 2: 使用本地图片并转为 data URL (Base64) ---
    import base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

    # 请将下面的路径替换为您本地图像的实际路径
    local_image_path = r"D:\RAG_img\data\人物介绍\塞罗奥特曼\赛罗1.jpg"
    encoded_image = encode_image(local_image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # 使用网络图片URL
                    # "image_url": {"url": image_url}
                    # 使用 Base64 编码的本地图像
                    "image_url": {"url": encoded_image}
                },
                {
                    "type": "text",
                    "text": "请简短描述画面中的内容，不要超过100个字。"  # 您的文本提示
                }
            ]
        }
    ]

    try:
        start_time = time.time()
        # 3. 发起聊天补全请求
        chat_completion = await client.chat.completions.create(
            model="/data/ai/model/models/cpatonn-mirror/Qwen3-VL-8B-Instruct-AWQ-4bit",  # 指定模型ID，通常与 --model 参数一致
            messages=messages,
            temperature=0.7,  # 控制生成文本的随机性
            max_tokens=200  # 限制生成文本的最大 token 数
        )

        # 4. 处理并打印响应
        print("API Response:")
        # 打印完整的响应对象 (可选)
        print(chat_completion)
        end_time = time.time()

        # 6. 计算响应时间
        response_time = end_time - start_time
        minutes, seconds = divmod(int(response_time), 60)

        # 如果需要更精确到小数点后几位的秒数，可以保留原始的秒数的小数部分
        seconds_fraction = response_time - int(response_time)
        formatted_response_time = '{:d}分 {:.2f}秒'.format(minutes, seconds + seconds_fraction)

        # 打印响应时间
        print(f"响应时间: {formatted_response_time}")

        # 提取并打印模型生成的回复内容
        if chat_completion.choices and chat_completion.choices[0].message:
            reply = chat_completion.choices[0].message.content
            print(f"Model Reply:\n{reply}")
        else:
            print("No content returned from model.")

        # 打印使用情况 (token 消耗)
        usage = chat_completion.usage
        if usage:
            print(
                f"\nToken Usage: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")

    except Exception as e:
        # 5. 处理潜在错误
        print(f"An error occurred while calling the API: {e}")


# --- 运行异步函数 ---
if __name__ == "__main__":
    asyncio.run(call_multimodal_api())