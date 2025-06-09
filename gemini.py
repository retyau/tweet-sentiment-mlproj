import os
import random
import re
import time
# 使用您指定的导入方式
from google import genai
from google.genai import types
from tqdm import tqdm

# --- 配置 ---
# 请在这里填入你的 Google API 密钥
GOOGLE_API_KEY = "AIzaSyBL5R5I8I8IXVrtqXhWzw-n5otLQ_91bkc"

# 数据集文件夹路径
DATA_FOLDER = 'twitter-datasets'

# 从每个类别中抽样的推文数量
SAMPLE_SIZE = 50

# --- 文件路径 ---
TRAIN_POS_FILE = os.path.join(DATA_FOLDER, 'train_pos.txt')
TRAIN_NEG_FILE = os.path.join(DATA_FOLDER, 'train_neg.txt')


def load_sample_tweets(filename: str, sample_size: int) -> list[str]:
    """从给定的文件中随机加载指定数量的推文。"""
    if not os.path.exists(filename):
        print(f"错误：找不到文件 {filename}")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) < sample_size:
        sample_size = len(lines)
    return random.sample(lines, sample_size)

# --- 严格按照您提供的格式重写的函数 ---
def generate_answer(client, sys, user, model="gemini-2.5-flash-preview-05-20", temperature=0):
    """
    严格遵循您提供的代码格式来调用 Gemini API。
    """
    try:
        # 使用 client.models.generate_content 的调用方式
        response = client.models.generate_content(
            model=model, # 使用函数传入的model参数
            config=types.GenerateContentConfig(
                temperature=temperature, # 使用函数传入的temperature参数
                # max_output_tokens=20000, # 可选，通常不需要这么大
                system_instruction=sys),
            contents=user
        )
        # 严格保留 (response.text) 的返回格式
        return (response.text)
    except Exception as e:
        print(f"\n调用API时发生错误: {e}")
        return "error"
# --- 函数重写结束 ---


def parse_llm_prediction(response_text: str) -> int:
    """解析LLM的响应文本，将其转换为数值标签。"""
    cleaned_response = response_text
    print(cleaned_response)
    if ":)" in cleaned_response:
        return 1
    if ":(" in cleaned_response:
        return -1
    return 0


def main():
    """主执行函数"""
    model_to_use = "gemini-2.5-flash-preview-05-20"
    print(f"正在使用 Google {model_to_use} 评估情感分类基线...")

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("\n错误：请在脚本中设置你的 GOOGLE_API_KEY。")
        return

    # 初始化 Google GenAI 客户端，严格遵循您的格式
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"初始化Google GenAI客户端失败: {e}")
        print("这可能是因为您安装的库版本不支持 genai.Client() 语法。")
        return

    # 1. 加载样本数据
    print(f"正在从积极和消极数据集中各加载 {SAMPLE_SIZE} 条推文...")
    pos_samples = load_sample_tweets(TRAIN_POS_FILE, SAMPLE_SIZE)
    neg_samples = load_sample_tweets(TRAIN_NEG_FILE, SAMPLE_SIZE)

    if not pos_samples or not neg_samples:
        print("加载数据失败，请检查文件路径是否正确。")
        return

    all_samples = [(tweet, 1) for tweet in pos_samples] + [(tweet, -1) for tweet in neg_samples]
    random.shuffle(all_samples)

    # 2. 迭代处理并评估
    correct_predictions = 0
    total_samples = len(all_samples)
    
    system_prompt = (
        "You are a high-frequency Twitter user and a master of global internet surfing. "
        "The following tweet originally contained either a :) or a :( smiley, but it has been removed. "
        "Your task is to predict whether it contained :) or :(. "
        "Output only the smiley itself and nothing else."
    )
    
    print(f"\n开始向 {model_to_use} 发送 {total_samples} 条推文进行分类...")
    
    for tweet, true_label in tqdm(all_samples, desc=f"{model_to_use} 评估进度"):
        # 调用严格按照您格式编写的函数
        llm_response = generate_answer(client, sys=system_prompt, user=[tweet], model=model_to_use)
        prediction = parse_llm_prediction(llm_response)

        if prediction == true_label:
            correct_predictions += 1
        print('===')
        print(correct_predictions)
        time.sleep(10) 

    # 3. 报告结果
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"\n--- {model_to_use} 基线评估结果 ---")
    print(f"总样本数: {total_samples}")
    print(f"正确预测数: {correct_predictions}")
    print(f"准确率: {accuracy:.2f}%")
    print("-------------------------")


if __name__ == "__main__":
    main()