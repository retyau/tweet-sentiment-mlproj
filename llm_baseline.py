import os
import random
import re
import time
from openai import OpenAI
from tqdm import tqdm

# --- 配置 ---
# 请在这里填入你的DeepSeek API密钥
# 重要：不要将你的密钥直接分享或上传到公共代码库
DEEPSEEK_API_KEY = "sk-5a8abb738b4341389d997af914bc0d51"

# 数据集文件夹路径
DATA_FOLDER = 'twitter-datasets'

# 从每个类别中抽样的推文数量，以进行评估
# 注意：数量越多，API调用成本越高，耗时越长
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
    # 确保抽样数量不超过文件总行数
    if len(lines) < sample_size:
        sample_size = len(lines)
    return random.sample(lines, sample_size)


def get_llm_response(client: OpenAI, tweet_text: str) -> str:
    """
    使用指定的LLM客户端和提示词来对单条推文进行情感分类。

    参数:
        client (OpenAI): 初始化好的OpenAI客户端实例。
        tweet_text (str): 需要分类的推文内容。

    返回:
        str: LLM返回的原始文本响应。
    """
    # 这是我们给LLM的指示。它被设计成让LLM只返回一个单词。
    system_prompt = (
        "You are an expert in sentiment analysis. "
        "Your task is to classify the sentiment of the following tweet. "
        "Please respond with a single word: either 'positive' or 'negative'."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": tweet_text}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0,  # 温度设为0，以获得更确定性的、可复现的结果
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n调用API时发生错误: {e}")
        return "error"


def parse_llm_prediction(response_text: str) -> int:
    """
    解析LLM的响应文本，将其转换为数值标签。

    返回:
        1: 如果响应为 "positive"。
        -1: 如果响应为 "negative"。
        0: 如果无法解析或发生错误。
    """
    cleaned_response = re.sub(r'[^a-zA-Z]', '', response_text.lower())
    if "positive" in cleaned_response:
        return 1
    if "negative" in cleaned_response:
        return -1
    return 0 # 表示无法识别的响应


def main():
    """主执行函数"""
    print("正在使用大语言模型（LLM）评估情感分类基线...")

    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
        print("\n错误：请在脚本中设置你的DEEPSEEK_API_KEY。")
        return

    # 初始化LLM客户端
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    except Exception as e:
        print(f"初始化OpenAI客户端失败: {e}")
        return

    # 1. 加载样本数据
    print(f"正在从积极和消极数据集中各加载 {SAMPLE_SIZE} 条推文...")
    pos_samples = load_sample_tweets(TRAIN_POS_FILE, SAMPLE_SIZE)
    neg_samples = load_sample_tweets(TRAIN_NEG_FILE, SAMPLE_SIZE)

    if not pos_samples or not neg_samples:
        print("加载数据失败，请检查文件路径是否正确。")
        return

    # 将样本和它们的真实标签合并
    # 真实标签：1 代表积极, -1 代表消极
    all_samples = [(tweet, 1) for tweet in pos_samples] + [(tweet, -1) for tweet in neg_samples]
    random.shuffle(all_samples) # 打乱顺序

    # 2. 迭代处理并评估
    correct_predictions = 0
    total_samples = len(all_samples)
    
    print(f"\n开始向LLM发送 {total_samples} 条推文进行分类...")
    
    for tweet, true_label in tqdm(all_samples, desc="LLM评估进度"):
        # 调用LLM进行分类
        llm_response = get_llm_response(client, tweet)
        
        # 解析预测结果
        prediction = parse_llm_prediction(llm_response)

        # 检查预测是否正确
        if prediction == true_label:
            correct_predictions += 1
        print(correct_predictions)
        print('='*5)
        # 为了避免触发API的速率限制，每次调用后暂停一小段时间
        time.sleep(0.01) 

    # 3. 报告结果
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n--- LLM基线评估结果 ---")
    print(f"总样本数: {total_samples}")
    print(f"正确预测数: {correct_predictions}")
    print(f"准确率: {accuracy:.2f}%")
    print("------------------------")


if __name__ == "__main__":
    main()