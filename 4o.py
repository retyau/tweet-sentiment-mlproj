import os
import random
import re
import time
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
# 请在这里填入你的OpenAI API密钥
# 为了安全，建议设置环境变量 OPENAI_API_KEY，SDK会自动读取
# 如果设置了环境变量，可以将下一行代码改为 OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "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"

# 数据集文件夹路径
DATA_FOLDER = 'twitter-datasets'

# 从每个类别中抽样的推文数量，以进行评估
SAMPLE_SIZE = 50

# --- File Paths ---
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


def get_gpt4o_response(client: OpenAI, tweet_text: str) -> str:
    """
    使用 gpt-4o 模型对单条推文进行情感分类。

    参数:
        client (OpenAI): 初始化好的OpenAI客户端实例。
        tweet_text (str): 需要分类的推文内容。

    返回:
        str: LLM返回的原始文本响应。
    """
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
            model="gpt-4o",  # <--- 使用 gpt-4o 模型
            messages=messages,
            temperature=0,
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
    return 0


def main():
    """主执行函数"""
    print("正在使用 OpenAI GPT-4o 评估情感分类基线...")

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("\n错误：请在脚本中设置你的 OPENAI_API_KEY，或将其设置为环境变量。")
        return

    # 初始化OpenAI客户端
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # 如果你设置了环境变量，可以使用下面这行代码代替上面那行
        # client = OpenAI()
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

    all_samples = [(tweet, 1) for tweet in pos_samples] + [(tweet, -1) for tweet in neg_samples]
    random.shuffle(all_samples)

    # 2. 迭代处理并评估
    correct_predictions = 0
    total_samples = len(all_samples)
    
    print(f"\n开始向 GPT-4o 发送 {total_samples} 条推文进行分类...")
    
    for tweet, true_label in tqdm(all_samples, desc="GPT-4o 评估进度"):
        llm_response = get_gpt4o_response(client, tweet)
        prediction = parse_llm_prediction(llm_response)

        if prediction == true_label:
            correct_predictions += 1
        print('===')
        print(correct_predictions)
        # 避免触发API速率限制
        time.sleep(0.01) 

    # 3. 报告结果
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n--- GPT-4o 基线评估结果 ---")
    print(f"总样本数: {total_samples}")
    print(f"正确预测数: {correct_predictions}")
    print(f"准确率: {accuracy:.2f}%")
    print("-------------------------")


if __name__ == "__main__":
    main()