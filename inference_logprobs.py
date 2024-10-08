'''
python inference_logprobs.py --permutations_data_dir ./data/permutations_data2.json \
     --save_dir ./data
'''
import json
import tqdm
import argparse
from openai import OpenAI

# 定义命令行参数
parser = argparse.ArgumentParser(prog='logprobs', description='')
parser.add_argument("--permutations_data_dir", type=str, help="Path to the input data (JSON)")
parser.add_argument("--save_dir", type=str, help="Directory to save the results")
args = parser.parse_args()

# 初始化OpenAI API客户端
client = OpenAI(
    api_key='xxx', #将这里换成你在api keys
    base_url="xxx"  # 替换为你要访问的 API 入口
)

def find_indices(lst, value):
    indices = []
    for i, elem in enumerate(lst):
        if (elem == value and len(lst[i + 1]) != 0 and lst[i + 1][0] == ":") or elem == 'A:':
            indices.append(i)
            return indices
    return indices

# 使用OpenAI API获取logprobs
def score(prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",  # 选择模型
        logprobs=True  # 获取logprobs
    )
    
    # 提取tokens及其logprobs
    input_tokens = [item.token for item in response.choices[0].logprobs.content]
    input_logprobs = [item.logprob for item in response.choices[0].logprobs.content]
    
    # 打印以查看提取是否正确
    # print("Tokens:", input_tokens)
    # print("Logprobs:", input_logprobs)
    
    # 找到 'A' 的索引
    index = find_indices(input_tokens, 'A')
    
    return input_tokens, input_logprobs, index[0] if index else 0

def display(prompt):
    input_tokens, input_logprobs, index = score(prompt)
    all_logprobs = 0
    # 累积从index开始的logprobs
    for i in range(index, len(input_logprobs)):
        all_logprobs += input_logprobs[i]
    return all_logprobs

# 加载输入数据
with open(args.permutations_data_dir, 'r') as file:
    datas = json.load(file)

logprobs_list = []

# 对每个数据计算logprobs
for index, data in enumerate(tqdm.tqdm(datas)):
    result = display(data["instruction"])
    logprobs_list.append(result)

# 保存logprobs结果
with open(f"{args.save_dir}/logprobs.json", 'w') as json_file:
    json.dump(logprobs_list, json_file, indent=4, ensure_ascii=False)
