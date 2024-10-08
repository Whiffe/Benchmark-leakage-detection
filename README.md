# 1 资料

我复现的源码: [https://github.com/Whiffe/Benchmark-leakage-detection/tree/main](https://github.com/Whiffe/Benchmark-leakage-detection/tree/main)

官网源码：[https://github.com/nishiwen1214/Benchmark-leakage-detection](https://github.com/nishiwen1214/Benchmark-leakage-detection)

论文：[https://arxiv.org/pdf/2409.01790](https://arxiv.org/pdf/2409.01790)

论文翻译：[arxiv-2024 Training on the Benchmark Is Not All You Need](https://blog.csdn.net/WhiffeYF/article/details/142239317)

b站复现视频：[https://www.bilibili.com/video/BV1Kb2WYNE2E/](https://www.bilibili.com/video/BV1Kb2WYNE2E/)

CSDN：[https://blog.csdn.net/WhiffeYF/article/details/142763060](https://blog.csdn.net/WhiffeYF/article/details/142763060)
# 2 我的总结
这篇论文还是非常通俗易懂的，就是交换题目的选项顺序，来计算logprobs，查看是否有异常值。
# 3 复现源码


## 首先你需要有gpt的api接口

```python
# 设置API key和API的基础URL，用于调用 OpenAI 接口
API_KEY = ""  # 替换为你的 API key
BASE_URL = ""  # 替换为API的基本URL
```

## 安装：

```python
pip install openai
```
## 执行指令
数据集的生成：

```bash
python data_process.py \
    --data_dir  ./data/example_data.json \
    --save_dir ./data
```
计算logprobs：

```bash
python inference_logprobs.py --permutations_data_dir ./data/permutations_data2.json \
     --save_dir ./data
```

## 源码
### data_process.py

```python
'''
python data_process.py \
    --data_dir  ./data/example_data.json \
    --save_dir ./data
'''

import json
import itertools
import argparse

'''
{
   'option': {
   'A': '由间充质增生形成', 
   'B': '人胚第4周出现', 
   'C': '相邻鳃弓之间为鳃沟',
    'D': '共5对鳃弓'
    },
   'question': '下列有关鳃弓的描述，错误的是'
}
'''
parser = argparse.ArgumentParser(prog='data_process', description='')
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()
with open(args.data_dir, 'r') as file:
    data_list = json.load(file)

# 定义你的字符列表
chars = ['A', 'B', 'C', 'D']

# 使用itertools.permutations生成所有排列yy
permutations_list = list(itertools.permutations(chars))
result = []

for index, row in enumerate(data_list):

    for perm in permutations_list:
        instruction = {
            "instruction":
f"""
{row['question']}:
A:{row['option'][perm[0]]}
B:{row["option"][perm[1]]}
C:{row["option"][perm[2]]} 
D:{row["option"][perm[3]]}
""",
        }
        result.append(instruction)

with open(f"{args.save_dir}/permutations_data.json", 'w') as json_file:
    json.dump(result, json_file, indent=4, ensure_ascii=False)
```

### inference_logprobs.py

```python
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
```


# 4 结果
数据集的打乱生成，就是对4个选项进行排列组合。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/298cc5ea01d4451fa8719983c5e66cb8.png)

对打乱后的数据计算logprobs
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bff5bf19661543fea74c7810b5951072.png)

```python
[
    -8.13673193681245,
    -9.511466482160149,
    -5.386469065935929,
    -9.432328614854779,
    -5.405103636728939,
    -9.08097951506345,
    -6.560878102447153,
    -5.13767155106839,
    -5.49210427956123,
    -53.71251137077802,
    -6.3262343066056,
    -8.972361489357953,
    -8.017449890078197,
    -4.852460129236841,
    -7.9068465161303,
    -5.729137238647233,
    -9.92944942808605,
    -15.058123669689795,
    -10024.199599759702,
    -8.1004607337965,
    -12.517808548910251,
    -33.58417227999325,
    -7.63461314750679,
    -13.655588575448178,
    -30.238195388717422,
    -22.205287599773747,
    -15.618651239780414,
    -16.08750962605555,
    -25.21407782270761,
    -19.756035716175465,
    -14.75928077,
    -18.81754370852463,
    -19.75940019079578,
    -22.26305767797671,
    -21.812592662996376,
    -39.88223290471884,
    -9.58445245659858,
    -25.465287367872612,
    -2.19529201559165,
    -21.777878197784858,
    -20.38349776538606,
    -0.16652563152008001,
    -3.18791372228803,
    -0.17106814774657997,
    -1.5009587775304096,
    -29.762072239900135,
    -30.634590575760512,
    -21.8722806283808
]
```
