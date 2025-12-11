# 通用型AI模型评价工具使用说明

## 项目概述
本工具是一个通用型AI模型评价系统，通过调用外部AI模型API接口，逐条评估样本数据，自动计算评估指标（准确率、召回率、混淆矩阵），输出可追溯的评价结果。

## 功能特点
- 支持二分类任务
- 支持任意可通过API调用的AI模型
- 自动计算多种评估指标
- 支持断点续跑和中间结果保存
- 完整的日志记录和评估报告
- 支持中文字符显示（包括混淆矩阵图片中的中文标签）
- 提供可视化的混淆矩阵图片输出

## 依赖库
- pandas
- requests
- scikit-learn
- numpy
- matplotlib
- seaborn
- configparser

## 安装依赖
```bash
pip install pandas requests scikit-learn numpy matplotlib seaborn
```

或者使用requirements.txt文件：
```bash
pip install -r requirements.txt
```

## 配置文件说明

### config.ini
```ini
[API]
; API配置
api_url = https://api.openai.com/v1/chat/completions  ; AI接口地址
api_key = your-api-key-here                          ; 接口密钥
temperature = 0                                      ; 生成温度（默认0）
timeout = 30                                         ; 请求超时时间（默认30秒）
max_retry = 3                                        ; 失败重试次数（默认3次）

[Prompt]
; Prompt配置
system_prompt_path = system_prompt.txt               ; 系统提示词文件路径（可选）
user_prompt = 请分析以下文本内容并给出分类标签和理由：{content}。请以JSON格式返回，包含"label"和"reason"两个字段。

[Data]
; 数据配置
input_xlsx_path = sample_data.xlsx                   ; 输入xlsx路径
output_xlsx_path = output_result.xlsx                ; 输出xlsx路径
label_field = label                                  ; AI返回结果中"标签"的字段名（默认"label"）
reason_field = reason                                ; AI返回结果中"理由"的字段名（默认"reason"）

[Evaluation]
; 评估配置
task_type = binary                                   ; 任务类型："binary"二分类
positive_label = 是                                  ; 二分类正例标签（二分类任务）
```

## 输入数据格式
输入数据为Excel文件（.xlsx），必须包含以下3列：
- ID：样本唯一标识（字符串/数字，不可重复）
- content：待AI分析的文本内容（字符串，非空）
- ground_truth：样本真实标签（字符串/数字）

## 使用方法

### 1. 使用
```bash
python main.py 
```

## 输出结果
1. `output_xlsx_path` 指定的Excel文件：包含原始数据及新增的label、reason字段
2. `output_result_report.txt`：评估报告，包含各项指标和混淆矩阵
3. 混淆矩阵图片：以PNG格式输出的可视化混淆矩阵
4. 运行日志：实时显示处理进度和异常信息

## 评估指标
- **二分类任务**：
  - 准确率（Accuracy）
  - 精确率（Precision）
  - 召回率（Recall）
  - F1值
  - 混淆矩阵（2x2）

- **多分类任务**：
  - 宏观/微观准确率、精确率、召回率、F1值
  - 多分类混淆矩阵（NxN）

## 异常处理
- 接口超时/返回非200状态码：重试max_retry次，仍失败则标记该样本"调用失败"
- AI返回结果缺失label字段：标记"结果解析失败"
- 每处理10条样本，自动保存一次中间结果到临时文件

## 注意事项
- 确保API密钥正确配置
- 配置文件编码为UTF-8
- 支持中文标签，无编码乱码问题
- 代码兼容Python 3.8+版本
- 混淆矩阵图片已支持中文字符显示
