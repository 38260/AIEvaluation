# 通用型AI模型评价工具使用说明

## 1. 项目概述
通用型AI模型评价系统，可通过外部API调用任意模型，对样本逐条评估并自动计算指标（准确率、精确率、召回率、F1、混淆矩阵），生成可追溯的报告与可视化。

## 功能特点
- 支持二分类任务
- 支持任意可通过API调用的AI模型
- 自动计算多种评估指标
- 支持断点续跑和中间结果保存
- 完整的日志记录和评估报告
- 支持中文字符显示（包括混淆矩阵图片中的中文标签）
- 提供可视化的混淆矩阵图片输出

## 3. 目录结构
```
AIEvaluation/
├─ main.py                      # 入口脚本
├─ config.ini                   # 主配置（API/Prompt/数据/评估）
├─ config copy.ini              # 备份示例
├─ requirements.txt             # 依赖列表
├─ system_prompt.txt            # 可选：系统提示词
├─ sample_data.xlsx             # 示例输入数据
├─ input/                       # 输入数据目录，放置待评估 xlsx/csv（如 sample1.xlsx）
├─ output/                      # 评估输出目录（结果表/报告/混淆矩阵PNG）
├─ assets/                      # 资源文件
├─ src/                         # 代码实现（如 ai_evaluator.py）
├─ docs/                        # 额外文档（如 PRD）

```
说明：
- 推荐数据放 `input/`，输出集中到 `output/`，日志在 `logs/`；若环境未自动建目录，请手动创建以避免写入失败。
- 若在 `config.ini` 中修改输出路径，请同步更新对应目录位置。

## 4. 环境与依赖
- Python 3.8+
- 依赖：pandas、requests、scikit-learn、numpy、matplotlib、seaborn、configparser

安装：
```bash
pip install -r requirements.txt
# 或
pip install pandas requests scikit-learn numpy matplotlib seaborn configparser
```

## 5. 配置文件详解（`config.ini`）
```ini
[API]
api_url = https://api.openai.com/v1/chat/completions  ; 模型接口地址
api_key = your-api-key-here                          ; 接口密钥
temperature = 0                                      ; 生成温度
timeout = 30                                         ; 请求超时时间（秒）
max_retry = 3                                        ; 失败重试次数

[Prompt]
system_prompt_path = system_prompt.txt               ; 系统提示词（可选）
user_prompt = 请分析以下文本内容并给出分类标签和理由：{content}。请以JSON格式返回，包含"label"和"reason"两个字段。

[Data]
input_xlsx_path = sample_data.xlsx                   ; 输入xlsx
output_xlsx_path = output_result.xlsx                ; 输出xlsx
label_field = label                                  ; AI返回中标签字段名
reason_field = reason                                ; AI返回中理由字段名

[Evaluation]
task_type = binary                                   ; binary
positive_label = 是                                  ; 二分类正例标签
```

**常用调优提示**
- 更换模型：调整 `api_url` 并在Prompt中适配接口格式
- 控制成本/稳定性：提高 `temperature` 生成多样性；降低可提升稳定性
- 超时与重试：数据量大或模型响应慢时适当提高 `timeout` 与 `max_retry`

## 6. 输入数据要求
Excel（.xlsx）需包含列：
- `ID`：样本唯一标识
- `content`：待分析文本
- `ground_truth`：真实标签

示例：`sample_data.xlsx`

## 7. 运行步骤
```bash
python main.py
```

运行中会每处理10条样本自动落盘临时结果，避免中途断线丢失。

## 8. 输出物
1) `output_xlsx_path`：原始数据附加 `label`、`reason`、预测对比  
2) `output_result_report.txt`：指标与混淆矩阵摘要  
3) 混淆矩阵PNG：中文标签可视化  
4) 日志：实时处理进度与异常

## 9. 评估指标
- 二分类：Accuracy、Precision、Recall、F1、2x2混淆矩阵
- 多分类：宏/微 Accuracy、Precision、Recall、F1，NxN混淆矩阵

## 10. 断点续跑与健壮性
- 网络或接口异常：自动按 `max_retry` 重试
- label字段缺失：标记“结果解析失败”
- 运行中断：使用中间结果文件自动续跑（无需手动介入）

## 11. 自定义模型/Prompt建议
- 确保模型响应JSON包含 `label` 与 `reason` 字段
- 多分类时在Prompt中告知完整标签集合，降低幻觉标签
- 如果接口格式不同，可在 `main.py` 中调整请求构造与解析逻辑

## 12. 常见问题
- 指标异常：确认 `ground_truth` 与模型返回的标签一致且大小写统一
- 混淆矩阵中文乱码：已默认加载中文字体，仍异常请检查本地字体环境
- 大数据量耗时：适当增大 `timeout`，或分批处理输入Excel

## 13. 注意事项
- 确保API密钥正确配置且具备调用权限
- 配置文件、数据文件请保持UTF-8编码
- 若调整 `label_field`/`reason_field`，需与模型返回字段保持一致
