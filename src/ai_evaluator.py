"""
AI模型评估工具 - 核心功能模块
提供AI模型评估的核心功能，包括指标计算、结果可视化等
"""

import pandas as pd
import requests
import json
import time
import logging
import configparser
import os
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AIModelEvaluator:
    """
    通用型AI模型评价工具
    """
    
    def __init__(self, config_path: str):
        """
        初始化评价器
        :param config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.validate_config()
        self.output_dir = None  # 初始化时未设置输出目录
        
    def load_config(self, config_path: str) -> configparser.ConfigParser:
        """
        加载配置文件
        :param config_path: 配置文件路径
        :return: 配置对象
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        return config
        
    def validate_config(self):
        """
        校验配置项
        """
        logger = logging.getLogger(__name__)
        required_sections = ['API', 'Prompt', 'Data', 'Evaluation']
        for section in required_sections:
            if not self.config.has_section(section):
                raise ValueError(f"配置文件缺少必要节: [{section}]")
                
        # API配置校验
        api_required = ['api_url', 'api_key']
        for item in api_required:
            if not self.config.has_option('API', item):
                raise ValueError(f"API配置缺少必要项: {item}")
                
        # Prompt配置校验
        prompt_required = ['user_prompt']
        for item in prompt_required:
            if not self.config.has_option('Prompt', item):
                raise ValueError(f"Prompt配置缺少必要项: {item}")
                
        # Data配置校验
        data_required = ['input_xlsx_path', 'output_xlsx_path']
        for item in data_required:
            if not self.config.has_option('Data', item):
                raise ValueError(f"数据配置缺少必要项: {item}")
                
        # Evaluation配置校验
        if not self.config.has_option('Evaluation', 'task_type'):
            raise ValueError("评估配置缺少必要项: task_type")
            
        task_type = self.config.get('Evaluation', 'task_type', fallback='binary')
        if task_type != 'binary':
            raise ValueError(f"当前仅支持二分类任务，task_type 必须为 'binary'，当前值: {task_type}")
            
        if task_type == 'binary' and not self.config.has_option('Evaluation', 'positive_label'):
            raise ValueError("二分类任务必须配置 positive_label")
            
        if task_type == 'binary' and not self.config.has_option('Evaluation', 'negative_label'):
            # 如果没有配置negative_label，默认为"否"
            logger.info("未配置negative_label，使用默认值'否'")
            
        logger.info("配置校验完成")
        
    def validate_input_data(self, file_path: str) -> bool:
        """
        校验输入数据文件
        :param file_path: 输入文件路径
        :return: 是否有效
        """
        logger = logging.getLogger(__name__)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"输入文件不存在: {file_path}")
            
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"无法读取Excel文件: {e}")
            
        required_columns = ['ID', 'content', 'ground_truth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入文件缺少必要列: {', '.join(missing_columns)}")
            
        if df['ID'].duplicated().any():
            logger.warning("发现重复的ID，将保留第一条记录")
            
        if df['content'].isna().any() or (df['content'] == '').any():
            logger.warning("发现空的content字段")
            
        if df['ground_truth'].isna().any() or (df['ground_truth'] == '').any():
            logger.warning("发现空的ground_truth字段")
            
        logger.info(f"输入数据校验完成，共 {len(df)} 条记录")
        return True
        
    def read_sample_data(self, file_path: str) -> pd.DataFrame:
        """
        读取样本数据
        :param file_path: 文件路径
        :return: DataFrame
        """
        logger = logging.getLogger(__name__)
        df = pd.read_excel(file_path)
        
        # 按ID去重，保留第一条
        df = df.drop_duplicates(subset=['ID'], keep='first')
        
        # 打印标签分布
        label_counts = df['ground_truth'].value_counts()
        logger.info(f"标签分布:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} 条")
            
        logger.info(f"总计 {len(df)} 条样本")
        return df
    
    def call_ai_api(self, prompt: str) -> Dict[str, Any]:
        """
        调用AI API接口
        :param prompt: 提示词
        :return: API返回结果
        """
        logger = logging.getLogger(__name__)
        
        # 实际API调用逻辑
        api_url = self.config.get('API', 'api_url')
        api_key = self.config.get('API', 'api_key')
        temperature = self.config.getfloat('API', 'temperature', fallback=0)
        timeout = self.config.getint('API', 'timeout', fallback=30)
        max_retry = self.config.getint('API', 'max_retry', fallback=3)
        model = self.config.get('API', 'model', fallback='gpt-4o:2025-01-01-preview')  # 从配置获取模型
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 构建请求体 - 根据不同API格式进行调整
        # 这里是OpenAI格式，可根据实际API调整
        
        
        # 读取系统提示词
        system_prompt_path = self.config.get('Prompt', 'system_prompt_path', fallback='')
        if system_prompt_path and os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_content = f.read()


        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content}, # 系统提示词
                {"role": "user", "content": prompt} # 用户提示词
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}  # 要求JSON格式返回
        }
        
        for attempt in range(max_retry + 1):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
               
                if response.status_code == 200:
                    result = response.json()
                    
                    # 提取AI返回的内容
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print(content)
                    try:
                        # 尝试解析JSON格式的返回
                        parsed_result = json.loads(content)
                        label_field = self.config.get('Data', 'label_field', fallback='label')
                        reason_field = self.config.get('Data', 'reason_field', fallback='reason')
                        
                        label = parsed_result.get(label_field, '无有效标签')
                        reason = parsed_result.get(reason_field, '无有效理由')
                        
                        return {
                            'label': label,
                            'reason': reason,
                            'status': 'success'
                        }
                    except json.JSONDecodeError:
                        # 如果不是JSON格式，尝试其他解析方式
                        # 这里可以根据实际API返回格式进行调整
                        return {
                            'label': '无有效标签',
                            'reason': content[:200] + '...' if len(content) > 200 else content,
                            'status': 'parse_error'
                        }
                else:
                    logger.warning(f"API返回状态码 {response.status_code}: {response.text}")
                    if attempt < max_retry:
                        time.sleep(2 ** attempt)  # 指数退避
                        continue
                    else:
                        return {
                            'label': 'API调用失败',
                            'reason': f'HTTP {response.status_code}',
                            'status': 'api_error'
                        }
                        
            except requests.exceptions.Timeout:
                logger.warning(f"API调用超时 (尝试 {attempt + 1}/{max_retry + 1})")
                if attempt < max_retry:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    return {
                        'label': 'API调用失败',
                        'reason': '请求超时',
                        'status': 'timeout_error'
                    }
            except Exception as e:
                logger.error(f"API调用异常: {e}")
                if attempt < max_retry:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    return {
                        'label': 'API调用失败',
                        'reason': str(e),
                        'status': 'exception_error'
                    }
    
    def process_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理样本数据，逐条调用AI API
        :param df: 样本数据
        :return: 包含预测结果的DataFrame
        """
        logger = logging.getLogger(__name__)
        # 添加预测结果列
        df = df.copy()
        df['label'] = ''
        df['reason'] = ''
        
        total_samples = len(df)
        logger.info(f"开始处理 {total_samples} 条样本")
        
        # 确保输出目录已经设置
        if self.output_dir is None:
            # 如果输出目录未设置，使用默认输出目录
            base_output_dir = os.path.dirname(self.config.get('Data', 'output_xlsx_path'))
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(base_output_dir, timestamp)
            os.makedirs(self.output_dir, exist_ok=True)
        
        for idx, row in df.iterrows():
            content = row['content']
            
            # 构建用户提示词
            user_prompt = self.config.get('Prompt', 'user_prompt')
            prompt = user_prompt.format(content=content)
            
            # 调用AI API
            result = self.call_ai_api(prompt)
            
            df.at[idx, 'label'] = result['label']
            df.at[idx, 'reason'] = result['reason']
            
            # 每处理10条样本，保存一次中间结果
            if (idx + 1) % 10 == 0 or (idx + 1) == total_samples:
                temp_output_path = os.path.join(self.output_dir, os.path.basename(self.config.get('Data', 'output_xlsx_path')).replace('.xlsx', '_temp.xlsx'))
                # 确保输出目录存在
                os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
                df.to_excel(temp_output_path, index=False)
                logger.info(f"已处理 {idx + 1}/{total_samples} 条样本，临时结果已保存")
                
            logger.info(f"已处理 {idx + 1}/{total_samples} 条样本，当前ID: {row['ID']}")
            
        logger.info("样本处理完成")
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算评估指标
        :param df: 包含预测结果的数据
        :return: 评估指标字典
        """
        logger = logging.getLogger(__name__)
        # 过滤掉调用失败/解析失败的样本
        valid_df = df[
            (df['label'] != 'API调用失败') & 
            (df['label'] != '无有效标签') & 
            (df['label'] != '') &
            (df['label'].notna())
        ].copy()
        
        if len(valid_df) == 0:
            logger.warning("没有有效的预测结果用于计算指标")
            return {}
            
        logger.info(f"有效样本数: {len(valid_df)} / 总样本数: {len(df)}")
        
        # 获取标签列
        y_true = valid_df['ground_truth'].tolist()
        y_pred = valid_df['label'].tolist()
        
        metrics = {
            'total_samples': len(df),
            'valid_samples': len(valid_df),
            'invalid_samples': len(df) - len(valid_df)
        }
        
        # 二分类指标
        positive_label = self.config.get('Evaluation', 'positive_label')
        
        # 检测标签的实际类型，确保类型匹配
        # 如果标签是数字类型，将positive_label转换为数字；否则保持字符串
        if len(y_true) > 0:
            sample_label = y_true[0]
            if isinstance(sample_label, (int, float)) or (isinstance(sample_label, str) and sample_label.isdigit()):
                try:
                    # 尝试将positive_label转换为数字
                    positive_label = int(positive_label) if '.' not in str(positive_label) else float(positive_label)
                except (ValueError, TypeError):
                    # 如果转换失败，保持原样
                    pass
        
        # 将标签转换为二分类格式（使用类型安全的比较）
        def convert_to_binary(label, pos_label):
            """将标签转换为二分类格式，支持类型转换"""
            # 尝试直接比较
            if label == pos_label:
                return 1
            # 尝试类型转换后比较
            try:
                if isinstance(label, (int, float)) and isinstance(pos_label, str):
                    if str(label) == pos_label or label == float(pos_label):
                        return 1
                elif isinstance(label, str) and isinstance(pos_label, (int, float)):
                    if label == str(pos_label) or float(label) == pos_label:
                        return 1
            except (ValueError, TypeError):
                pass
            return 0
        
        y_true_binary = [convert_to_binary(label, positive_label) for label in y_true]
        y_pred_binary = [convert_to_binary(label, positive_label) for label in y_pred]
        
        # 计算指标
        metrics['accuracy'] = round(accuracy_score(y_true_binary, y_pred_binary), 4)
        metrics['precision'] = round(precision_score(y_true_binary, y_pred_binary, zero_division=0), 4)
        metrics['recall'] = round(recall_score(y_true_binary, y_pred_binary, zero_division=0), 4)
        
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1'] = round(2 * (metrics['precision'] * metrics['recall']) / 
                               (metrics['precision'] + metrics['recall']), 4)
        else:
            metrics['f1'] = 0.0
        
        # 获取负向标签
        negative_label = self.config.get('Evaluation', 'negative_label', fallback='否')
        
        # 混淆矩阵
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        metrics['confusion_matrix'] = cm
        metrics['confusion_matrix_labels'] = [f'预测_{positive_label}', f'预测_{negative_label}']
        metrics['confusion_matrix_real_labels'] = [f'真实_{positive_label}', f'真实_{negative_label}']
        
        return metrics
    
    def plot_confusion_matrix(self, cm, real_labels, pred_labels):
        """
        绘制混淆矩阵图片
        :param cm: 混淆矩阵
        :param real_labels: 真实标签列表
        :param pred_labels: 预测标签列表
        """
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
            plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pred_labels, yticklabels=real_labels)
            plt.title('混淆矩阵 - 二分类任务', fontsize=16)
            plt.xlabel('预测标签', fontsize=12)
            plt.ylabel('真实标签', fontsize=12)
            plt.tight_layout()
            
            # 生成图片文件名
            import time
            timestamp = int(time.time())
            
            # 使用输出目录
            if self.output_dir:
                image_path = os.path.join(self.output_dir, f'output_result_confusion_matrix_{timestamp}.png')
            else:
                # 如果没有设置输出目录，使用原始路径
                output_path = self.config.get('Data', 'output_xlsx_path')
                image_path = output_path.replace('.xlsx', f'_confusion_matrix_{timestamp}.png')
            
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形以释放内存
            
            logger = logging.getLogger(__name__)
            logger.info(f"混淆矩阵图片已保存至: {image_path}")
            
            # 自动打开混淆矩阵图片
            import subprocess
            import platform
            if platform.system() == 'Windows':
                subprocess.run(['start', image_path], shell=True)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', image_path])
            else:  # Linux
                subprocess.run(['xdg-open', image_path])
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"生成混淆矩阵图片时出错: {e}")
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """
        打印评估指标
        :param metrics: 评估指标
        """
        logger = logging.getLogger(__name__)
        if not metrics:
            logger.warning("没有指标可以打印")
            return
            
        print("\n" + "="*50)
        print("评估结果汇总")
        print("="*50)
        print(f"总样本数: {metrics['total_samples']}")
        print(f"有效样本数: {metrics['valid_samples']}")
        print(f"无效样本数: {metrics['invalid_samples']}")
        
        print(f"准确率 (Accuracy): {metrics.get('accuracy', 'N/A')}")
        print(f"精确率 (Precision): {metrics.get('precision', 'N/A')}")
        print(f"召回率 (Recall): {metrics.get('recall', 'N/A')}")
        print(f"F1值: {metrics.get('f1', 'N/A')}")
        
        # 打印混淆矩阵
        if 'confusion_matrix' in metrics:
            print("\n混淆矩阵:")
            cm = metrics['confusion_matrix']
            real_labels = metrics['confusion_matrix_real_labels']
            pred_labels = metrics['confusion_matrix_labels']
            
            # 打印表头
            header = "真实\\预测" + "\t"
            for label in pred_labels:
                header += f"{label}\t"
            print(header)
            
            # 打印矩阵内容
            for i, real_label in enumerate(real_labels):
                row_str = f"{real_label}\t"
                for j in range(len(pred_labels)):
                    row_str += f"{cm[i][j]}\t"
                print(row_str)
            
            # 生成混淆矩阵图片
            self.plot_confusion_matrix(cm, real_labels, pred_labels)
        
        print("="*50)
    
    def save_evaluation_report(self, metrics: Dict[str, Any], output_path: str):
        """
        保存评估报告
        :param metrics: 评估指标
        :param output_path: 输出路径
        """
        logger = logging.getLogger(__name__)
        
        # 使用输出目录
        if self.output_dir:
            report_path = os.path.join(self.output_dir, os.path.basename(output_path).replace('.xlsx', '_report.txt'))
        else:
            report_path = output_path.replace('.xlsx', '_report.txt')
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AI模型评估报告\n")
            f.write("="*50 + "\n")
            f.write(f"总样本数: {metrics.get('total_samples', 'N/A')}\n")
            f.write(f"有效样本数: {metrics.get('valid_samples', 'N/A')}\n")
            f.write(f"无效样本数: {metrics.get('invalid_samples', 'N/A')}\n")
            
            f.write(f"准确率 (Accuracy): {metrics.get('accuracy', 'N/A')}\n")
            f.write(f"精确率 (Precision): {metrics.get('precision', 'N/A')}\n")
            f.write(f"召回率 (Recall): {metrics.get('recall', 'N/A')}\n")
            f.write(f"F1值: {metrics.get('f1', 'N/A')}\n")
            
            # 写入混淆矩阵
            if 'confusion_matrix' in metrics:
                f.write("\n混淆矩阵:\n")
                cm = metrics['confusion_matrix']
                real_labels = metrics['confusion_matrix_real_labels']
                pred_labels = metrics['confusion_matrix_labels']
                
                # 写入表头
                header = "真实\\预测" + "\t"
                for label in pred_labels:
                    header += f"{label}\t"
                f.write(header + "\n")
                
                # 写入矩阵内容
                for i, real_label in enumerate(real_labels):
                    row_str = f"{real_label}\t"
                    for j in range(len(pred_labels)):
                        row_str += f"{cm[i][j]}\t"
                    f.write(row_str + "\n")
        
        logger.info(f"评估报告已保存至: {report_path}")
    
    def run(self):
        """
        执行完整评价流程
        """
        import time
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        logger.info("开始AI模型评价流程")
        
        # 步骤0: 创建以当前时间命名的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.dirname(self.config.get('Data', 'output_xlsx_path'))
        self.output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"输出目录创建: {self.output_dir}")
        
        # 步骤1: 配置加载与校验
        logger.info("步骤1: 配置加载与校验完成")
        
        # 步骤2: 样本数据读取
        input_xlsx_path = self.config.get('Data', 'input_xlsx_path')
        self.validate_input_data(input_xlsx_path)
        df = self.read_sample_data(input_xlsx_path)
        logger.info("步骤2: 样本数据读取完成")
        
        # 步骤3: AI接口批量调用
        df_result = self.process_samples(df)
        logger.info("步骤3: AI接口批量调用完成")
        
        # 步骤4: 结果整合
        output_xlsx_path = os.path.join(self.output_dir, os.path.basename(self.config.get('Data', 'output_xlsx_path')))
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_xlsx_path), exist_ok=True)
        df_result.to_excel(output_xlsx_path, index=False)
        
        # 删除临时文件，只在主文件成功保存后才删除临时文件
        temp_output_path = output_xlsx_path.replace('.xlsx', '_temp.xlsx')
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
                logger.info(f"临时文件已删除: {temp_output_path}")
            except OSError as e:
                logger.warning(f"删除临时文件失败: {e}")
            
        logger.info(f"步骤4: 结果已保存至 {output_xlsx_path}")
        
        # 步骤5: 评估指标计算与输出
        metrics = self.calculate_metrics(df_result)
        self.print_metrics(metrics)
        self.save_evaluation_report(metrics, output_xlsx_path)
        logger.info("步骤5: 评估指标计算与输出完成")
        
        logger.info("AI模型评价流程完成")


def evaluate_model(config_path: str):
    """
    便捷函数：直接运行模型评估
    :param config_path: 配置文件路径
    :return: 评估结果
    """
    evaluator = AIModelEvaluator(config_path)
    evaluator.run()
    return evaluator


if __name__ == "__main__":
    # 保留原main.py的命令行功能
    import argparse
    parser = argparse.ArgumentParser(description='通用型AI模型评价工具')
    parser.add_argument('--config', type=str, default='config.ini', help='配置文件路径')
    
    args = parser.parse_args()
    
    evaluator = AIModelEvaluator(args.config)
    evaluator.run()
