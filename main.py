"""
AI模型评估工具 - 主入口文件
"""
import logging
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ai_evaluator import AIModelEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='通用型AI模型评价工具')
    parser.add_argument('--config', type=str, default='config.ini', help='配置文件路径')
    
    args = parser.parse_args()
    
    evaluator = AIModelEvaluator(args.config)
    evaluator.run()


if __name__ == "__main__":
    main()