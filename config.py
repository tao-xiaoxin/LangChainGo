# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from pathlib import Path

# 查找并加载.env文件
# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(ENV_PATH)


# 配置访问器
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8434/v1/")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "qwq:32b")


# 导出配置
config = Config()
