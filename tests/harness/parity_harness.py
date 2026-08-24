import os

# 在"HF 库被 import 之前"设置离线开关
# 保证 HF 不会联网校验/下载，reference 完全确定、可复现，harness 永不漂移
os.environ["HF_HUB_OFFLINE"] = 1

import transformers
