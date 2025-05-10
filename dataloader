import json
from pathlib import Path
from log.logger_utils import Logger

logger = Logger().get_logger()

class DataLoader:
    def __init__(self, config):
        self.dataset_path = Path(config['dataset_path'])  # 直接使用完整路径

    def load_dataset(self):
        logger.info(f"加载数据集 | 路径: {self.dataset_path}")
        try:
            dataset = self._load_jsonl(self.dataset_path)
            logger.info(f"加载完成 | 有效条目: {len(dataset)}")
            if not dataset:
                logger.warning("空数据集或加载失败")
            return dataset
        except Exception as e:
            logger.error(f"数据集加载异常: {str(e)}")
            raise

    def _load_jsonl(self, file_path):
        """加载JSONL格式数据集"""
        dataset = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        dataset.append(self._format_item(item))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
        return dataset

    def _format_item(self, raw_item):
        """统一数据格式"""
        return {
            "problem": raw_item.get("problem")
                     or raw_item.get("input")
                     or raw_item.get("question", ""),
            "solution": raw_item.get("solution")
                      or raw_item.get("answer", "")
        }
