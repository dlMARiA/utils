from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


class TextToVectorConverter:
    def __init__(self, text, save_path=None, model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        初始化文本向量转换器。

        :param text: 需要转换为向量的输入文本。
        :param save_path: （可选）向量保存路径，默认为None不保存。
        :param model_name: 用于文本向量化的模型名称，默认为'sentence-transformers/all-mpnet-base-v2'。
        """
        self.text = text
        self.save_path = save_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vector = None  # 存储生成的向量
        self.vectorize_and_save()

    def vectorize_and_save(self):
        """
        将文本转换为向量，并根据save_path决定是否保存。
        """
        try:
            # 确保生成的向量是numpy数组
            self.vector = np.array(self.embeddings.embed_query(self.text))
            # 仅在提供保存路径时保存文件
            if self.save_path:
                np.save(self.save_path, self.vector)
                print(f"向量已保存至 {self.save_path}")
        except Exception as e:
            print(f"向量化或保存过程中发生错误: {e}")