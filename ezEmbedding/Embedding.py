import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from openai import OpenAI


class bertEmbedding:

    def __init__(self, model_path, tokenizer_path, from_encoder="last4", pooling_method="mean"):

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.from_encoder = from_encoder
        self.pooling_method = pooling_method
        self.model = BertModel.from_pretrained(model_path, output_hidden_states=True)

    def get_tokens(self, text):
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
        return tokens

    def all_token_embeddings(self, text):
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        output = self.model(token_ids)

        last_hidden_state, self.cls_pooling, hidden_states = output[0], output[1], output[2]

        if self.from_encoder == "last1":
            token_embeddings = last_hidden_state.detach().numpy()[0, :, :]

        elif self.from_encoder == "last4":
            last4_states = hidden_states[8:12]
            token_embeddings = torch.cat(last4_states, dim=2).detach().numpy()[0, :, :]

        return token_embeddings

    def get_word_index(self, text, word):
        word_tokens = self.tokenizer.tokenize(word)
        text_tokens = self.tokenizer.tokenize(text)

        first_token_index = [index for (index, value) in enumerate(text_tokens) if value == word_tokens[0]]

        word_index = []
        for index in first_token_index:

            following_bool = []
            try:
                for order in range(len(word_tokens)):
                    following_bool.append(word_tokens[order] == text_tokens[index + order])

            except IndexError:
                following_bool = [False]

            if following_bool.count(False) == 0:
                new_index = np.array(range(index, index + len(word_tokens)))
                word_index.append(new_index)

        if word_index:
            return np.array(word_index)

        else:
            return np.array([])

    def get_word_embedding(self, text, word):
        word_indexes = self.get_word_index(text, word)

        if word_indexes.any():
            all_token_embeddings = self.all_token_embeddings(text)
            word_embeddings = []
            for word_index in word_indexes:

                word_embedding = [0]
                for token_index in word_index:
                    word_embedding = word_embedding + all_token_embeddings[token_index, :]
                word_embedding = word_embedding / len(word_index)

                word_embeddings.append(word_embedding)

            return np.array(word_embeddings)

        else:
            return np.array([])

    def get_sentence_embedding(self, text):
        all_token_embeddings = self.all_token_embeddings(text)

        if self.pooling_method == "mean":
            sentence_embedding = all_token_embeddings.mean(axis=0)
            return sentence_embedding

        if self.pooling_method == "max":
            sentence_embedding = all_token_embeddings.max(axis=0)
            return sentence_embedding

        elif self.pooling_method == "cls_head":
            return self.cls_pooling

        else:
            return None


class qwenEmbeddingLocal:

    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
        self.model = AutoModel.from_pretrained(model_path)

    def get_sentence_embedding(self, text):
        batch_dict = self.tokenizer(text, truncation=True, return_tensors="pt")
        batch_dict.to(self.model.device)

        outputs = self.model(**batch_dict)
        embedding = outputs.last_hidden_state.detach().numpy()[0, -1]

        return embedding


class EmbeddingAPI:

    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def get_sentence_embedding(self, text):
        completion = self.client.embeddings.create(model=self.model_name, input=text, encoding_format="float")
        embedding = np.array(completion.data[0].embedding)
        return embedding


if __name__ == "__main__":
    '''bertEmbedding用例'''

    # 建立一个bertEmbedding实例，属性包括模型的本地路径、tokenizer的本地路径、需要哪几个编码器的输出，以及句向量的池化方法
    # 一般来讲，tokenizer和模型在同一个文件夹中，注意tokenizer的本地路径是在模型的本地路径后面加入斜线"/"
    bertEmbedding = bertEmbedding("../bert_localpath", "../bert_localpath/",
                                  from_encoder="last4", pooling_method="mean")
    input_text_1 = "小明刚买了一个苹果手机，我也要买一个苹果。"
    input_text_2 = "小明今天去水果店买了苹果。"
    word = "苹果"

    # 获取句子里所有的token
    # 在开头和末尾分别加入识别句子开始和结束的标签，即[CLS]和[SEP]，所以token数会比实际输入的句子多两个
    # 返回一个列表
    tokens = bertEmbedding.get_tokens(input_text_1)

    # 获取每个token的嵌入
    # 返回一个二维数组，第一个维度是token，第二个维度是对应的向量维度
    # 如果实例属性中设置from_encoder="last4"(默认)，则使用后四个隐藏状态输出的拼接，得到3072维的向量
    # 如果只需要最后一个隐藏状态的输出，需要设置from_encoder="last1"，得到768维向量
    all_token_vecs = bertEmbedding.all_token_embeddings(input_text_1)

    # 获取所需要的词语在句子中的位置（第几个token）
    # 与上面同理，第一个token是[CLS]
    # 返回一个二维数组，第一个维度是词语出现的次数，第二个维度是每次出现的位置
    word_index = bertEmbedding.get_word_index(input_text_1, word)

    # 获取词语的嵌入，即词语中包含的每个token的平均向量
    # 返回一个二维数组，第一个维度是词语出现的次数，第二个维度是每次的嵌入向量的维度
    word_embeddings = bertEmbedding.get_word_embedding(input_text_1, word)

    # 获取整个句子的嵌入，返回一个单维数组
    # 如果实例属性中设置pooling_method="mean"(默认)，则返回所有token的平均向量，维数与token一致(768或3072)
    # 如果设置pooling_method="max"，则对每一维度，在所有token中取最大值，维数与token一致(768或3072)
    # 如果设置pooling_method="cls_head"，则返回transformers中默认的768维cls_head池化向量
    sentence_vec = bertEmbedding.get_sentence_embedding(input_text_1)

    '''qwenEmbeddingLocal用例'''

    # 建立一个qwenEmbeddingLocal实例，属性包括模型的本地路径、tokenizer的本地路径
    # tokenizer的本地路径是在模型的本地路径后面加入斜线"/"
    qwenEmbedding = qwenEmbeddingLocal("../qwen_localpath", "../qwen_localpath/", )
    input_texts = "一段测试文本"

    # 获取整个句子的嵌入向量，返回一个单维数组
    sentence_vec = qwenEmbedding.get_sentence_embedding(input_texts)

    '''EmbeddingAPI用例'''

    # 建立一个EmbeddingAPI实例，属性包括base_url、api_key和model_name
    # openai、qwen和其他嵌入模型，只要是嵌入模型，都可以使用，以下是以qwen的text-embedding-v4模型为例：
    EmbeddingAPI = EmbeddingAPI(api_key=os.environ.get("QWEN_API_KEY"),
                                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                model_name="text-embedding-v4")

    # 获取整个句子的嵌入向量，返回一个单维数组
    sentence_vec = EmbeddingAPI.get_sentence_embedding("一段测试文本")
