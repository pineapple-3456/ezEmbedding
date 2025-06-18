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
        text_tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']

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


class EmbeddingAPI:

    def __init__(self, api_key, base_url, model_name, dimensions=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.dimensions = dimensions

    def get_sentence_embedding(self, text):
        if self.dimensions:
            completion = self.client.embeddings.create(model=self.model_name, input=text,
                                                       encoding_format="float", dimensions=self.dimensions)
        else:
            completion = self.client.embeddings.create(model=self.model_name, input=text,
                                                       encoding_format="float")
        embedding = np.array([completion.data[i].embedding for i in range(len(completion.data))])
        return embedding

