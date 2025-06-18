# ezEmbedding

这个包是我的文本嵌入工具箱，也就是封装了我自己研究中需要反复使用的功能：即更方便地使用大模型获取文本嵌入，以及进行后续的分析。通过这个包，可以把一般的文本嵌入过程简化为两行代码，即对模型类进行实例化和调用实例方法得到嵌入向量的数组。

会随缘更新，取决于之后研究中需要用到什么功能。另外由于本人水平有限，并非计算机或机器学习科班出身，不能保证这些代码在任何情况下都输出正确结果，欢迎指正。

可以通过pip install git下载，同时会自动下载依赖的包，包括：`numpy`, `torch`, `transformers`和`openai`    (参见requirement.txt)

```
pip install git+https://github.com/pineapple-3456/ezEmbedding.git
```

如果无法下载，也可以将仓库克隆到本地，将其中的ezEmbedding文件夹拖入项目文件夹，然后下载requirement.txt中的依赖包。

## Embedding

Embedding模块的功能是提取文本嵌入，包括两个类：

```python
bertEmbedding(model_path="", tokenizer_path="", from_encoder="last4", pooling_method="mean")
```

使用Bert提取句子以及其中token的嵌入，注意需要将模型下载到本地，例如[bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)（没有测试过所有bert模型，但似乎bert-base系列都能够支持）。

属性：

* `model_path`：模型的本地路径.

* `tokenizer_path`：tokenizer的路径，一般是模型路径后加上一个斜杠。

* `from_encoder`：句子和token的嵌入需要包含哪几个隐藏状态，包括最后四层的拼接`"last4"` (默认) 和最后一层`"last1"`.

* `pooling_method`：句子池化的方法，包括将所有token的向量进行平均`"mean"` (默认)，对每个维度选取最大值`"max"`以及直接输出transformer自动返回的CLS向量`"cls_head"`.

方法：

* `get_tokens(text="")`：返回一个列表，包含句子中的所有token，注意在句首和句尾分别添加了[CLS]和[SEP]。

* `all_token_embeddings(text="")`：获取每个token的嵌入，返回一个二维数组，第一个维度是token，第二个维度是对应的向量维度。维度取决于实例中设定的`from_encoder`，最后四层共3072个维度。如果只需要最后一层，则为768维。

* `get_word_index(text="", word="")`：获取词语在句子中出现位置，即对应第几个token，返回一个二维数组，第一个维度是词语出现的次数，第二个维度是每次出现的位置。

* `get_word_embedding(text="", word="")`：获取包含句子上下文信息的词语嵌入。返回一个二维数组，第一个维度是词语出现的次数，第二个维度是每次的嵌入向量的维度。词语嵌入向量是词语中包含的每个token向量的平均。

* `get_sentence_embedding(text="")`：获取句子的嵌入，池化方法取决于实例的`pooling_method`。如果是`"mean"` 或`"max"`，则句向量维度与token维度一致 (768或3072)，如果是`"cls_head"`，则返回transformers中默认的768维CLS向量。

```python
EmbeddingAPI(api_key="", base_url="", model_name="", dimensions=float())
```

调用API获取句子嵌入，支持GPT和Qwen。

属性：

* `api_key`：api key字符串，如果配置了环境变量也可以用os获取。

* `base_url`：base url字符串。

* `model_name`：模型的名字。

* `dimensions`：输出向量的维度。

方法：

- `get_sentence_embedding(text="")`：获取句子的嵌入，返回一个数组。

## vecCalculation

vecCalculation模块目前只有两个函数：

```python
cos_similarity(vec_1, vec_2)
```

输入两个列表或数组，对两个向量计算余弦相似度，返回一个float。

```python
SCWEAT(target, attribute_1, attribute_2, sig=False)
```

单类别词嵌入联系测验。目标词是一个单维数组，两种属性词分别是两个二维数组，每一行是一个词向量。`sig`决定是否输出显著性，由于置换检验比较耗时，默认不输出显著性。返回一个字典，`["effect_size"]`为效应量，`["p_value"]`为显著性p值。如果`sig=False`则只包含`["effect_size"]`一个键。



邮箱：tanhaoyuan3456@163.com