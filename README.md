# ChatGLM_MultiGPUCPU_eval
简易实现ChatGLM单机调用多个计算设备（GPU、CPU）进行推理
# <b>推理</b>
## 使用
#### 从仓库下载 MultiDevices.py
#### 加载模型
```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half()
```
#### 调用函数
```python
>>> model = MultiDevices.ConfigMultiDevices(model,embeddings='cpu',
                                                  layers={'cuda:0': '1-14',
                                                          'cuda:1': '15-28'},
                                                  final_layernorm='cuda:1')
word_embeddings -> cpu
layer 0 -> cuda:0
layer 1 -> cuda:0
layer 2 -> cuda:0
layer 3 -> cuda:0
layer 4 -> cuda:0
layer 5 -> cuda:0
layer 6 -> cuda:0
layer 7 -> cuda:0
layer 8 -> cuda:0
layer 9 -> cuda:0
layer 10 -> cuda:0
layer 11 -> cuda:0
layer 12 -> cuda:0
layer 13 -> cuda:0
layer 14 -> cuda:1
layer 15 -> cuda:1
layer 16 -> cuda:1
layer 17 -> cuda:1
layer 18 -> cuda:1
layer 19 -> cuda:1
layer 20 -> cuda:1
layer 21 -> cuda:1
layer 22 -> cuda:1
layer 23 -> cuda:1
layer 24 -> cuda:1
layer 25 -> cuda:1
layer 26 -> cuda:1
layer 27 -> cuda:1
final_layernorm -> cuda:1
lm_head -> cpu
hooked.
```
```python
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
```
