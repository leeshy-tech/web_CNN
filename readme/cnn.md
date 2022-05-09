# Web搜索大作业

## 大纲：

- 任务描述
- 数据集介绍
- 代码实现
- 使用
---
#### 任务描述

本次任务是对话文本语义匹配。  
即针对中文问答数据中的问句对，
判定两个句子语义是否相同或者相近。

举例：
> Eg1:  
> Q1： “什么音乐你是知道的”  
> Q2： “网易云音乐关闭”  
> Label：0  

> Eg2:  
> Q1： “我睡一会儿”  
> Q2： “我还是眯一会儿吧”  
> Label：1  

label表示问句之间的语义是否相同。  
若相同则标为1，
若不相同则标为0.


**要求：**  
**基于示例CNN代码（或自行设计算法），实现在测试集上语义相似度判断任务（要包含F1值）**    
**另：**  
**示例代码仅供参考，给出的参数仅仅是稍微试过的，而没经过精细化调参。**  

---
#### 数据集介绍
本次数据集是节选自节选自千言提供的OPPO小布对话文本语义匹配中的部分数据集，格式为json文件。   
数据存放位置：CNN/data/


数据集概况如下：  

| 名称         | 数量     | 标签  |
|------------|--------|-----|
| train.json | 157173 | 有   |
| dev.json   | 10000  | 有   |
| test.json| 10000  | 有   |



---
#### 代码实现

1. 读取数据集：  
```python
def load_data(data_path):
    """
    区分训练/验证/测试集
    @param data_path: 数据json文件存放位置
    @return: 训练/验证/测试集
    """

    with open(data_path) as f:
        data = json.load(f)

    return data
``` 
同学们直接打开json文件观察数据集看到的应该是ascii码，这和数据的写入方式有关。


2. 数据预处理

一般来说，基于CNN的文本语义相似度任务需要如下预处理过程：
- 将原始文本分词并转换成以词的序列 
- 将词序列转换成以词编号（每个词表中的词都有唯一编号）为元素的序列 
- 将词的编号序列中的每个元素（某个词）展开为词向量的形式。  
**注意：转成词向量需要借助已经建立好的映射，文件太大就没有放进压缩包了。请大家自行下载 网址：[维基百科中文词向量](https://pan.baidu.com/s/1ZKePwxwsDdzNrfkc6WKdGQ)**  
**下载之后放哪儿？ 这个无关紧要，只要把hyp.py中的embed_path修改成存放位置即可**
```python
# data_preprocess.py
def read_sentences(dataset, vocab, is_train, repr='word', test_vocab=None):
    """
    将输入转换为id，创建词表
    参数pred_mode作用是控制是否返回标签
    因为和gen_data共用一个函数，因此需要根据试验集合调整返回的参数
    """

    # 数据读取
    question_1, question_2 = [], []
    max_len_1, max_len_2 = 0, 0
    punc = punctuation + u'1-9.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s: '

    seq1 = []
    seq2 = []
    label = []

    for data in dataset:
        seq1.append(data['q1'])
        seq2.append(data['q2'])
        label.append(int(data['label']))

    # 数据清洗
    for i in range(len(label)):
        if label[i] > 0:
            label[i] = 1
        else:
            label[i] = 0


    # 对每对问句处理
    for i in range(len(seq1)):
        seq1[i] = re.sub(r"[{}]+".format(punc), " ", seq1[i])
        seq2[i] = re.sub(r"[{}]+".format(punc), " ", seq2[i])

        # 将问句分成一个个token
        q1_tokens = split_sent(seq1[i], repr)
        q2_tokens = split_sent(seq2[i], repr)

        # 获取句子最长度
        if len(q1_tokens) > max_len_1:
            max_len_1 = len(q1_tokens)
        if len(q2_tokens) > max_len_2:
            max_len_2 = len(q2_tokens)

        token_id1, token_id2 = [], []

        # 对单个问句中的每个token进行处理
        for token in q1_tokens:
            # repr = 'word'
            if token not in vocab[repr]:
                if is_train:
                    # 如果在训练集，就注册词库
                    # eg: vocab['word']['我'] = 10
                    vocab[repr][token] = len(vocab[repr])
                elif repr == 'word' and token not in test_vocab[repr]:
                    # 如果不是在训练集，且未在测试词库注册，则注册
                    # eg" test_vocab['word']['注册'] = 1000
                    test_vocab[repr][token] = len(vocab[repr]) + len(test_vocab[repr])
            if token in vocab[repr]:
                # 如果这个token在词库注册了，那么就把对应的键值塞入列表
                token_id1.append(vocab[repr][token])
            elif repr == 'word':
                token_id1.append(test_vocab[repr][token])
            else:
                token_id1.append(OOV_WORD_INDEX)
        # print("-----", token_id1)
        question_1.append(token_id1)
        for token in q2_tokens:
            if token not in vocab[repr]:
                if is_train:
                    vocab[repr][token] = len(vocab[repr])
                elif repr == 'word' and token not in test_vocab[repr]:
                    test_vocab[repr][token] = len(vocab[repr]) + len(test_vocab[repr])
            if token in vocab[repr]:
                token_id2.append(vocab[repr][token])
            elif repr == 'word':
                token_id2.append(test_vocab[repr][token])
            else:
                token_id2.append(OOV_WORD_INDEX)
        question_2.append(token_id2)

    return question_1, question_2, max_len_1, max_len_2, label


```

```python
    train_vocab_emb, test_vocab_emb = construct_vocab_emb("./experimental-data", vocab['word'], test_vocab['word'], 300,
                                                          base_embed_path=embed_path)
```

3. 参数设置  
```python
# hyp.py
hyp = {
    'embed_path': '[Your "sgns.wiki.bigram" path]', # 记得修改位置
    'data_path': '../../data',
    'batch_size': 128,
    'nb_filters': 128,
    'dropout_rate': 0.3,
    'embed_size': 300,
    'learning_rate': 0.05,
    'epoches': 60,
    'save_model_name': "checkpoint_textCNN.pt"
}

```

4. 网络模型
```python
class creat_model(nn.Module):
    def __init__(self, batch_size, vocab_size, embedding_matrix,
                 nb_filters, embed_size=300, dropout_rate=0.5, num_classes=2,
                 kernel_dim=100, kernel_sizes=(2, 3, 4)):
        super(creat_model, self).__init__()
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.nb_filters = nb_filters
        self.kernel_dim = kernel_dim
        self.kernel_sizes = kernel_sizes
        self.num_class = num_classes
        # 词嵌入层
        self.embedding_layer = self.add_embed_layer(self.embedding_matrix, 
                                                    self.vocab_size['word'], self.embed_size)
        # CNN编码层
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (k, self.embed_size)) for k in kernel_sizes])
        # 输出分类层
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim * 2, num_classes)
        nn.init.xavier_uniform_(self.fc.weight.data, gain=1)
        nn.init.constant_(self.fc.bias.data, 0.1)

    def add_embed_layer(self, vocab_emb, vocab_size, embed_size):
        if vocab_emb is not None:
            # 预训练词向量
            embed_layer = nn.Embedding(vocab_size, embed_size)
            pretrained_weight = np.array(vocab_emb)
            embed_layer.weight.data.copy_(torch.from_numpy(pretrained_weight))
            for p in embed_layer.parameters():
                p.requires_grad = False
        else:
            # 随机初始化
            print("Embedding with random weights")
            embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        return embed_layer

    def forward(self, query_word_input, doc_word_input):
        # 这里对视频中的代码做了修改，使用了多个并行的卷积
        # 经过词嵌入层，获得词向量
        query_word_emb = self.embedding_layer(query_word_input).unsqueeze(1)
        doc_word_emb = self.embedding_layer(doc_word_input).unsqueeze(1)
        # print(query_word_emb.size())
        # [batch_size, 1, seq_len, embedding_dim]
        # 经过卷积层，和最大池化层
        query_word_emb = [F.relu(conv(query_word_emb)).squeeze(3) for conv in self.convs]
        query_word_emb = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in query_word_emb]
        doc_word_emb = [F.relu(conv(doc_word_emb)).squeeze(3) for conv in self.convs]
        doc_word_emb = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in doc_word_emb]
        # 拼接不同卷积层的特征
        query_concated = torch.cat(query_word_emb, 1)
        query_concated = self.dropout_layer(query_concated)
        doc_concated = torch.cat(doc_word_emb, 1)
        doc_concated = self.dropout_layer(doc_concated)
        # 拼接query和doc
        concated = torch.cat([query_concated, doc_concated], dim=-1)
        prediction = self.fc(concated)
        return prediction
```

4. 训练过程
```python
# train_CNN.py
    # ===================== TRAIN Model ======================
    # ### 定义模型 ###
    model = creat_model(batch_size, vocab_size, merge_vocab_emb, nb_filters, embed_size, dropout_rate)
    model = model.to(device)
    # 定义优化器
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                          weight_decay=1e-6, momentum=0.9, nesterov=True)
    lr_reducer = ReduceLROnPlateau(optimizer=opt, verbose=True)
    print("use SGD optimizer")
    # 定义损失函数
    print("compile model with binary_crossentropy")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion.to(device)

    try:
        total_start_time = time.time()
        best_acc, best_f1, best_thresh = None, None, None
        best_auc = None
        best_fpr, best_tpr = None, None
        train_auc_list, valid_auc_list = [], []
        print("-" * 90)
        for epoch in range(epoches):
            epoch_start_time = time.time()
            # 训练
            train_loss, train_fpr, train_tpr, train_auc = train_fc(model, train_dataset, train_dataset['sim'],
                                                                   batch_size, opt, criterion)
            train_auc_list.append(train_auc)
            print("|start of epoch{:3d} | time : {:2.2f}s | loss {:5.6f} | train_auc {}".format(epoch + 1,
                                                                                                time.time() - epoch_start_time,
                                                                                                train_loss, train_auc))
            # 验证集上验证性能
            val_loss, val_fpr, val_tpr, val_auc, val_f1 = validate(model, valid_dataset, valid_dataset['sim'],
                                                                   batch_size, criterion)
            valid_auc_list.append(val_auc)
            lr_reducer.step(val_loss)
            print("-" * 10)
            print("| end of epoch {:3d}| time: {:2.2f}s | loss: {:.4f} |valid_auc {} |valid_f1 {}".format(epoch + 1,
                                                                                                          time.time() - epoch_start_time,
                                                                                                          val_loss,
                                                                                                          val_auc,
                                                                                                          val_f1))
            if not best_auc or best_auc < val_auc:
                best_auc = val_auc
                best_fpr = val_fpr
                best_tpr = val_tpr
                model_state_dict = model.state_dict()
                print("save the best model... best_auc: %s" % best_auc)
                model_weight = hyp['save_model_name']
                torch.save(model_state_dict, model_weight)
            with open('train_textCNN_auc.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(train_auc_list)
            with open('valid_textCNN_auc.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(valid_auc_list)
    except KeyboardInterrupt:
        print("-" * 90)
        print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time) / 60.0))


```

5. 测试过程
```python
    # =================== Test model =====================
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion.to(device)
    print("load best model ... ")
    # 定义一个新的模型
    new_model = creat_model(batch_size, vocab_size, merge_vocab_emb, nb_filters, embed_size, dropout_rate)
    new_model = new_model.to(device)
    # 加载最佳模型的参数赋给新建模型
    # model_weight = "checkpoint_textCNN.pt"
    new_model.load_state_dict(torch.load(model_weight), strict=False)
    print(model_weight)
    # 测试集测试
    test_loss, test_fpr, test_tpr, test_auc, test_f1 = validate(new_model, test_dataset, test_dataset['sim'], \
                                                                batch_size, criterion)
    # 打印结果
    print("test_loss:", test_loss)
    print("test_auc:", test_auc)
    print('test_f1:', test_f1)
```
---

#### 使用
训练：
```
python train_CNN.py
```
测试：
```python
python test_CNN.py
```




