import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


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
        self.embedding_layer = self.add_embed_layer(self.embedding_matrix, self.vocab_size['word'], self.embed_size)
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
