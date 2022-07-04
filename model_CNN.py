import torch
import torch.nn as nn
import torch.nn.functional as F

"""
context_size = 3            # 模型设定的需给定的词数，这里设为3
emb_dim = 32                # 取出的词向量设为32维的特征
middle_size = 128           # 输出的二维张量的大小，这里设为128
"""


class Net(nn.Module):
    def __init__(self, vocab_size, context_size=3, emb_dim=32, middle_size=128, bind_emb=True):
        super(Net, self).__init__()  # 继承父类nn.Module的属性，用其初始化方法初始化继承的属性

        self.n_gram = context_size  # 设置n_gram为给定的词数3

        # Embedding进行词嵌入，随机初始化映射为一个向量矩阵
        # 参数1是嵌入字典的词的数量，参数2是每个嵌入向量的大小，此处为词向量维度32
        # nn.embedding会对较短的句子进行padding，将填充位置的值映射为0（padding_idx为0，即默认用0进行填充）
        self.w_emb = nn.Embedding(vocab_size, emb_dim)

        self.net = nn.Sequential(  # Sequential是一个顺序容器，模块按照它们在构造函数中传递的顺序添加到其中
            # 参数1表示输入的二维张量的大小，参数2表示输出的二维张量的大小
            # nn.Linear即表示，将输入的样本的参数1个特征，转换为参数2个特征（batch_size, in_features）->（batch_size, out_features）
            nn.Linear(emb_dim * context_size, middle_size),
            nn.GELU(),  # GELU函数处理
            nn.Linear(middle_size, emb_dim)  # 最后一层不需要添加激活函数
        )

        self.classifier = nn.Linear(emb_dim, vocab_size)  # 设置分类器

        if bind_emb:
            self.classifier.weight = self.w_emb.weight  # 绑定词向量与分类器的权重

    def forward(self, input):  # 传入input: 输入的语料，表示有多少行，每行有多少个词（batch_size, seql）
        output = self.w_emb(input)  # 将input转换为embedding类型，得到一个向量矩阵（batch_size, seql, vocab_size）

        _l = []

        # n_data指的是一个长度为seql的句子可以提供的数据条数，由“传入的矩阵的最大维数 - 给定词数”所得
        # input.size即一句话有多少个词
        n_data = input.size(-1) - self.n_gram

        for i in range(self.n_gram):
            # _l[]: (batch_size, n_data, vocab_size)
            # narrow函数: 参数1表示裁剪的纬度(0行1列)，参数2表示从选中的纬度的第几维开始裁剪，参数3表示裁剪的长度
            _l.append(output.narrow(1, i, n_data))

        # _input_net: (batch_size, n_data, vocab_size * n_gram)
        # torch.cat函数: 参数1为等待连接的张量序列，参数2为给定的纬度(当dim=-1时，表示按最后一个纬度）
        _input_net = torch.cat(_l, dim=-1)

        # 将(batch_size, n_data, vocab_size*n_gram)->(batch_size, n_data, vocab_size)
        # 本质上是作最后一维的变化，即将“三个词的词向量”->“词表上的概率预测”
        output = self.classifier(self.net(_input_net))

        # output = F.log_softmax(output， dim=-1)  # 按照行(1)或者列(0)来做归一化的，然后在softmax的结果上再做多一次log运算，对最后一维进行概率运算

        return output

    def decode(self, word):
        word_emb = self.w_emb(word)     # 对输入的三个词汇的索引进行词嵌入，得到一个向量矩阵
        # print(word_emb)

        word_emb = word_emb.view(1, -1)     # 合并第2维和第3维，得到（1，32*词汇个数），32*词汇个数即所有词的特征个数
        # print(word_emb)

        output = self.classifier(self.net(word_emb))    # 将处理后的向量矩阵放入两层神经模型进行进一步的处理
        log_max_prob = F.log_softmax(output, dim=-1)
        # print(log_max_prob)

        return log_max_prob







