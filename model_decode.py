import torch
import numpy as np
from model_CNN import Net

PATH_MODEL = "./train_state/train_state_73000.pth"
PATH_INDEX = "en_index.npy"


def load(srcf):
    v = np.load(srcf, allow_pickle=True).item()  # 字典的导入
    return v


if __name__ == '__main__':
    en2index = load(PATH_INDEX)
    # print(en2index)

    words = ['I', 'like', "to"]
    word2id = [en2index[i] for i in words]
    input = torch.LongTensor(word2id)
    # print(input)
    # print("**"*50)

    model = Net(len(en2index))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    net_state_dict = torch.load(PATH_MODEL)
    model.load_state_dict(net_state_dict)

    # pretrain_dict = torch.load(PATH_MODEL)
    # net_state_dict = model.state_dict()
    # pretrain_dict_1 = {k: v for k, v in pretrain_dict.items() if k in net_state_dict}
    # net_state_dict.update(pretrain_dict_1)
    # model.load_state_dict(net_state_dict)

    get_words = 50

    for i in range(get_words):
        words = input[i:i+3].to(device)  # 每次取三个词的索引
        print(words)
        output = torch.argmax(model.decode(words))  # 解码获得最可能的下一个词
        # print(output)
        input = input.to(device)
        input = torch.cat((input, output.unsqueeze(0)), dim=0)  # 拼接成一个53个词的一维tensor
        # print(input)

    # print("**" * 50)
    input = input.tolist()  # 将tensor中的数值以列表形式存储，方便处理
    # print(input)

    result_words = []

    for i in range(len(input)):
        for en, index in en2index.items():
            if index == input[i]:
                r = en.replace("@@", "")
                result_words.append(r)
    print(result_words)



# 问题：解码得到的50个词后半部分重复