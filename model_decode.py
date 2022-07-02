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

    words = ['I', 'say', "that"]
    word2id = [en2index[i] for i in words]
    input = torch.LongTensor(word2id)

    model = Net(len(en2index))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    net_state_dict = torch.load(PATH_MODEL)
    model.load_state_dict(net_state_dict)

    steps = 50

    with torch.no_grad():
        for i in range(steps - 1):
            words = input[i:i + 3].to(device)  # 每次取三个词
            output = torch.argmax(model.decode(words))  # 解码获得最可能的下一个词
            input = input.to(device)
            input = torch.cat((input, output.unsqueeze(0)), dim=0)  # 拼接

        input = input.tolist()
        # print(input)
        s = []

        for i in range(len(input)):
            for en, index in en2index.items():
                if index == input[i]:
                    r = en.replace("@@ ", "")
            s.append(r)
    print(s)
