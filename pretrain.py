import random
import h5py
import numpy as np

PATH_E = "F:/PyTorch学习/BPE_handle/corpus.tc.en"
PATH_T = "F:/PyTorch学习/BPE_handle/train.txt"
PATH_B = "F:/PyTorch学习/BPE_handle/train_BPE.txt"
PATH_B_S = "F:/PyTorch学习/BPE_handle/train_BPE_sort.txt"
PATH_B_S_H = "F:/PyTorch学习/BPE_handle/train_BPE_sort_handle.txt"
PATH_F5 = "F:/PyTorch学习/BPE_handle/result.hdf5"
PATH_INDEX = "F:/PyTorch学习/BPE_handle/en_index"


# 剔除长度超过128个token的句子，获得训练集文件
def file_get():
    with open(PATH_T, "w", encoding="utf-8") as fwt:
        with open(PATH_E, "r", encoding="utf-8") as frd:
            for line in frd:
                tmp = line.strip()
                if len(tmp.split()) <= 128:
                    fwt.write(tmp)
                    fwt.write("\n")


# 对BPE处理完后的数据集进行排序
def file_sort():
    data = {}  # 数据集以字典形式存储
    with open(PATH_B, "r", encoding="utf-8") as frd:
        for line in frd:
            l = len(line.strip().split())  # 由于每一行已经经过处理，故可直接统计词长
            if 3 <= l <= 256:
                if l in data:
                    data[l].append(line.strip())  # 相似长度句子聚合，添加进字典data
                else:
                    data[l] = [line.strip()]

    # 同一长度的句子通过random包的shuffle方法做一次顺序打乱，防止数据收集时相近分布的数据聚集在一起，导致模型训练时拟合局部数据分布
    for line in data.values():
        if len(line) > 1:
            random.shuffle(line)

    # 存储已排好序的字典
    with open(PATH_B_S, "w", encoding="utf-8") as fwt:
        for i in sorted(data, reverse=True):  # 排序,reverse=True，表示单词反向排序，满足高频到低频条件
            fwt.write("\n".join(data[i]))
            fwt.write("\n")

    with open(PATH_B_S_H, "w", encoding="utf-8") as fwt:
        with open(PATH_B_S, "r", encoding="utf-8") as frd:
            for line in frd:
                tmp = line.strip()
                if tmp:
                    tmp = tmp.replace("@@ ", "")
                    fwt.write(tmp)


# 收集数据集上的词典，建立单词到索引的一对一映射并保存结果（0索引对应padding，其它单词的索引从1开始累加，单词顺序从高频到低频)
def file_handle_index():
    words = {}

    # 词频统计
    with open(PATH_B_S, "r", encoding="utf-8") as frd:
        for line in frd:
            tmp = line.strip()
            if tmp:
                for w in tmp.split():
                    words[w] = words.get(w, 0) + 1

    vocab = {word: i for i, word in enumerate(sorted(words, reverse=True), 2)}
    vocab['<padding>'] = 0
    vocab['<unk>'] = 1
    print(vocab)
    np.save(PATH_INDEX, vocab)

    return vocab


# 根据词典索引，将排序后的数据切分为batch（每个batch包含的token数量尽可能多，但不要超过2560个token）并转换为tensor（为matrix，batch_size * seql)
# batch_size: 这个batch中包含几个句子，seql: 这些句子中最长句子的长度（短句使用padding，映射索引为0，填充到相同长度）
# yield学习: 当执行到yield关键词的代码时，函数会暂时返回，下次调用该函数时，会从上次暂停的地方继续运行，起到一个暂时返回的作用
def file_handle_batch(srcf, data):
    batch = []  # 以列表形式初始化一个batch
    index = 0  # 设置索引为0
    ready = True  # 是否进行数据切分的标志
    matrix_line = 1  # 设置初始矩阵行数为1

    for line in srcf:
        tmp = line.strip()
        if tmp:
            tmp = tmp.split()
            if ready:
                seql = len(tmp)  # 获得最长句子的长度
                batch_size = int(2560 / seql)  # 由于每个batch包含的token数目不超过2560，所以用2560/seql确定batch中的句子数，也是矩阵的行数
                batch = []  # 创建一个空batch
                ready = False  # 置为False，表示当前正在进行数据切分为batch

                if matrix_line < batch_size:
                    # 矩阵行数 < batch包含的句子数时，矩阵行数+1，
                    matrix_line += 1
                    # 对每行的数据进行填充，seql-len(tmp)表示剩余的空位，用0进行填充，将处理好的数据存入batch
                    batch.append([data.get(w, 1) for w in tmp] + [0 for _ in range(seql - len(tmp))])
            else:
                ready = True  # 置为True，表示可进行下一个batch的处理及存储
                matrix_line = 1  # 将下一个矩阵的行数置1处理
                yield batch, index  # 暂时返回batch及相应索引
                index += 1
    if batch:
        yield batch, index


# 按hdf5格式存入文件,HDF5存储格式：src(一个group)/<k, v>：存转换后的数据，k为数据的索引，从0自增，v为具体数据张量
# ndata:一个只有一个元素的向量，存src中数据的数量；nword:一个只有一个元素的向量，存第3步收集的词典大小
def file_save(srcf, data, f5):
    index = 0
    group = f5.create_group("group")  # 创建名为“group”的group

    for batch, index in file_handle_batch(srcf, data):  # 获得batch及对应的索引
        m_array = np.array(batch, dtype=np.int32)  # 创建ndarray，用来存储batch
        group.create_dataset(str(index), data=m_array)  # 创建数据集，以索引大小作为数据集的名字，batch作为数据集的数据

    f5["ndata"] = np.array(index, dtype=np.int32)  # 存src中数据的数量，即索引，将其写入文件的主键"ndata"下
    f5["nword"] = np.array(len(data), dtype=np.int32)  # 存收收集的词典大小，将其写入文件的主键"nword"下


if __name__ == '__main__':
    # file_get()
    # file_sort()
    word = file_handle_index()
    with open(PATH_B_S, "r", encoding="utf-8") as frd:
        with h5py.File(PATH_F5, "w") as f5:
            print("正在处理")
            file_save(frd, word, f5)
            print("hdf5文件保存成功")
