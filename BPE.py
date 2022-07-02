"""
BPE（byte-pair encoding，字节对编码算法）
1）主要目的：数据压缩，把词的本身的意思和词的形态变化部分分开，有效的减少了词表的数量

2）算法描述：字符串里频率最常见的一对字符被一个没有在这个字符中出现的字符代替的层层迭代过程（迭代合并出现频率高的字符对）

3）算法流程：
    1.设定最大subwords个数V，也就是最后希望获得的子词数目
    2.将所有单词拆分为单个字符，并在最后添加一个结束符_，同时标记出该单词出现的次数，如{'l o v e_': 12}（词频统计）
    3.统计连续字节对出现的频率，选择出现频率最高的进行合并，合并出新的subword
    4.重复步骤3，直到subwords的个数为V，或最高频率为1时

4）相关说明：
    1.结束符_的意义在于表示subword是词后缀

5）subword-nmt学习：
    1.learn-bpe: learn BPE merge operations on input text.（学习BPE对输入文本的合并操作）
    2.apply-bpe: apply given BPE operations to input text.（将给定的BPE操作应用于输入文本）
    3.get-vocab: extract vocabulary and word frequencies from input text.（从输入文本中提取词汇和词频）
    4.learn-joint-bpe-and-vocab: executes recommended workflow for joint BPE.（执行联合BPE的推荐工作流，即边学习边生成词汇词频提取文件）

6)实现——在miniconda下使用subword-nmt
    1.下载subword-nmt包：pip install subword-nmt
    2.在训练集上学BPE，得到分词结果；统计每一个连续字节对，并保存为code_file：
        subword-nmt learn-bpe -s 合并操作次数 --input 输入的文件路径 --output 输出的code_file文件路径
        subword-nmt learn-bpe -s 5000 --input F:\PyTorch学习\BPE_handle\train.txt --output F:\PyTorch学习\BPE_handle\code_file.txt
    3.对训练集初步BPE的结果统计vocabulary——统计每一个连续字节对出现的频率（词频统计）：
        subword-nmt apply-bpe -c code_file文件路径 --input F:\PyTorch学习\corpus.tc.en.txt | subword-nmt get-vocab --output 输出的vocab_file文件路径
        subword-nmt apply-bpe -c F:\PyTorch学习\BPE_handle\code_file.txt --input F:\PyTorch学习\BPE_handle\corpus.tc.en | subword-nmt get-vocab --output F:\PyTorch学习\BPE_handle\vocab_file.txt
    （2、3步可直接合并为subword-nmt learn-joint-bpe-and-vocab -s 5000 -i 输入的train_file文件 -o 输出的code_file文件 --write-vocabulary vocab_file文件）
        subword-nmt learn-joint-bpe-and-vocab -s 5000 -i F:\PyTorch学习\BPE_handle\train.txt -o F:\PyTorch学习\BPE_handle\code_file.txt --write-vocabulary F:\PyTorch学习\BPE_handle\vocab_file.txt

    4.同时使用学到的BPE和统计的词频vocabulary对训练集应用BPE
        subword-nmt apply-bpe -c 输入的codes_file文件路径 --vocabulary  输入的vocab_file路径 --vocabulary-threshold 词频阈值 --input 输入的train文件路径 --output 输出的train_file_BPE文件路径
        subword-nmt apply-bpe -c F:\PyTorch学习\BPE_handle\code_file.txt --vocabulary  F:\PyTorch学习\BPE_handle\vocab_file.txt --vocabulary-threshold 10 --input F:\PyTorch学习\BPE_handle\train.txt --output F:\PyTorch学习\BPE_handle\train_BPE.txt
    5.用训练集上学到的BPE和统计的词频对验证集应用BPE
        subword-nmt apply-bpe -c 输入的codes_file文件路径 --vocabulary  输入的vocab_file路径 --vocabulary-threshold 词频阈值 --input 输入的test文件路径 --output 输出的train_file_BPE文件路径
        subword-nmt apply-bpe -c F:\PyTorch学习\BPE_handle\code_file.txt --vocabulary  F:\PyTorch学习\BPE_handle\vocab_file.txt --vocabulary-threshold 10 --input F:\PyTorch学习\BPE_handle\test.txt --output F:\PyTorch学习\BPE_handle\test_BPE.txt
"""
