import pandas as pd
import numpy as np
import code

# 读取文件
def read_text(data_path, max_input_len, max_output_len):
    '''
    :param data_path: 输入的是数据csv文件的地址，数据中的特征有'txt','summarization'
    :return:
        t : txt
        s : summarization
    中间过程：在文本的取出时，由于BERT模型在输入时存在最长长度限制，因此在去取txt时需要对超过规定长度的txt取到我们规定的最长长度
    '''
    df = pd.read_csv(data_path)
    text = df["text"].values
    summarization = df["summarization"].values

    for t, s in zip(text, summarization):
        #只取摘要长度小于要求的输出出去， 同时限制txt的长度不能超过最大txt长度
        if len(s) <= max_output_len:
            yield t[:max_input_len], s

# 构建好用作模型输入的data_id
def data_generator(data_path, tokenizer, batch, max_output_len, max_input_len):
    '''
    制作数据生成器
    :return: 输出的是经过分词、单词转化为id、根据BERT fine-tuning要求格式处理后的txts和summarys.
             batch_size, seq_size
    '''
    while True:
        txts, summarys = [], []
        for txt, summary in read_text(data_path, max_output_len, max_input_len):
            pad_id = tokenizer.pad_token_id
            # 分词以及转化为id
            txt_token_id, summary_token_id = bert_token_and_to_id(tokenizer, txt, summary)
            if batch == 1:
                yield txt_token_id, summary_token_id
            else:
                txts.append(txt_token_id)
                summarys.append(summary_token_id)
                if len(txts) == batch:
                    # pad数据
                    txts = padding(txts, pad_id)
                    summarys = padding(summarys, pad_id)
                    yield txts, summarys
                    txts, summarys = [], []

def bert_token_and_to_id(tokenizer, first, second = None):
    '''
    ：data_process

    根据BERT模型中对文本输入要求进行处理
    单输入情况：         [CLS] first [SEP]
    双句子输入情况：     [CLS] first [SEP] second [SEP]

    因此若在对输入进行处理时，对first_sentence前后分别加上[CLS], [SEP]. 变为[CLS] first [SEP]
                            对second_sentence的最后加上[SEP].  变为second [SEP]

    ：output
    在输入的结果为对两个做完处理后的句子进行分词以及 tokens_to_ids后的结果
    = =，虽然在我这是懒了点直接获取了sep 和 cls的id，再和分词，to_id后的句子进行拼接
    '''

    #加CLS
    first_token_id = []
    first_token_id.append(tokenizer.cls_token_id)
    # 对文本进行分词和转化为id
    first_token = tokenizer.basic_tokenizer.tokenize(first)

    first_token_id.extend(tokenizer.convert_tokens_to_ids(first_token))
    #加SEP
    first_token_id.append(tokenizer.sep_token_id)

    if second is not None:
        #分词和转化成id
        second_token = tokenizer.basic_tokenizer.tokenize(second)
        second_token_id = tokenizer.convert_tokens_to_ids(second_token)
        #加sep
        second_token_id.append(tokenizer.sep_token_id)
        return first_token_id, second_token_id

    return first_token_id

# padding数据，添加0到数据的batch中最长的长度
def padding(txt, pad_id):
    '''
    :param txt: 输入的是batch后的数据
    :return:    padding后的数据
    '''
    max_len = max([len(i) for i in txt])
    padding_txt = [i + [pad_id] * (max_len - len(i)) for i in txt]
    return padding_txt


def read_vocab(path):
    f = open(path, encoding='utf-8')
    read = f.read().split('\n')
    f.close()
    return read

def make_data(path, train_path, dev_path, scale = 0.9):
    # 读取数据
    data = pd.read_csv(path, encoding='utf_8')
    # 获取random排序
    num = np.arange(len(data))
    np.random.shuffle(num)
    limit_num = int(len(data) * scale)
    # 读取数据
    train_data = data.iloc[num[:limit_num]]
    dev_data = data.iloc[num[limit_num:]]

    train_data.to_csv(train_path, index=0, encoding='utf_8_sig')
    dev_data.to_csv(dev_path, index=0, encoding='utf_8_sig')