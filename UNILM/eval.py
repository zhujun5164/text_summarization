import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pytorch_transformers.modeling_bert import BertConfig, BertForMaskedLM
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch
import code
import argparse
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

from model_utils import Bert_for_UNILM, make_token_ids, UNILM_MASK
# from pytorch_pretrained_bert.modeling import BertForMaskedLM
from data_utils import data_generator, read_text, read_vocab, make_data, bert_token_and_to_id
from rouge import Rouge

def eval(txt, topk): # batch_size, seq_size
    model.eval()
    # 获取sent_id
    segment_ids = [0] * len(txt)
    target_ids = [[] for _ in range(topk)]  # 1, topk
    target_scores = [0] * topk  # 1, topk

    for i in range(args.max_output_len):
        # 第一次输入时，target_ids是None, len(t)为0，因此输入的BERT模型中的为txt
        _target_ids = [txt + t for t in target_ids]  # topk, seq_size + len(t)
        _segment_ids = torch.tensor([segment_ids + [1] * len(t) for t in target_ids])  # topk, seq_size + len(t)
        attention_mask = UNILM_MASK(_segment_ids)  # topk, seq_size + len(t)

        # 转为tensor格式
        _target_ids = torch.tensor(_target_ids).to(device)
        _segment_ids = _segment_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 将所生成的数据丢入BERT  topk, seq_size, vocab_size
        with torch.no_grad():
            output = model(_target_ids, token_type_ids = _segment_ids, attention_mask = attention_mask)
            # output = model(_target_ids, token_type_ids=_segment_ids)
        # 取最后一个单词的输出作为预测，并softmax获得在单词中每个词的预

        # 测概率
        y_pre = torch.softmax((output[0][:,-1,3:]), -1)  # topk, vocab_size

        # y_pre = torch.softmax((output[:, -1]), -1)  # topk, vocab_size
        probs = np.log(y_pre.cpu().numpy() + 1e-6)
        # 取topk的
        token_topk = np.argsort(probs, -1)[:, -topk:]  # topk, topk

        ##
        candidate_ids, candidate_scores = [], []
        for j, (ids, score) in enumerate(zip(target_ids, target_scores)): # for次数 topk
            if i == 0 and j > 0:
                continue
            for k in token_topk[j]: # topk
                candidate_ids.append((ids + [k + 3])) # 获取由第一个ids作为输入到BERT，以及分别与前topk个预测进行结合
                # candidate_ids.append((ids + [k]))
                candidate_scores.append(score + probs[j][k]) # 加分数，原先概率应为乘，但取了log后只需要做加操作

        # 循环后candidate_ids, candidate_scores 维度 topk * topk，这样就得到输入句子中每个的topk可能
        token_topk = np.argsort(candidate_scores)[-topk:]  # 取topk * topk中的topk的下标

        for j, k in enumerate(token_topk):
            # target_ids[j].append(candidate_ids[k][-1]) # 加到target_ids中
            target_ids[j] = candidate_ids[k]
            target_scores[j] = candidate_scores[k] # 换分数
        ends = [j for j, k in enumerate(target_ids) if k[-1] == tokenizer.sep_token_id]
        if len(ends) > 0:
            k = np.argmax([target_scores[j] for j in ends])
            return tokenizer.decode(target_ids[ends[k]])

    return tokenizer.decode(target_ids[np.argmax(target_scores)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--folder_path', type=str, default=None, required=True,
                        help='the path where own the trian_data/vocab/config/model bin')
    parser.add_argument('--task_name', type=str, default=None, required=True,
                        help='the train result and model data will save in folder_path')
    parser.add_argument('--data_name', type=str, default='data.csv',
                        help='the train and dev data, and the keys must own summarization and text')
    parser.add_argument('--train_name', type=str, default='train.csv',
                        help='if the train_data is not exit in data_path, the code will create a new train_data by '
                             'itself and with the train_name')
    parser.add_argument('--dev_name', type=str, default='dev.csv',
                        help='if the dev_data is not exit in data_path, the code will create a new dev_data by '
                             'itself and with the dev_name')

    # model parameters
    # 因为在中国...还是强制性需要使用本地数据
    parser.add_argument('--config_name', type=str, default=None, required=True,
                        help='the config data in the folder_path')
    parser.add_argument('--vocab_name', type=str, default=None, required=True,
                        help='the vocab data in the folder_path')
    parser.add_argument('--model_name', type=str, default=None, required=True,
                        help='the model data in the folder_path')
    parser.add_argument('--use_UNILM', action='store_true',
                        help='if use use_UNILM, use the UNILM mask in the train, else use base BERT')
    parser.add_argument('--use_summary_loss', action='store_true',
                        help='if with use_summary_loss, use a new loss just attention the summary, '
                             'else use the loss just mask the PAD word')
    parser.add_argument('--limit_vocab', action='store_true',
                        help='if with limit_vocab, use a new vocab in the train')
    parser.add_argument('--limit_vocabulary_name', type=str, default=None,
                        help='the limit_vocabulary use for limit_vocab task')
    parser.add_argument('--limit_vocab_model_name', type=str, default=None,
                        help='the limit_vocab_model use for limit_vocab task')
    parser.add_argument('--min_count', type=int, default=0,
                        help='the number of word more than min_count can be used in new vocab')

    parser.add_argument('--load_model', type=str, default=None,
                        help='if have, the code will contiune train the load model')
    # parser.add_argument('--text_name', type=str, default='UNILM', required=True)

    # other parameters
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=10,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_input_len', type=int, default=400,
                        help='the limit length for input text')
    parser.add_argument('--max_output_len', type=int, default=32,
                        help='the limit length for input summary and predict sentence')
    parser.add_argument('--epochs', type=int, default=30,
                        help='train epoch')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='train batch_size')
    parser.add_argument('--topk', type=int, default=4,
                        help='topk for beam_search')
    parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()

    if args.use_cuda:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
    else:
        device = torch.device('cpu')

    # join data path
    save_path = os.path.join(args.folder_path, args.task_name)
    data_path = os.path.join(args.folder_path, args.data_name)
    train_path = os.path.join(args.folder_path, args.train_name)
    dev_path = os.path.join(args.folder_path, args.dev_name)

    # dev阶段这些应该是不需要做存在判断了
    # if os.path.exists(data_path) is not True:
    #     os.mkdir(data_path)
    # make data, 如果train_path或者dev_path都存在，通过data_path做数据
    # if (os.path.exists(train_path) and os.path.exists(dev_path)) is not True:
    #     make_data(data_path, train_path, dev_path)

    config_path = os.path.join(args.folder_path, args.config_name)
    vocab_path = os.path.join(args.folder_path, args.vocab_name)

    if args.limit_vocab:
        new_vocab_path = os.path.join(args.folder_path, args.limit_vocabulary_name)
        new_model_path = os.path.join(args.folder_path, args.limit_vocab_model_name)
        vocab_path = new_vocab_path
        model_path = new_model_path

    # 创建分词器
    # 使用vocab文件进行pretrain的时候出了错误，输出的vocab成为了乱码
    tokenizer = BertTokenizer.from_pretrained(vocab_path)

    # 获取初始的bert模型参数目录
    config = BertConfig.from_pretrained(config_path)
    if args.limit_vocab:
        config.vocab_size = tokenizer.vocab_size

    rouge = Rouge()

    if os.path.exists(save_path) is not True:
        os.mkdir(save_path)

    dev_data = pd.read_csv(dev_path, encoding='utf-8')
    dev_texts = list(dev_data['text'].values)
    dev_summaries = list(dev_data['summarization'].values)

    summary_all = []
    for summary in dev_summaries:
        summary = ''.join([word + ' ' for words in summary for word in words])
        summary_all.append(summary)

    # 改为for i in save_model.
    # 导入checkpoint
    # 预测
    outputs_path = os.path.join(save_path, 'dev_outputs')
    rouge_scores = []
    load_models = []
    listdirs = os.listdir(data_path)
    for listdir in listdirs:
        model_path = os.path.join(data_path, listdir)
        # 因为在training代码上确定了保存模型的尾椎是pt结尾的，所以这里做判断的时候就直接使用endwith('.pt')
        # 如果结尾为pt结尾，就说明是训练所保存的模型，就读入然后做eval
        if model_path.endswith('.pt'):
            # 读取模型
            model = BertForMaskedLM.from_pretrained(model_path, config=config)
            model = model.to(device)
            pre_all = []
            for test in dev_texts:
                text_id = bert_token_and_to_id(tokenizer, test)
                pre = eval(text_id[:args.max_input_len], args.topk)
                # 防止生成[]导致rouge报错
                if pre == []:
                    pre = ['1']

                pre = ''.join([word + ' ' for words in pre for word in words])
                pre_all.append(pre)

            # 计算rouge-1-r
            # rouge计算需要每个字用空格分隔
            print('iter: ', iter)
            for i in pre_all:
                print(i)
            scores = rouge.get_scores(pre_all, summary_all, avg=True)

            # 存储score数据
            rouge_scores.append(scores)
            load_models.append(listdir)
        else:
            continue

    with open(outputs_path, 'w') as f:
        n = len(rouge_scores)
        for i in range(n):
            print(load_models[i], ':', rouge_scores[i])