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

def train(iter, txt, summary):

    model.train()
    optimizer.zero_grad()
    txt = torch.tensor(txt)
    summary = torch.tensor(summary)

    token_type_id = make_token_ids(txt, summary)   # batch_size, seq_size

    if args.use_UNILM:
        attention_mask = UNILM_MASK(token_type_id)      # batch_size, seq_size, seq_size
        attention_mask = attention_mask.to(device)

    token_type_id = token_type_id.to(device)
    bert_input = torch.cat((txt, summary), 1).to(device)

    if args.use_UNILM:
        outputs = model(bert_input, token_type_ids = token_type_id, attention_mask = attention_mask)
    else:
        outputs = model(bert_input, token_type_ids = token_type_id)

    # vocab_embed = model.bert.embeddings.word_embeddings.weight
    # vocab_embed = vocab_embed.permute(1,0).clone()
    # outputs = torch.matmul(outputs, vocab_embed)
    # outputs = torch.softmax(outputs, -1)

    # batch_size, seq_size, vocab_size
    y_pre = outputs[0][:, :-1]
    target = bert_input[:, 1:]
    y_mask = 1. - (target == tokenizer.pad_token_id).float()

    acc_mask = token_type_id[:, 1:].clone()
    acc_mask[y_mask == 0] = 0

    # function loss
    # log_probs = torch.log(y_pre + 1e-12)
    # log_probs = log_probs.permute(0,2,1)
    # loss = torch.nn.functional.nll_loss(log_probs, target, reduction="none")

    loss = loss_fn(y_pre.reshape(-1, y_pre.size(2)), target.reshape((-1)))

    if args.use_summary_loss:
        loss = torch.sum(loss.mul(acc_mask.reshape(-1))) / torch.sum(acc_mask)
    else:
        loss = torch.sum(loss.mul(y_mask.reshape(-1))) / torch.sum(y_mask)

    # 试着做一个Loss,让summary部分的loss比重更大
    # _loss = _loss.mul(y_mask)
    # loss_s = _loss.mul(acc_mask) * 10
    # loss_t = _loss.mul(1. - acc_mask)
    # loss = loss_s + loss_t

    loss.backward()
    optimizer.step()

    # token_type_id 0, 0, 0, 0, 1, 1, 1
    acc_pre = (y_pre.argmax(-1).squeeze() == target).float()
    acc = torch.sum(torch.mul(acc_pre, acc_mask), -1) / torch.sum(acc_mask, -1)
    acc = torch.mean(acc)

    # acc = torch.sum(((y_pre.argmax(-1).squeeze() == target).float()) * acc_mask, -1) / torch.sum(acc_mask, -1)
    # acc = torch.mean(acc)

    # CrossEntropyLoss
    # loss = loss_fn(y_pre.view(-1, tokenizer.vocab_size), target.view(-1))

    iter += 1
    return loss.item(), acc.item(), iter

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

    if os.path.exists(data_path) is not True:
        os.mkdir(data_path)
    # make data, 如果train_path或者dev_path都存在，通过data_path做数据
    if (os.path.exists(train_path) and os.path.exists(dev_path)) is not True:
        make_data(data_path, train_path, dev_path)

    config_path = os.path.join(args.folder_path, args.config_name)
    vocab_path = os.path.join(args.folder_path, args.vocab_name)
    model_path = os.path.join(args.folder_path, args.model_name)

    if args.limit_vocab:
        if (os.path.exists(args.folder_path + '/' + args.limit_vocabulary_name) and os.path.exists(args.folder_path + '/' + args.limit_vocab_model_name)) is True:
            new_vocab_path = os.path.join(args.folder_path, args.limit_vocabulary_name)
            new_model_path = os.path.join(args.folder_path, args.limit_vocab_model_name)
            # 感觉可以尝试做一些检查流程，检查new_vocab_size与model.bert.embeddings.word_embedding.weight.shape[0]是否相等
            # 如果不相等则需要重新构建，并保存

        else:
            new_vocab_name = 'new_vocab_' + str(args.min_count) + '.txt'
            new_model_name = 'bert-' + str(args.min_count) + '-limit_vocab_model.bin'
            new_vocab_path = os.path.join(args.folder_path, new_vocab_name)
            new_model_path = os.path.join(args.folder_path, new_model_name)

            # 检查是否有限制字典的数据存在，没有就创造
            # 感觉有小bug，判断的不是与limit word对应的
            if (os.path.exists(new_vocab_path) or os.path.exists(new_model_path)) is not True:
                # 好了。。。开始作死添加东西吧
                # 读取所有文件并统计词频
                word_collection = []
                for text, _ in read_text(data_path, args.max_input_len, args.max_output_len):
                    word_collection.extend([i for i in text])
                # counter_vocab = Counter(word_collection).most_common(config.new_vocab_size)

                counter_vocab = [word[0] for word in list(Counter(word_collection).items()) if word[1] > args.min_count]

                # build new_vocab
                _new_vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
                _new_vocab.extend(counter_vocab)

                # 获取到了新的单词表，这样就需要设置一个单词表对应与旧单词表的字典，以及自己本身的一个字典
                # 这样就需要读取下旧字典
                read = read_vocab(vocab_path)
                old_vocab = {}
                for n, i in enumerate(read):
                    old_vocab[i] = n

                # 看看new_vocab的单词是否全在old_vocab上，如果不在的就删去, 同时获取到new_vocab在old_vocab中对应的id
                new_vocab = []
                vocab_dict = {}
                token_num = []
                for n, word in enumerate(_new_vocab):
                    if word in old_vocab:
                        new_vocab.append(word)
                        vocab_dict[word] = len(vocab_dict)
                        token_num.append(old_vocab[word])

                # 更改下保存的数据并保存为新的模型
                model_dict = torch.load(model_path)
                model_dict['bert.embeddings.word_embeddings.weight'] = model_dict['bert.embeddings.word_embeddings.weight'][token_num]
                # 保存模型
                torch.save(model_dict, new_model_path)

                # save vocab
                f = open(new_vocab_path, 'w', encoding='utf-8')
                for n, i in enumerate(new_vocab):
                    if n == 0:
                        f.write(i)
                    else:
                        f.write('\n')
                        f.write(i)
                f.close()

        vocab_path = new_vocab_path
        model_path = new_model_path

    # 创建分词器
    # 使用vocab文件进行pretrain的时候出了错误，输出的vocab成为了乱码
    tokenizer = BertTokenizer.from_pretrained(vocab_path)

    # 获取初始的bert模型参数目录
    config = BertConfig.from_pretrained(config_path)
    if args.limit_vocab:
        config.vocab_size = tokenizer.vocab_size

    # 构建model
    model = BertForMaskedLM.from_pretrained(model_path, config=config)
    model = model.to(device)

    # load_model判断
    if args.load_model is not None:
        load_model_path = os.path.join(args.folder_path, args.load_model)
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_dict'])
        iter = checkpoint['iter']
        best_loss = checkpoint['best_loss']
        best_acc = checkpoint['best_acc']
        best_rouge_1_r = checkpoint['best_rouge_1_r']
    else:
        best_loss, best_acc, best_rouge_1_r = 999, 0, 0
        iter = 0

    count_loss, count_acc, n = 0, 0, 0

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = (50000 / args.batch_size) / args.gradient_accumulation_steps * args.epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    writer = SummaryWriter(save_path)
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

    for i in range(args.epochs):
        for txt, summary in data_generator(train_path, tokenizer, args.batch_size,
                                           args.max_input_len, args.max_output_len):
            loss, acc, iter = train(iter, txt, summary)
            count_loss += loss
            count_acc += acc
            n += 1

            if iter % 500 == 0:
                count_loss = count_loss / n
                count_acc = count_acc / n

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
                score = rouge.get_scores(pre_all, summary_all, avg=True)
                rouge_1_r = score['rouge-1']['r']
                writer.add_scalar("rouge_1_r", rouge_1_r, iter)

                # print('iter:', iter, "loss:", count_loss, "acc:", count_acc, 'rouge_1_r', rouge_1_r)
                if count_loss < best_loss and count_acc >= best_acc and rouge_1_r > best_rouge_1_r:
                    best_loss = count_loss
                    best_acc = count_acc

                    model_name = 'best_model' + str(iter) + '.pt'
                    model_path = os.path.join(save_path, model_name)

                    state = model.state_dict()
                    for key in state:
                        state[key] = state[key].clone().cpu()

                    torch.save({
                        "model_dict": state,
                        "iter": iter + 1,
                        "best_loss": best_loss,
                        "best_acc": best_acc,
                        "best_rouge_1_r": best_rouge_1_r
                        }, model_path)

                count_loss, count_acc, n = 0, 0, 0

            writer.add_scalar("Loss/train", loss, iter)
            writer.add_scalar("Accuracy/train", acc, iter)