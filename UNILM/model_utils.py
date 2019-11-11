from pytorch_transformers.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch


def init_linear_wt(linear):
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)

# 重做一下我们用的bert
class Bert_for_UNILM(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_for_UNILM, self).__init__(config)

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        weight = self.bert.embeddings.word_embeddings.weight
        self.classifier.weight.data = weight.data

        # init_linear_wt(self.classifier)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, position_ids = None, token_type_ids = token_type_ids, attention_mask = attention_mask, head_mask = head_mask)
        outputs = outputs[0]
        # batch_size, seq_size ,hidden_size
        # outputs = self.dropout(outputs)

        # outputs = torch.softmax(self.classifier(outputs), -1)
        outputs = self.classifier(outputs)
        outputs = torch.softmax(outputs, -1)

        return outputs  # batch_size, seq_size, vocab_size


# 做token_type_ids
def make_token_ids(first, second = None):
    # 其实如果不是两个句子做训练的话，可以不用做make_token_ids，在BERT中若不输入make_token_ids的话会默认做全0的token_type_ids
    # 感觉在使用时输入的应该为batch_size, seq_size的数据因此输出的应为batch_size, seq_size
    token_type_ids_0 = torch.zeros_like(first)
    if second is not None:
        token_type_ids_1 = torch.ones_like(second)
        token_type_ids = torch.cat((token_type_ids_0, token_type_ids_1), -1)
        return token_type_ids
    return token_type_ids_0

# 制作文章UNILM中提到的mask
# 在文章中对于mask中需要相关注意的是，对于txt文本，希望的是他只能在txt之间进行相互注意；
#                                 对于summary文本，希望的是他能观察到当前的位置，以及在这之前的所以txt文本
def UNILM_MASK(token_type_id):
    # token_type_id  用作句子向量区分的一个编码，txt部分为0， summary部分为1
    # batch_size, seq_size
    # _batch_size = token_type_id.size(0)
    _seq_size = token_type_id.size(1)
    a_mask = torch.tril(torch.ones(1, 1, _seq_size, _seq_size))

    mask_13 = token_type_id.unsqueeze(1).unsqueeze(3)
    mask_12 = token_type_id.unsqueeze(1).unsqueeze(2)

    a_mask = (1 - mask_13) * (1 - mask_12) + mask_13 * a_mask
    a_mask = a_mask.view(-1, _seq_size, _seq_size)
    return a_mask