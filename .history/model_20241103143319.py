# coding: UTF-8
import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.roberta_model = RobertaForMaskedLM.from_pretrained("model/roberta-base").to(device)
        self.roberta_model.resize_token_embeddings(args.vocab_size)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        self.vocab_size = args.vocab_size

    # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
    def forward(self, batch_arg, arg_mask, mask_indices, batch_size):
        word_emb = self.roberta_model.roberta.embeddings.word_embeddings(batch_arg).to(device)
        temp_emb = self.roberta_model(attention_mask = arg_mask, inputs_embeds = word_emb)[0].to(device)

        prediction = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(temp_emb[i], mask_indices[i])
            if i == 0:
                prediction = e_emb
            else:
                prediction = torch.cat((prediction, e_emb),dim=0)
        return prediction


    def extract_event(self, embed, mask_idx):
        mask_embed = embed[mask_idx]
        mask_embed = torch.unsqueeze(mask_embed, 0)
        return mask_embed
    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.roberta_model.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
