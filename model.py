# coding: UTF-8
import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM,AutoModelForMaskedLM, AutoModelForSeq2SeqLM
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.model_name == 'roberta'
        # self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        # self.roberta_model = RobertaForMaskedLM.from_pretrained("model/roberta-base").to(device)
        self.roberta_model = AutoModelForMaskedLM.from_pretrained('FacebookAI/roberta-base').to(device)
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


class MLP_albert(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.model_name == 'albert'
        self.albert_model = AutoModelForMaskedLM.from_pretrained('albert/albert-base-v2').to(device)
        self.albert_model.resize_token_embeddings(args.vocab_size)
        for param in self.albert_model.parameters():
            param.requires_grad = True

        self.hidden_size = 128

        self.vocab_size = args.vocab_size

    # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
    def forward(self, batch_arg, arg_mask, mask_indices, batch_size):
        word_emb = self.albert_model.albert.embeddings.word_embeddings(batch_arg).to(device)
        temp_emb = self.albert_model(attention_mask = arg_mask, inputs_embeds = word_emb)[0].to(device)

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
        da = self.albert_model.albert.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp

class MLP_T5(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.model_name == 't5'
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base").to(device)
        self.t5_model.resize_token_embeddings(args.vocab_size)
        for param in self.t5_model.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size

    # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
    def forward(self, batch_arg, arg_mask, mask_indices, batch_size):
        # 创建 decoder_input_ids，使用 <pad> 作为起始标记
        # decoder_input_ids = self.t5_model._shift_right(batch_arg)
        # batch_arg = torch.unsqueeze(batch_arg[0,:mask_indices+1],0)
        # arg_mask = torch.unsqueeze(arg_mask[0,:mask_indices+1],0)
        # 使用 input_ids 和 decoder_input_ids
        outputs = self.t5_model(input_ids=batch_arg,  decoder_input_ids=batch_arg, attention_mask=arg_mask)
        temp_emb = outputs.logits.to(device)

        prediction = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(temp_emb[i], mask_indices[i])
            if i == 0:
                prediction = e_emb
            else:
                prediction = torch.cat((prediction, e_emb), dim=0)
        return prediction


    def extract_event(self, embed, mask_idx):
        mask_embed = embed[mask_idx]
        mask_embed = torch.unsqueeze(mask_embed, 0)
        return mask_embed

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.t5_model.shared.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
