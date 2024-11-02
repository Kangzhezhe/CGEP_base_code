# coding: UTF-8
from copy import deepcopy
import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class SeDGPL(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.roberta_model = RobertaForMaskedLM.from_pretrained("model/roberta-base").to(device)
        self.roberta_model.resize_token_embeddings(args.vocab_size)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        self.robert_text = deepcopy(self.roberta_model)

        for param in self.robert_text.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        # gate1
        self.W1_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W1_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # gate2
        self.W2_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W2_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.vocab_size = args.vocab_size

    # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
    def forward(self, batch_arg, arg_mask, mask_indices, batch_size,sentences,event_tokenizer_pos, event_key_pos):
        for i in range(batch_size):
            for k in sentences[i]:
                sent_emb = self.robert_text.roberta(sentences[i][k]['input_ids'], sentences[i][k]['attention_mask'])[0].to(device)
                sentences[i][k]['emb'] = sent_emb[0][sentences[i][k]['position']]
        

        word_emb = self.roberta_model.roberta.embeddings.word_embeddings(batch_arg).to(device)

        for i in range(batch_size):
            for j in range(len(event_tokenizer_pos[i])):
                instance_emb = (word_emb[i][event_tokenizer_pos[i][j]]).clone().unsqueeze(0)
                sent_emb = (sentences[i][event_key_pos[i][j]]['emb']).clone().unsqueeze(0)
                gate_1 = torch.sigmoid(self.W1_1(instance_emb) + self.W1_2(sent_emb)).to(device)
                out_gate_1 = (torch.mul(gate_1, instance_emb) + torch.mul(1.0 - gate_1, sent_emb)).to(device)

                # gate_2 = torch.sigmoid(self.W2_1(out_gate_1) + self.W2_2(type_emb)).to(device)
                # out_gate_2 = (torch.mul(gate_2, out_gate_1) + torch.mul(1.0 - gate_2, type_emb)).to(device).squeeze(0)

                # word_emb[i][event_tokenizer_pos[i][j]] = out_gate_2
                word_emb[i][event_tokenizer_pos[i][j]] = out_gate_1
                assert str(int(batch_arg[i][event_tokenizer_pos[i][j]])) in event_key_pos[i][j]

        temp_emb = self.roberta_model.roberta(attention_mask = arg_mask, inputs_embeds = word_emb)[0].to(device)

        anchor_maks = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(temp_emb[i], mask_indices[i])
            if i == 0:
                anchor_maks = e_emb
            else:
                anchor_maks = torch.cat((anchor_maks, e_emb),dim=0)

        prediction = self.roberta_model.lm_head(anchor_maks)
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



class SeDGPL1(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.roberta_model = RobertaForMaskedLM.from_pretrained("model/roberta-base").to(device)
        self.roberta_model.resize_token_embeddings(args.vocab_size)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        self.robert_text = deepcopy(self.roberta_model)
        self.robert_type = deepcopy(self.roberta_model)

        for param in self.robert_text.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        # gate1
        self.W1_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W1_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # gate2
        self.W2_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W2_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.vocab_size = args.vocab_size

    # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
    def forward(self, batch_arg, arg_mask, mask_indices, batch_size,sentences,event_tokenizer_pos, event_key_pos,batch_Type_arg, mask_Type_arg):
        for i in range(batch_size):
            for k in sentences[i]:
                sent_emb = self.robert_text.roberta(sentences[i][k]['input_ids'], sentences[i][k]['attention_mask'])[0].to(device)
                sentences[i][k]['emb'] = sent_emb[0][sentences[i][k]['position']]
        

        Type_emb = self.robert_type.roberta(batch_Type_arg, attention_mask=mask_Type_arg, output_hidden_states=True)[0].to(device)

        word_emb = self.roberta_model.roberta.embeddings.word_embeddings(batch_arg).to(device)

        for i in range(batch_size):
            for j in range(len(event_tokenizer_pos[i])):
                instance_emb = (word_emb[i][event_tokenizer_pos[i][j]]).clone().unsqueeze(0)
                sent_emb = (sentences[i][event_key_pos[i][j]]['emb']).clone().unsqueeze(0)
                type_emb = (Type_emb[i][event_tokenizer_pos[i][j]]).clone().unsqueeze(0)
                gate_1 = torch.sigmoid(self.W1_1(instance_emb) + self.W1_2(sent_emb)).to(device)
                out_gate_1 = (torch.mul(gate_1, instance_emb) + torch.mul(1.0 - gate_1, sent_emb)).to(device)

                gate_2 = torch.sigmoid(self.W2_1(out_gate_1) + self.W2_2(type_emb)).to(device)
                out_gate_2 = (torch.mul(gate_2, out_gate_1) + torch.mul(1.0 - gate_2, type_emb)).to(device).squeeze(0)

                word_emb[i][event_tokenizer_pos[i][j]] = out_gate_2
                assert str(int(batch_arg[i][event_tokenizer_pos[i][j]])) in event_key_pos[i][j]

        temp_emb = self.roberta_model.roberta(attention_mask = arg_mask, inputs_embeds = word_emb)[0].to(device)

        anchor_maks = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(temp_emb[i], mask_indices[i])
            if i == 0:
                anchor_maks = e_emb
            else:
                anchor_maks = torch.cat((anchor_maks, e_emb),dim=0)

        prediction = self.roberta_model.lm_head(anchor_maks)
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