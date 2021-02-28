import cmath

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from attention import Attention
from util import get_sent_tensor, get_doc_tensor


class SentenceGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, device):
        super(SentenceGRU, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True,
                          num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim * 2)

    def forward(self, sentences, sentence_lengths, word_masks):
        sentence_lengths, sorted_indices = torch.sort(sentence_lengths, descending=True)
        _, unsorted_indices = torch.sort(sorted_indices, descending=False)

        sentences_sorted = sentences.index_select(0, sorted_indices)

        sentences_packed = pack_padded_sequence(sentences_sorted, sentence_lengths.cpu(), batch_first=True)
        sentences_embd = self.gru(sentences_packed)[0]
        sentences_embd = pad_packed_sequence(sentences_embd, batch_first=True, total_length=sentences.shape[1])[0]
        sentences_embd = sentences_embd.index_select(0, unsorted_indices)

        att_w = self.attention(sentences_embd)
        att_w.data.masked_fill_(word_masks, -cmath.inf)
        att_w = F.softmax(att_w, dim=1)
        sentences_embd1 = torch.matmul(att_w.transpose(1, 2), sentences_embd).squeeze(1)
        sentences_embd1 = self.linear1(sentences_embd1)

        sentences_embd2 = torch.max(sentences_embd.masked_fill(word_masks, -cmath.inf), dim=1)[0]
        sentences_embd2 = self.linear2(sentences_embd2)

        sentences_embd = sentences_embd1 + sentences_embd2
        # sentences_embd = sentences_embd2
        sentences_embd = torch.tanh(sentences_embd)
        # sentences_embd = F.leaky_relu(sentences_embd)
        sentences_embd = self.dropout(sentences_embd)

        return sentences_embd


class DocumentGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, device):
        super(DocumentGRU, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True,
                          num_layers=num_layers, batch_first=False, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim * 2)

    def forward(self, batches, num_sents, masks):
        num_sents, sorted_indices = torch.sort(num_sents, descending=True)
        _, unsorted_indices = torch.sort(sorted_indices, descending=False)

        sents_sorted = batches.index_select(1, sorted_indices)
        sents_packed = pack_padded_sequence(sents_sorted, num_sents.cpu(), batch_first=False)

        docs_embd = self.gru(sents_packed)[0]
        docs_embd = pad_packed_sequence(docs_embd, batch_first=False, total_length=batches.shape[0])[0]
        docs_embd = docs_embd.index_select(1, unsorted_indices).transpose(0, 1)

        att_w = self.attention(docs_embd)
        att_w.data.masked_fill_(masks, -cmath.inf)
        att_w = F.softmax(att_w, dim=1)
        docs_embd1 = torch.matmul(att_w.transpose(1, 2), docs_embd).squeeze(1)
        docs_embd1 = self.linear1(docs_embd1)

        docs_embd2 = torch.max(docs_embd.masked_fill(masks, -cmath.inf), dim=1)[0]
        docs_embd2 = self.linear2(docs_embd2)

        docs_embd = docs_embd1 + docs_embd2
        # docs_embd = docs_embd2
        docs_embd = torch.tanh(docs_embd)
        # docs_embd = F.leaky_relu(docs_embd)
        docs_embd = self.dropout(docs_embd)

        return docs_embd


class Model(nn.Module):
    def __init__(self, args, input_dim, device, num_class):
        super(Model, self).__init__()
        self.sent_encoder = SentenceGRU(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                        dropout=args.dropout, device=device)
        self.doc_embed = DocumentGRU(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                     dropout=args.dropout, device=device)
        self.classifier = nn.Linear(args.hidden_dim, num_class)
        self.device = device

        self.sent_norm = nn.BatchNorm1d(args.hidden_dim)
        self.doc_norm = nn.BatchNorm1d(args.hidden_dim)

    def forward(self, docs):
        sent_tensors, num_sents, sent_lens, word_masks = get_sent_tensor(docs, self.device)
        sent_tensors = sent_tensors.to(self.device)
        sent_tensors = self.sent_encoder(sent_tensors, sent_lens.to(self.device),
                                         word_masks.to(self.device))

        # sent_tensors = self.sent_norm(sent_tensors)
        sent_tensors = sent_tensors.unsqueeze(1)
        batches, masks = get_doc_tensor(sent_tensors, num_sents)

        # print(batches.shape)
        docs_embd = self.doc_embed(batches, num_sents.to(self.device), masks.to(self.device))

        # docs_embd = self.doc_norm(docs_embd)
        return self.classifier(docs_embd)

