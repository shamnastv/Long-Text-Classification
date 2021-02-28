import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def get_sent_tensor(docs, device):
    max_l = 1
    num_sents = []
    sent_tensor = []
    sent_lens = []
    word_masks = []
    for doc in docs:
        num_sents.append(len(doc))
        for sent in doc:
            max_l = max(max_l, len(sent))
        if len(doc) == 0:
            print('zero doc')

    for doc in docs:
        for sent in doc:
            if len(sent) == 0:
                print('zero sent')

            len_s = len(sent)
            pad = max_l - len_s
            sent_lens.append(len_s)

            word_masks.append([0] * len_s + [1] * pad)
            sent_tensor.append(torch.from_numpy(sent).float())

    sent_tensor = pad_sequence(sent_tensor, batch_first=True)
    num_sents = torch.tensor(num_sents).long()
    sent_lens = torch.tensor(sent_lens).long()
    word_masks = torch.tensor(word_masks).long().unsqueeze(2).eq(1)

    return sent_tensor, num_sents, sent_lens, word_masks


def get_doc_tensor(sent_tensors, num_sents):
    num_sents_np = num_sents.numpy()
    max_seq = max(num_sents_np)
    start = 0
    batches = []
    masks = []
    for i in range(len(num_sents_np)):
        num_s = num_sents_np[i]
        end = start + num_s
        padding = max_seq - num_s
        pad_tuple = [0, 0, 0, 0, 0, padding]
        padded_sents = F.pad(sent_tensors[start:end], pad_tuple)
        batches.append(padded_sents)
        masks.append([0] * num_s + [1] * padding)
        start = end
    masks = torch.tensor(masks).long().unsqueeze(2).eq(1)
    batches = torch.cat(batches, dim=1)
    return batches, masks
