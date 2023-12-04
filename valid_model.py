import argparse

import seaborn
import torch
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('../')
from dataset import *
from dataset import BERTDataset,collate_mlm

def random_word(sentence, vocab):
    tokens = sentence.split()
    tokens_len = [len(token) for token in tokens]
    chars = tokens
    output_label = []

    for i, char in enumerate(chars):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                chars[i] = vocab.mask_index

            # 10% randomly change token to random token
            elif prob < 0.9:
                chars[i] = random.randrange(vocab.vocab_size)

            # 10% randomly change token to current token
            else:
                chars[i] = vocab.char2index(char)

            output_label.append(vocab.char2index(char))

        else:
            chars[i] = vocab.char2index(char)
            output_label.append(0)

    return chars, output_label


def draw(data, x, y, ax):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, # 取值0-1
                    cbar=False, ax=ax)


def Modelload(path):
    assert path is not None
    print(f"path:{path}")
    mlm_encoder = torch.load(path)
    return mlm_encoder


# 验证模型是否收敛
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", default = "output/bert.model/model_mlm/mlm_ep1 .model", required=False, type=str, help="model of pretrain")
    parser.add_argument("-v", "--vocab_path", default = "data/category.vocab",required=False, type=str, help="path of vocab")
    args = parser.parse_args()

    model_path = args.model_path
    vocab_path = args.vocab_path

    vocab = WordVocab.load_vocab(vocab_path)

    model = torch.load(model_path).to("cuda:0")
    model.eval()

    d = BERTDataset("./data/category_train.txt", vocab = vocab)


    batch = [d.__getitem__(i) for i in range(5000)] 
    collated = collate_mlm(batch)


    collated = {key: value.to("cuda:0") for key, value in collated.items()}

    mask_lm_output, attn_list = model.forward(collated["mlm_input"], collated["input_position"])

    prediction = torch.argmax(mask_lm_output, dim=2)
    
    
    batch_chars = []
    batch_size, seq_len = prediction.shape
    correct = 0
    masked_counts = 0
    for b in range(batch_size):
        print(f"Batch # {b}")
        char_predictions = vocab.index2char(prediction[b].tolist())
        label = collated["mlm_label"][b]
        masked_index = torch.nonzero(collated["mlm_label"][b]).reshape(1, -1)[0]

        for idx in masked_index:
            masked_counts += 1
            if vocab.index2char(label[idx].item()) == char_predictions[idx]:
                correct += 1
            print(f"Masked label: {vocab.index2char(label[idx].item())} --> Prediction: {char_predictions[idx]}")
    print(f"Accuracy: {correct/masked_counts}")
    import pdb
    pdb.set_trace()
    

    for layer in range(3):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Layer", layer+1)
        for h in range(4):
            # a = model.bert.layers[layer].multihead.attention[0,h].data
            draw(model.bert.layers[layer].multihead.attention[0, h].data, #[0, h].data,
                 sent, sent if h == 0 else [], ax=axs[h])
        plt.show()


if __name__ == '__main__':
    main()