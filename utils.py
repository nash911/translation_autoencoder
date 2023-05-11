from __future__ import unicode_literals, print_function, division
from io import open

import numpy as np
import unicodedata
import torch
import re
import random
import time
import math

import matplotlib.pyplot as plt
from lang import Lang

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepareData(lang1, lang2, max_length, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def plot_loss(axs, train_loss, valid_loss, plot_freq=1, show=False, save=True, path=None):
    # Training (NLL) Loss plot
    x = (np.arange(len(train_loss)) * plot_freq).tolist()
    axs.clear()
    axs.plot(x, train_loss, color='red',
             label='Train Loss')
    axs.plot(x, valid_loss, color='blue', label='Validation Loss')
    axs.set(title='Learning Curves')
    axs.set(ylabel='NLLL')
    axs.set(xlabel='Iterations')
    axs.legend(loc='upper right')

    if save:
        plt.savefig(path + "plots/learning_curves.png")

    if show:
        plt.show(block=False)
        plt.pause(0.01)


# def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length, device):
#     with torch.no_grad():
#         input_tensor = tensorFromSentence(input_lang, sentence, device)
#         input_length = input_tensor.size()[0]
#         encoder_hidden = encoder.initHidden()
#
#         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
#             encoder_outputs[ei] += encoder_output[0, 0]
#
#         decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
#
#         decoder_hidden = encoder_hidden
#
#         decoded_words = []
#         decoder_attentions = torch.zeros(max_length, max_length)
#
#         for di in range(max_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             decoder_attentions[di] = decoder_attention.data
#             topv, topi = decoder_output.data.topk(1)
#             if topi.item() == EOS_token:
#                 decoded_words.append('<EOS>')
#                 break
#             else:
#                 decoded_words.append(output_lang.index2word[topi.item()])
#
#             decoder_input = topi.squeeze().detach()
#
#         return decoded_words, decoder_attentions[:di + 1]
#
#
# def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_length, device,
#                      n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang,
#                                             pair[0], max_length, device)
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')
