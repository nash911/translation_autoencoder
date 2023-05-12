from __future__ import unicode_literals, print_function, division
import numpy as np
import random
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import tensorFromSentence, tensorsFromPair, timeSince, plot_loss

SOS_token = 0
EOS_token = 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, device, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.device = device
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, teacher_ratio, device):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_attention(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, criterion, max_length, teacher_ratio, device):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, input_lang, output_lang, pairs, encoder_optimizer,
               decoder_optimizer, n_iters, device, max_length, teacher_ratio=0.5,
               n_evals=20, print_every=1000, eval_every=1000, plot_every=100,
               plot_show=False, path=None):
    start = time.time()

    # Create a matlibplot canvas for plotting learning curves
    fig, axs = plt.subplots(1, figsize=(10, 6))

    train_losses = []
    test_losses = []
    print_train_loss_total = 0  # Reset every print_every
    plot_train_loss_total = 0  # Reset every plot_every
    test_loss_total = 0  # Reset every plot_every

    # Split data into training and validation sets
    num_pairs = len(pairs)
    train_size = int(num_pairs * 0.9)
    pair_inds = np.arange(num_pairs)
    np.random.shuffle(pair_inds)
    train_inds = pair_inds[:train_size]
    test_inds = pair_inds[train_size:]

    # Create tensor from pairs for training and validation sets
    training_pairs = [tensorsFromPair(input_lang, output_lang, pairs[ind], device)
                      for ind in train_inds]
    test_pairs = [tensorsFromPair(input_lang, output_lang, pairs[ind], device)
                  for ind in test_inds]
    eval_pairs = [pairs[ind] for ind in test_inds]

    print(f"Train Size: {len(train_inds)} -- Test Size: {len(test_inds)}")

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = random.choice(training_pairs)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        if type(decoder) == AttnDecoderRNN:
            loss = train_attention(
                input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, max_length, teacher_ratio, device)
        else:
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, criterion, teacher_ratio, device)

        print_train_loss_total += loss
        plot_train_loss_total += loss

        if iter % print_every == 0:
            # Evaluate current model
            test_loss_total = 0
            for el in range(n_evals):
                test_pair = random.choice(test_pairs)
                input_tensor = test_pair[0]
                target_tensor = test_pair[1]

                if type(decoder) == AttnDecoderRNN:
                    test_loss_total += test_attention(
                        input_tensor, target_tensor, encoder, decoder, criterion,
                        max_length, device)
                else:
                    test_loss_total += test(
                        input_tensor, target_tensor, encoder, decoder, criterion, device)

            print_loss_avg = print_train_loss_total / print_every
            print_train_loss_total = 0

            test_loss_avg = test_loss_total / n_evals
            test_losses.append(test_loss_avg)

            print('%s (%d %d%%) %.4f - %.4f' %
                  (timeSince(start, iter / n_iters), iter, iter / n_iters * 100,
                   print_loss_avg, test_loss_avg))

            if path is not None:
                torch.save(encoder.state_dict(), path + 'models/encoder.pth')
                torch.save(decoder.state_dict(), path + 'models/decoder.pth')

        if iter % plot_every == 0:
            plot_loss_avg = plot_train_loss_total / plot_every
            train_losses.append(plot_loss_avg)
            plot_train_loss_total = 0
            plot_loss(axs, train_losses, test_losses, plot_freq=plot_every,
                      show=plot_show, save=True, path=path)

        if iter % eval_every == 0:
            print(f"\nModel evaluation after {iter} iterations:")
            evaluateRandomly(encoder, decoder, input_lang, output_lang, eval_pairs, max_length,
                             device, n=10)

    # Final Model Evaluation
    print("\nFinal Model Evaluation:")
    evaluateRandomly(encoder, decoder, input_lang, output_lang, eval_pairs, max_length,
                     device, n=10)


def test(input_tensor, target_tensor, encoder, decoder, criterion, device):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        encoder_hidden = encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

        encoder.train()
        decoder.train()

        return loss.item() / target_length


def test_attention(input_tensor, target_tensor, encoder, decoder, criterion, max_length,
                   device):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        encoder_hidden = encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

        encoder.train()
        decoder.train()

        return loss.item() / target_length


def evaluate(encoder, decoder, input_lang, output_lang, sentence, device):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []

        # for di in range(max_length):
        while True:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        encoder.train()
        decoder.train()

        return decoded_words


def evaluate_attention(encoder, decoder, input_lang, output_lang, sentence, max_length,
                       device):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        encoder.train()
        decoder.train()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_length, device,
                     n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        if type(decoder) == AttnDecoderRNN:
            output_words, attentions = evaluate_attention(
                encoder, decoder, input_lang, output_lang, pair[0], max_length, device)
        else:
            output_words = \
                evaluate(encoder, decoder, input_lang, output_lang, pair[0], device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
