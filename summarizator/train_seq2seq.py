"""Train a sequence to sequence model.

This script is sourced from Siraj Rival
https://github.com/llSourcell/How_to_make_a_text_summarizer/blob/master/train.ipynb
"""
import os
import time
import random
import argparse
import json

import numpy as np
from keras.callbacks import TensorBoard

import config
from sample_gen import gensamples
from utils import prt, load_embedding, process_vocab, load_split_data
from model import create_model, inspect_model
from generate import gen
from constants import WEIGHTS_FILENAME, seed, nb_unknown_words


def main():
    # python summarizator/train_seq2seq.py --batch-size 10 --epochs 100 --nsamples 50 --temperature 0.2 --lr 0.001
    # python train_seq2seq.py --batch-size 1000 --epochs 10 --nsamples 5000 --temperature 0.2 --lr 0.001
    # python tg/summarizator/train_seq2seq.py --batch-size 10 --epochs 100 --nsamples 50 --temperature 0.2 --lr 0.001 --warm-start

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1000, help='input batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--rnn-size', type=int, default=512, help='size of RNN layers')
    parser.add_argument('--rnn-layers', type=int, default=3, help='number of RNN layers')
    parser.add_argument('--nsamples', type=int, default=5000, help='number of samples per epoch')
    parser.add_argument('--val-percent', type=int, default=0.1, help='percent of data validate on')
    parser.add_argument('--nflips', type=int, default=0, help='number of flips')
    parser.add_argument('--temperature', type=float, default=.2, help='RNN temperature')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0001')
    parser.add_argument('--warm-start', action='store_true')
    args = parser.parse_args()
    batch_size = args.batch_size

    # set sample sizes
    nb_train_samples = int((1 - args.val_percent) * args.nsamples)  # num training samples
    nb_val_samples = int(args.val_percent * args.nsamples)  # num validation samples
    
    # seed weight initialization
    random.seed(seed)
    np.random.seed(seed)

    embedding, idx2word, word2idx, glove_idx2idx = load_embedding(nb_unknown_words)
    vocab_size, embedding_size = embedding.shape
    oov0 = vocab_size - nb_unknown_words
    idx2word = process_vocab(idx2word, vocab_size, oov0, nb_unknown_words)
    X_train, X_test, Y_train, Y_test = load_split_data(nb_train_samples + nb_val_samples, seed)
    print("X train: {}, X test: {}\nY train: {}, Y test: {}\n".format(len(X_train), len(X_test), len(Y_train), len(Y_test)))

    print('Random head, description:')
    i = random.randint(1, 1000)
    prt('H', Y_train[i], idx2word)
    prt('D', X_train[i], idx2word)

    # save model initialization parameters
    model_params = (dict(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        LR=args.lr,
        rnn_layers=args.rnn_layers,
        rnn_size=args.rnn_size,
    ))

    _m_init_tmp = config.path_models + '/' + 'model_params.json'

    try:

        with open(_m_init_tmp, 'w') as f:
            json.dump(model_params, f)
    except Exception as e:
        print("Unable to save init model parameters to file {}".format(_m_init_tmp))
        print(e)

    model = create_model(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        LR=args.lr,
        embedding=embedding,
        rnn_layers=args.rnn_layers,
        rnn_size=args.rnn_size,
    )
    inspect_model(model)

    # load pre-trained model weights
    pretrained_weights_filename = config.path_models + '/' + '{}.hdf5'.format(WEIGHTS_FILENAME)

    if args.warm_start and WEIGHTS_FILENAME and os.path.exists(pretrained_weights_filename):
        model.load_weights(pretrained_weights_filename)
        print('Model weights loaded from {}'.format(pretrained_weights_filename))

    # print samples before training
    gensamples(
        skips=2,
        k=10,
        batch_size=batch_size,
        short=False,
        temperature=args.temperature,
        use_unk=True,
        model=model,
        data=(X_test, Y_test),
        idx2word=idx2word,
        oov0=oov0,
        glove_idx2idx=glove_idx2idx,
        vocab_size=vocab_size,
        nb_unknown_words=nb_unknown_words,
    )

    # get train and validation generators
    r = next(
        gen(X_train,
            Y_train,
            batch_size=batch_size,
            nb_batches=None,
            nflips=None,
            model=None,
            debug=False,
            oov0=oov0,
            glove_idx2idx=glove_idx2idx,
            vocab_size=vocab_size,
            nb_unknown_words=nb_unknown_words,
            idx2word=idx2word)
    )

    traingen = gen(X_train,
                   Y_train,
                   batch_size=batch_size,
                   nb_batches=None,
                   nflips=args.nflips,
                   model=model,
                   debug=False,
                   oov0=oov0,
                   glove_idx2idx=glove_idx2idx,
                   vocab_size=vocab_size,
                   nb_unknown_words=nb_unknown_words,
                   idx2word=idx2word)

    valgen = gen(X_test,
                 Y_test,
                 batch_size=batch_size,
                 nb_batches=nb_val_samples // batch_size,
                 nflips=None,
                 model=None,
                 debug=False,
                 oov0=oov0,
                 glove_idx2idx=glove_idx2idx,
                 vocab_size=vocab_size,
                 nb_unknown_words=nb_unknown_words,
                 idx2word=idx2word)

    # define callbacks for training
    callbacks = [TensorBoard(
        log_dir=config.path_logs + '/' + str(time.time()),
        histogram_freq=0,
        write_graph=False,
        write_images=False)]

    # train model and save weights
    h = model.fit_generator(
        traingen,
        samples_per_epoch=nb_train_samples,
        nb_epoch=args.epochs,
        validation_data=valgen,
        nb_val_samples=nb_val_samples,
        callbacks=callbacks,
    )
    try:
        prefix_model_name = "batch_{}_epochs_{}_rnn_size_{}".format(args.batch_size, args.epochs, args.rnn_size)
        model.save_weights(pretrained_weights_filename + "_" + prefix_model_name)#, overwrite=True)
        print('Model weights saved to {}'.format(pretrained_weights_filename))

    except Exception as e:
        print("Unable to save model weights to file {}".format(pretrained_weights_filename))
        print(e)

    # print samples after training
    gensamples(
        skips=2,
        k=10,
        batch_size=batch_size,
        short=False,
        temperature=args.temperature,
        use_unk=True,
        model=model,
        data=(X_test, Y_test),
        idx2word=idx2word,
        oov0=oov0,
        glove_idx2idx=glove_idx2idx,
        vocab_size=vocab_size,
        nb_unknown_words=nb_unknown_words,
    )


if __name__ == "__main__":
    main()
