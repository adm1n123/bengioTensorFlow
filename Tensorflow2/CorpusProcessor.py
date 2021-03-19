import nltk
import numpy as np

from nltk.corpus import brown
from nltk.corpus import wordnet

nltk.download("brown")
nltk.download("wordnet")


class CorpusProcessor:

    def __init__(self, ngram):
        self.ngram = ngram
        self.context_size = self.ngram - 1
        self.word2index = {}
        self.index2word = {}
        self.UNKNOWN_SYMBOL = '<UNK>'
        self.vocab = {self.UNKNOWN_SYMBOL}      # put the <UNK> symbol in vocab.
        self.vocab_len = None
        self.TRAIN_SIZE = 12000

    def set_train_size(self, percent):
        paras = len(brown.paras())
        self.TRAIN_SIZE = int(paras * percent / 100)  # 70 % training.

    def get_index2word(self, idx):
        if idx >= len(self.vocab):
            print("word index out of range")
            exit(1)
        return self.index2word[idx]

    def get_word2index(self, word):
        unk_index = self.word2index[self.UNKNOWN_SYMBOL]
        return self.word2index.get(word, unk_index)  # if word is not in vocabulary return unknown index.

    def get_train_data(self):
        print("Number of paragraphs ", len(brown.paras()))

        train_brown_corpus = []

        train_input = []
        train_target = []
        dev_input = []
        dev_target = []
        word_frequency = {}  # creating dictionary
        for para_id, paragraph in enumerate(brown.paras()):  # paragraph is list of sentence and sentence is list of words.
            if para_id == self.TRAIN_SIZE:
                break
            para_words = []
            for sentence in paragraph:
                for word in sentence:
                    word_lower = word.lower()
                    para_words.append(word_lower)
                    word_frequency[word_lower] = word_frequency.get(word_lower, 0) + 1
            train_brown_corpus.append(para_words)  # appending(not extending) list at the end.

        frequent_words = set(word for word, frequency in word_frequency.items() if frequency >= 5)

        self.vocab = self.vocab.union(frequent_words)   # adding frequents words to vocab NOTE: <UNK> is already added so do union.
        self.vocab_len = len(self.vocab)

        print("Vocabulary size: ", len(self.vocab))
        # self.list_to_file(vocab, 'data/vocabulary.txt')

        self.word2index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index2word = {idx: word for idx, word in enumerate(self.vocab)}

        # dict_to_file(word_index, "data/word_index.txt")

        for para_id, paragraph in enumerate(brown.paras()):
            for sentence in paragraph:
                for idx, word in enumerate(sentence):
                    if idx + self.context_size >= len(sentence):
                        break
                    context_words = [word.lower() for word in sentence[idx:idx+self.context_size]]
                    context_words_idx = list(map(self.get_word2index, context_words))
                    target_word_idx = [self.get_word2index(sentence[idx+self.context_size].lower())]
                    if para_id < self.TRAIN_SIZE:
                        train_input.append(context_words_idx)
                        train_target.append(target_word_idx)
                    else:
                        dev_input.append(context_words_idx)
                        dev_target.append(target_word_idx)

        train_input = np.array(train_input)     # 2D list row size is self.context_size
        train_target = np.array(train_target)   # 2D list row size is 1.(target word). number of rows in input and target should be same.
        dev_input = np.array(dev_input)
        dev_target = np.array(dev_target)

        print("train_input: ", train_input.shape, " train_target: ", train_target.shape, " \ndev_input: ",
              dev_input.shape, " dev_target: ", dev_target.shape)

        return train_input, train_target, dev_input, dev_target

    def get_train_data_sentence(self, sent=None):
        sentence = 'The cat is walking in the bedroom'.split(" ")
        if sent is not None:
            sentence = sent.lower()

        input_data = []
        target_data = []
        for idx, word in enumerate(sentence):
            if idx + self.context_size >= len(sentence):
                break
            context_words = [word.lower() for word in sentence[idx:idx + self.context_size]]
            context_words_idx = list(map(self.get_word2index, context_words))

            target_word_idx = [self.get_word2index(sentence[idx + self.context_size].lower())]
            input_data.append(context_words_idx)
            target_data.append(target_word_idx)
        return input_data, target_data

    def get_vocab(self):
        return self.vocab

    def get_corpus_paras(self):
        return brown.paras()

    def list_to_file(self, mylist, filename):
        with open(filename, 'w') as file:
            for element in mylist:
                file.write('%s\n' % element)

    def dict_to_file(self, mydict, filename):
        with open(filename, 'w') as file:
            for key, value in mydict.items():
                file.write(str(value)+':  '+str(key)+'\n')
