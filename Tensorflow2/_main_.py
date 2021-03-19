import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from CorpusProcessor import CorpusProcessor
from BengioModel import BengioModel

N_GRAM = 5


def main():

    corpus = CorpusProcessor(ngram=N_GRAM)
    train_input, train_target, dev_input, dev_target = corpus.get_train_data()
    # use padding of size context_size so that no zeros are added at the end and also get the padded output matrix if
    # python list is raising error. convert to padded list.

    bengio = BengioModel(ngram=N_GRAM, corpus=corpus)
    bengio.create_nn()

    model = bengio.model
    print(model.summary())

    model.fit(x=train_input, y=train_target, batch_size=100, epochs=2, verbose=1)
    # loss, accuracy = model.evaluate(x=dev_input, y=dev_target, batch_size=100, verbose=True)
    # print("loss: {}, accuracy: {}".format(loss, accuracy))

    cos_similarities(bengio, corpus)
    run_examples(bengio, corpus)


def predict_one(model, input_list):
    # since predict takes batch input convert input list to 2D list then pass with batchsize=1 or use model(input_list).
    words_probability_batch = model.predict([input_list], batch_size=1)  # predict/fit/evaluate only takes batch input(list of list) for single input use self.model(input).
    words_probability = words_probability_batch[0]  # output is batch output return first.
    return np.argmax(words_probability)  # return the index of word with highest probability.
    # predicted_word_vector = self.model(input_list, training=False)


def run_examples(bengio, corpus):
    # pass context_size no. of words and compare the predicted output with actual sentence.
    # this method will take window of size = context_size and predict the next word and slide window by one to predict the next words so on till end of sentence
    eg1 = 'The cat is walking in the bedroom'.lower()
    eg2 = 'A dog was running in a room'.lower()
    sentences = [eg1]
    for sentence in sentences:
        words = sentence.split(" ")
        words_ids = list(map(corpus.get_word2index, words))
        predicted_list = words_ids[:bengio.context_size]  # store n-1 words since there is no way to predict them.
        context_words = words_ids[:bengio.context_size]  # context window

        for target_word in words_ids[bengio.context_size:]:  # take each word of sentence.
            predicted_word_idx = predict_one(bengio.model, context_words)
            predicted_list.append(predicted_word_idx)
            context_words.pop(0)  # delete first word of context
            context_words.append(target_word)  # add next word to context.

        predicted_sentence = ' '.join(list(map(corpus.get_index2word, predicted_list)))
        print("Actual Sentence: {}\nPredicted sentence with context size:{} is: {}".format(sentence, bengio.context_size, predicted_sentence))


def cos_similarities(bengio, corpus):
    data = 'cat dog human male female computer keyboard walking room'.lower()
    words = data.split(" ")
    words_ids = list(map(corpus.get_word2index, words))
    word_vectors = bengio.embedding_layer(tf.constant(words_ids)).numpy()  # get the words vector from embedding layer

    for i in range(len(words)):
        for j in range(len(words)):
            if i >= j:
                continue
            print("word pair: [{}, {}], Cosine Similarity: {}".format(words[i], words[j], cosine_similarity([word_vectors[i]], [word_vectors[j]])))
            # print("word1: {}, word vector: {}\nword2: {}, word vector: {}".format(words[i], word_vectors[i], words[j], word_vectors[j]))


def generate_sentence(bengio, corpus, context=None):
    context = 'The cat is walking in the bedroom'.lower()
    word_ids = list(map(corpus.get_word2index, context.split(" ")))
    if len(word_ids) < bengio.context_size:
        print("need more context words")
        return
    predicted_list = word_ids[:bengio.context_size]


if __name__ == '__main__':
    main()

