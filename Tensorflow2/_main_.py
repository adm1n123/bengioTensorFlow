import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from CorpusProcessor import CorpusProcessor, CustomCorpusProcessor
from BengioModel import BengioModel, BengioModelModified


N_GRAM = 4

OUTPUT_DIM = 50    # word vec dimension.
HIDDEN_NEURONS = 60  # hidden layer neurons
BATCH_SIZE = 100  # keep batch size small tensorflow may overflow memory for batch size > 10k
EPOCHS = 2


def main():
    # run_bengio_on_brown_corpus()
    # run_bengio_modified_on_custom_corpus()
    bengio_on_custom_corpus()


def run_bengio_on_brown_corpus():
    corpus = CorpusProcessor(ngram=N_GRAM)

    train_input, train_target, dev_input, dev_target = corpus.get_train_data()
    # use padding of size context_size so that no zeros are added at the end and also get the padded output matrix if
    # python list is raising error. convert to padded list.

    bengio = BengioModel(ngram=N_GRAM, corpus=corpus, output_dim=OUTPUT_DIM, hidden_neurons=HIDDEN_NEURONS)
    bengio.create_nn()

    print(bengio.model.summary())

    bengio.model.fit(
        x=train_input,
        y=train_target,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        shuffle=True,   # shuffle the input for each epoch
        validation_data=(dev_input, dev_target)
    )

    # loss, accuracy = model.evaluate(x=dev_input, y=dev_target, batch_size=100, verbose=True)
    # print("loss: {}, accuracy: {}".format(loss, accuracy))

    print("Dumping to file")
    dump_to_file(bengio, corpus, prefix="bengio_STD_")

    evaluate_context_learning(bengio, corpus)

    cos_similarities(bengio, corpus)
    run_examples(bengio, corpus)


def run_bengio_modified_on_custom_corpus():
    corpus = CustomCorpusProcessor(ngram=N_GRAM)
    new_train, new_target = corpus.get_train_data_from_sentences()

    bengio = BengioModelModified(ngram=N_GRAM, corpus=corpus, output_dim=OUTPUT_DIM, hidden_neurons=HIDDEN_NEURONS)
    bengio.create_nn()

    print(bengio.model.summary())

    bengio.model.fit(
        x=new_train,
        y=new_target,
        batch_size=1,
        epochs=5,
        verbose=1,
        shuffle=True   # shuffle the input for each epoch
    )

    print("Dumping to file")
    dump_to_file(bengio, corpus, prefix="bengio_cust_corp_")

    corpus.print_words_from_train_data(new_train, new_target)

    evaluate_context_learning(bengio, corpus)

    cos_similarities(bengio, corpus)
    run_examples(bengio, corpus)


def bengio_on_custom_corpus():
    corpus = CorpusProcessor(ngram=N_GRAM)
    print(corpus.get_vocab())
    corpus.add_words()
    print(corpus.get_vocab())

    train_input, train_target, dev_input, dev_target = corpus.get_train_data()


    custom_train, custom_target = corpus.get_train_data_from_sentences()
    corpus.print_words_from_train_data(custom_train, custom_target)

    # use padding of size context_size so that no zeros are added at the end and also get the padded output matrix if
    # python list is raising error. convert to padded list.

    bengio = BengioModel(ngram=N_GRAM, corpus=corpus, output_dim=OUTPUT_DIM, hidden_neurons=HIDDEN_NEURONS)
    bengio.create_nn()

    print(bengio.model.summary())

    bengio.model.fit(
        x=train_input,
        y=train_target,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        shuffle=True,   # shuffle the input for each epoch
        validation_data=(dev_input, dev_target)
    )

    bengio.model.fit(
        x=custom_train,
        y=custom_target,
        batch_size=1,
        epochs=5,
        verbose=1,
        shuffle=True   # shuffle the input for each epoch
    )

    print("Dumping to file")
    dump_to_file(bengio, corpus, prefix="bengio_on_custom_corpus_")

    corpus.print_words_from_train_data(custom_train, custom_target)
    evaluate_context_learning(bengio, corpus)

    fruits = 'papaya banana grapes mango'
    activity = 'plays eats runs play eat run'
    other = 'today yesterday'
    hverbs = 'do does is has have did was were had will shall'
    nouns = 'raju amit robbin nancy david john alice'

    cos_similarities(bengio, corpus, fruits)
    cos_similarities(bengio, corpus, activity)
    cos_similarities(bengio, corpus, other)
    cos_similarities(bengio, corpus, hverbs)
    cos_similarities(bengio, corpus, nouns)

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
    # eg1 = 'The cat is walking in the bedroom'.lower()
    eg2 = 'A dog was running in a room'.lower()
    sentences = [eg2]
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
        print("\nActual Sentence: {}\nPredicted sentence with context size:{} is: {}".format(sentence, bengio.context_size, predicted_sentence))


def cos_similarities(bengio, corpus, words=None):
    data = 'cat dog human male female computer keyboard walking room'.lower()
    if words is not None:
        data = words

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


def evaluate_context_learning(bengio, corpus):
    present = ['eats', 'runs', 'plays', 'running', 'playing', 'running']
    past = ['ate', 'ran', 'played', 'walking', 'eating', 'playing']
    present_word_idx = list(map(corpus.get_word2index, present))
    past_word_idx = list(map(corpus.get_word2index, past))

    present_vec = bengio.embedding_layer(tf.constant(present_word_idx)).numpy()
    past_vec = bengio.embedding_layer(tf.constant(past_word_idx)).numpy()

    diff_word = []
    diff_vec = []
    for i in range(len(present_word_idx)):
        diff_word.append(present[i]+'_vector - '+past[i]+'_vector')
        diff_vec.append(np.subtract(present_vec[i], past_vec[i]))

    for i in range(len(present)):
        for j in range(i+1, len(present)):
            print('cosine similarity: ('+diff_word[i]+').('+diff_word[j]+') is: ', cosine_similarity([diff_vec[i]], [diff_vec[j]]))




def dump_to_file(bengio, corpus, prefix=""):
    weights = bengio.model.get_layer('embedding').get_weights()[0]      # or bengio.embedding_layer
    vocab = corpus.get_vocab()
    out_v = open('data/'+prefix+'vectors.tsv', 'w', encoding='utf-8')
    out_m = open('data/'+prefix+'metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


if __name__ == '__main__':
    main()

