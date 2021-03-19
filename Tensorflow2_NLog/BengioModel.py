import tensorflow as tf


class BengioModel:
    def __init__(self, ngram, corpus):
        self.corpus = corpus
        self.ngram = ngram
        self.context_size = self.ngram - 1  # last n - 1 words used as contexts.
        self.input_dim = self.corpus.vocab_len    # one-hot encoding vector length (size of vocabulary)
        self.output_dim = 5     # word vector dimension.
        self.input_length = self.context_size    # context size(no of previous words)
        self.hidden_neurons = 5    # number of neurons in hidden layer.
        self.output_neurons = self.corpus.vocab_len   # number of neurons in output layer i.e. |V|. each neuron is a class.
        self.model = None
        self.embedding_layer = None

    def create_nn(self):
        self.model = tf.keras.Sequential()  # creating neural network model.
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            embeddings_initializer=tf.keras.initializers.random_uniform,
            mask_zero=False,
            input_length=self.input_length,
            trainable=True
        )

        self.model.add(self.embedding_layer)  # add the bottom layer(embedding layer).

        # embedding layer takes sequence of indexes of words and output sequence of word vectors. so we need to flatten the output to one vector.
        self.model.add(tf.keras.layers.Flatten())    # embedding layer output 2D matrix, sequence of word vectors so covert that to 1D vector.

        # add hidden layer
        self.model.add(tf.keras.layers.Dense(
            units=self.hidden_neurons,
            activation=tf.keras.activations.tanh,
            use_bias=True)
        )

        # add output layer
        self.model.add(tf.keras.layers.Dense(
            units=self.output_neurons,
            activation=tf.keras.activations.softmax,
            use_bias=True)
        )

        # if passing one-hot encoding target eg. [0, 0, 0, 1, 0, 0, 0, 0] for 4th word as next word use loss=sparse_categorical_crossentropy.
        # if you are passing target word index use loss categorical
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',   # taking loss after softmax. logits is output before activation of output layer(softmax) so loss is calculated based on output before softmax.
            metrics=['accuracy']
        )
        return None

