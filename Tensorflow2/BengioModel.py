import tensorflow as tf


class BengioModel:
    def __init__(self, ngram, corpus, output_dim, hidden_neurons):
        self.corpus = corpus
        self.ngram = ngram
        self.context_size = self.ngram - 1  # last n - 1 words used as contexts.
        self.input_dim = self.corpus.vocab_len    # one-hot encoding vector length (size of vocabulary)
        self.output_dim = output_dim     # word vector dimension.
        self.input_length = self.context_size    # context size(no of previous words)
        self.hidden_neurons = hidden_neurons    # number of neurons in hidden layer.
        self.output_neurons = self.corpus.vocab_len   # number of neurons in output layer i.e. |V|. each neuron is a class.
        self.model = None
        self.embedding_layer = None

    def create_nn(self):
        self.model = tf.keras.Sequential()  # creating neural network model.
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(-1, 1),
            mask_zero=False,
            input_length=self.input_length,
            trainable=True,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=.001)     # It is for weighs(word vectors). l2=.01 is constant multiplied to loss its hyperparameter.
        )

        self.model.add(self.embedding_layer)  # add the bottom layer(embedding layer).

        # embedding layer takes sequence of indexes of words and output sequence of word vectors. so we need to flatten the output to one vector.
        self.model.add(tf.keras.layers.Flatten())    # embedding layer output 2D matrix, sequence of word vectors so covert that to 1D vector.

        # add hidden layer
        self.model.add(tf.keras.layers.Dense(
            units=self.hidden_neurons,
            activation=tf.keras.activations.tanh,
            kernel_regularizer=tf.keras.regularizers.l2(l2=.001),    # It is for hidden layer weights i.e. H not biases
            use_bias=True)
        )

        # add output layer
        self.model.add(tf.keras.layers.Dense(
            units=self.output_neurons,
            activation=tf.keras.activations.softmax,
            kernel_regularizer=tf.keras.regularizers.l2(l2=.001),    # It is for output layer weights i.e. U not biases
            use_bias=True)
        )

        # SparseCategoricalCrossentropy/CategoricalCrossentropy loss is supposed to be calculated from probability
        # If you don't use softmax at output layer and supply from_logits=False it will treat output as probability so entire calculated will be messed up. because some of the -ve values would be passed in log(x).
        # if passing one-hot encoding target eg. [0, 0, 0, 1, 0, 0, 0, 0] for 4th word as next word use loss=categorical_crossentropy.
        # if you are passing target word index(expected class index of output) use loss space_categorical_crossentropy.
        # NOTE: space_categorical_crossentropy is function which required y, y_hat and SparseCategoricalCrossentropy() is class object which can be used in model.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # default is false no need to pass.

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['accuracy']
        )
        return None

    def add_softmax(self):
        self.model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
        return None

