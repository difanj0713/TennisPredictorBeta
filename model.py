import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, LSTMCell, RNN, Bidirectional, concatenate
import tensorflow as tf

class BiLSTMSeq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, **kwargs):
        super(BiLSTMSeq2SeqModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Define the encoder
        self.encoder_inputs = tf.keras.Input(shape=(None,))
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_dim, return_sequences=True, return_state=True))

        # Define the decoder
        self.decoder_inputs = tf.keras.Input(shape=(None,))
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.decoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_dim, return_sequences=True, return_state=True))
        self.decoder_dense = tf.keras.layers.Dense(units=vocab_size, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        encoder_inputs, decoder_inputs = inputs

        # Encode the input sequence
        encoder_outputs = self.encoder_embedding(encoder_inputs)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm(encoder_outputs)
        encoder_states = [tf.keras.layers.Concatenate()([forward_h, backward_h]),
                          tf.keras.layers.Concatenate()([forward_c, backward_c])]

        # Decode the input sequence
        decoder_outputs = self.decoder_embedding(decoder_inputs)
        decoder_outputs, forward_h, forward_c, backward_h, backward_c = self.decoder_lstm(decoder_outputs,
                                                                                          initial_state=encoder_states)
        decoder_states = [tf.keras.layers.Concatenate()([forward_h, backward_h]),
                          tf.keras.layers.Concatenate()([forward_c, backward_c])]
        decoder_outputs = self.decoder_dense(decoder_outputs)

        return decoder_outputs


if __name__ == "__main__":
    vocab_size = 100
    embedding_dim = 10
    hidden_dim = 10
    batch_size = 32
    sequence_length = 5

    # Create a model instance
    model = BiLSTMSeq2SeqModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    # Generate some dummy input data
    encoder_inputs = np.random.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    decoder_inputs = np.random.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))

    # Pass the input data to the model
    outputs = model((encoder_inputs, decoder_inputs))

    # Print the output shape
    print(outputs.shape)
