import tensorflow as tf
from keras.layers import Embedding , LSTM ,Attention , InputLayer ,Concatenate ,TimeDistributed
from keras.models import Sequential

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
embedding_vector_features=100
voc_size=10000
sent_length=62
ten = [62]

def model_():
    input_1 = Input(shape=ten)
    embedding = Embedding(voc_size,embedding_vector_features,input_length=sent_length)(input_1)
    lstm = LSTM(500,return_sequences=True)(embedding)
    input_2 = Input(shape=[None,])
    lstm_1 = LSTM(500,return_sequences=True)(lstm)
    embedding_1 = Embedding(voc_size,embedding_vector_features,input_length=sent_length)(input_2)
    lstm_2 = LSTM(500,return_sequences=True)(lstm_1)
    lstm_3 = LSTM(500,return_sequences=True,return_state=True)(embedding_1)
    attention_layer = Attention([lstm_2,lstm_3])
    concat_layer = Concatenate([lstm_2,lstm_3])
    time_distributed = TimeDistributed(layer=concat_layer)
    t_d = TimeDistributed(Dense(16670))(lstm_2
                                   )
    model = Model(inputs=[input_1,input_2],outputs=t_d)
    model.compile(optimizer='adam', loss='mse')
    return model
model = model_()
print(model.summary(
))