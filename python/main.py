import os
import re
import sys
import pickle

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
import numpy as np


def preprocess(text):
	text_input = re.sub(r'[^a-zA-Z1-9]+', ' ', str(text))
	output = re.sub(r'\d+', '',text_input)
	return output.lower().strip()

data = ''

folder = '../transcripts'
# for file in sorted(os.listdir(folder)):
#     print(file)
#     with open(os.path.join(folder,file)) as f:
#         # data += preprocess(f.readlines())
#         data += ''.join(f.readlines())

# print(data)
files = [os.path.join(folder,f) for f in sorted(os.listdir(folder))]
text_dataset = tf.data.TextLineDataset(['../transcripts/01_iron_man.txt'])
max_features = 5000
max_len = 500

vectorize_layer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=max_len)
vectorize_layer.adapt(text_dataset.batch(64))


input_sequences = []

with open('ngrams.txt','w') as file:
    for line in text_dataset:
        tokens = vectorize_layer(line)
        print(line)
        for i in range(1, len(tokens)):
            n_gram_sequence = tokens[:i+1]
            print(n_gram_sequence)
            input_sequences.append(n_gram_sequence)
            # file.write(np.array2string(n_gram_sequence.numpy(),separator=',')+"\n")
# sys.exit(0)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


# with open('input_sequences.pkl','wb') as file:
#     pickle.dump(input_sequences, file)
# input_sequences = pickle.load('./input_sequences.pkl')
# sys.exit(0)

# print(vectorize_layer.get_vocabulary())

model = Sequential()
# model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
# model.add(vectorize_layer)
model.add(Embedding(max_features, max_len))
model.add(Bidirectional(LSTM(150, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(max_features/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(max_features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy')>0.93):
			print("\nReached 93% accuracy so cancelling training!")
			self.model.stop_training = True

callbacks = myCallback()

predictors, labels = input_sequences[:,:-1],input_sequences[:,-1]
labels = ku.to_categorical(labels, num_classes=max_features)

history = model.fit(predictors, labels, epochs=10, verbose=1, callbacks=[callbacks])
input_data = [["doctor"], ["spider-man"]]
print(model.predict(input_data))