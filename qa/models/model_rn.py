from __future__ import print_function

from keras.layers import Input, Embedding, LSTM, Reshape, concatenate, regularizers, Bidirectional, Conv1D, \
    MaxPooling1D, Permute, Conv2D, MaxPooling2D, SimpleRNN
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import stack

from pypagai.models.base import KerasModel


class RNNoLSTM(KerasModel):

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 32
        config['lstm-units'] = 32

        return config

    def __init__(self, cfg):
        super().__init__(cfg)

        EMBED_SIZE = cfg['embed-size']
        LSTM_UNITS = cfg['lstm-units']

        story = Input((self._story_maxlen, self._sentences_maxlen,), name='story')
        labels = Input((self._sentences_maxlen,), name='labels')
        question = Input((self._query_maxlen,), name='question')

        story_encoded = Embedding(self._vocab_size, EMBED_SIZE)(story)
        story_encoded = Dropout(0.3)(story_encoded)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :])(story_encoded)
            fact_object = Reshape((EMBED_SIZE*self._sentences_maxlen, ))(fact_object)
            objects.append(fact_object)

        question_encoded = Embedding(self._vocab_size, EMBED_SIZE)(question)
        question_encoded = Dropout(0.3)(question_encoded)
        question_encoded = Reshape((EMBED_SIZE*self._query_maxlen, ))(question_encoded)

        relations = []
        for fact_object_1 in objects:
            for fact_object_2 in objects:
                relations.append(concatenate([fact_object_1, fact_object_2, question_encoded]))

        relations = concatenate(relations)
        response = Dense(256, activation='relu')(relations)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)

        response = Dense(256, activation='relu')(response)
        response = Dense(512, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question, labels], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class ConvQueryRN(KerasModel):

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 32
        config['lstm-units'] = 32

        return config

    def __init__(self, cfg):
        super().__init__(cfg)

        EMBED_SIZE = cfg['embed-size']
        LSTM_UNITS = cfg['lstm-units']

        story = Input((self._story_maxlen, self._sentences_maxlen,), name='story')
        labels = Input((self._sentences_maxlen,), name='labels')
        question = Input((self._query_maxlen,), name='question')

        story_encoded = Embedding(self._vocab_size, EMBED_SIZE)(story)
        story_encoded = Dropout(0.3)(story_encoded)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :])(story_encoded)
            fact_object = Reshape((EMBED_SIZE*self._sentences_maxlen, ))(fact_object)

            objects.append(fact_object)

        question_encoded = Embedding(self._vocab_size, EMBED_SIZE)(question)
        question_encoded = Dropout(0.3)(question_encoded)

        question_encoded = Conv1D(int(EMBED_SIZE/2), kernel_size=(3, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(question_encoded)
        question_encoded = MaxPooling1D(strides=(1, ))(question_encoded)

        question_encoded = Reshape((int(EMBED_SIZE/2), ))(question_encoded)

        relations = []
        for fact_object_1 in objects:
            for fact_object_2 in objects:
                relations.append(concatenate([fact_object_1, fact_object_2, question_encoded]))

        relations = concatenate(relations)

        response = Dense(256, activation='relu')(relations)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)

        response = Dense(256, activation='relu')(response)
        response = Dense(512, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question, labels], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class ConvStoryRN(KerasModel):

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 32
        config['lstm-units'] = 32

        return config

    def __init__(self, cfg):
        super().__init__(cfg)

        EMBED_SIZE = cfg['embed-size']
        LSTM_UNITS = cfg['lstm-units']

        story = Input((self._story_maxlen, self._sentences_maxlen,), name='story')
        labels = Input((self._sentences_maxlen,), name='labels')
        question = Input((self._query_maxlen,), name='question')

        story_encoded = Embedding(self._vocab_size, EMBED_SIZE)(story)
        story_encoded = Dropout(0.3)(story_encoded)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :])(story_encoded)

            fact_object = Conv1D(EMBED_SIZE, kernel_size=(5, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(fact_object)
            fact_object = MaxPooling1D(strides=(1, ))(fact_object)
            fact_object = Conv1D(int(EMBED_SIZE/2), kernel_size=(1, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(fact_object)
            fact_object = MaxPooling1D(strides=(1, ))(fact_object)

            fact_object = Reshape((int(EMBED_SIZE/2), ))(fact_object)
            objects.append(fact_object)

        question_encoded = Embedding(self._vocab_size, EMBED_SIZE)(question)
        question_encoded = Dropout(0.3)(question_encoded)
        question_encoded = Reshape((EMBED_SIZE*self._query_maxlen, ))(question_encoded)

        relations = []
        for fact_object_1 in objects:
            for fact_object_2 in objects:
                relations.append(concatenate([fact_object_1, fact_object_2, question_encoded]))

        relations = concatenate(relations)

        response = Dense(256, activation='relu')(relations)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)

        response = Dense(256, activation='relu')(response)
        response = Dense(512, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question, labels], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class ConvInputsRN(KerasModel):

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 128
        config['lstm-units'] = 32

        return config

    def __init__(self, cfg):
        super().__init__(cfg)

        EMBED_SIZE = cfg['embed-size']
        CONV_UNITS = cfg['lstm-units']

        story = Input((self._story_maxlen, self._sentences_maxlen,), name='story')
        labels = Input((self._sentences_maxlen,), name='labels')
        question = Input((self._query_maxlen,), name='question')

        story_encoded = Embedding(self._vocab_size, EMBED_SIZE)(story)
        story_encoded = Dropout(0.3)(story_encoded)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :])(story_encoded)

            fact_object = Conv1D(CONV_UNITS, kernel_size=(5, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(fact_object)
            fact_object = MaxPooling1D(strides=(1, ))(fact_object)

            fact_object = Conv1D(int(CONV_UNITS/2), kernel_size=(2, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(fact_object)
            # fact_object = MaxPooling1D(strides=(1, ))(fact_object)

            fact_object = Reshape((int(CONV_UNITS/2), ))(fact_object)
            objects.append(fact_object)

        question_encoded = Embedding(self._vocab_size, EMBED_SIZE)(question)
        question_encoded = Dropout(0.3)(question_encoded)

        question_encoded = Conv1D(int(CONV_UNITS/2), kernel_size=(3, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(question_encoded)
        question_encoded = MaxPooling1D(strides=(1, ))(question_encoded)

        question_encoded = Reshape((int(CONV_UNITS/2), ))(question_encoded)

        relations = []
        for fact_object_1 in objects:
            for fact_object_2 in objects:
                relations.append(concatenate([fact_object_1, fact_object_2, question_encoded]))

        relations = concatenate(relations)

        response = Dense(256, activation='relu')(relations)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.4)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.3)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.2)(response)

        response = Dense(256, activation='relu')(response)
        response = Dense(512, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question, labels], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class ConvRN(KerasModel):

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 128
        config['lstm-units'] = 32

        return config

    def __init__(self, cfg):
        super().__init__(cfg)

        EMBED_SIZE = cfg['embed-size']
        CONV_UNITS = cfg['lstm-units']

        story = Input((self._story_maxlen, self._sentences_maxlen,), name='story')
        labels = Input((self._sentences_maxlen,), name='labels')
        question = Input((self._query_maxlen,), name='question')

        story_encoded = Embedding(self._vocab_size, EMBED_SIZE)(story)
        story_encoded = Dropout(0.3)(story_encoded)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :])(story_encoded)

            fact_object = Conv1D(CONV_UNITS, kernel_size=(5, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(fact_object)
            fact_object = MaxPooling1D(strides=(1, ))(fact_object)

            fact_object = Conv1D(int(CONV_UNITS/2), kernel_size=(2, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(fact_object)

            objects.append(fact_object)

        question_encoded = Embedding(self._vocab_size, EMBED_SIZE)(question)
        question_encoded = Dropout(0.3)(question_encoded)

        question_encoded = Conv1D(int(CONV_UNITS/2), kernel_size=(3, ), strides=(1, ), padding='valid', kernel_initializer='normal', activation='relu')(question_encoded)
        question_encoded = MaxPooling1D(strides=(1, ))(question_encoded)

        relations = []
        for fact_object_1 in objects:
            for fact_object_2 in objects:
                relations.append(concatenate([fact_object_1, fact_object_2, question_encoded]))

        relations = concatenate(relations)

        relations = LSTM(32)(relations)

        response = Dense(256, activation='relu')(relations)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.4)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.3)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.2)(response)

        response = Dense(256, activation='relu')(response)
        response = Dense(512, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question, labels], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
