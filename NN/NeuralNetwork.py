"""
Neural Network Model (NN) (30 marks)

Using the same training data representation from the previous question
build a neural network based model that can re-rank passages.

You may use existing packages, namely Tensorflow or PyTorch in this subtask
Justify your choice by describing why you chose a particular architecture and how it fits to our problem.
You are allowed to use different types of neural network architectures
(e.g. feed forward, convolutional, recurrent and/or transformer based neural networks)

Using the metrics you have implemented in the first part,
report the performance of your model on the validation data.

Describe how you perform input processing, as well as the representation/features used.
Your marks for this part will depend on the appropriateness of the model you have chosen for the task,
as well as the representations/features used in training.

"""

import os

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, Conv1D, MaxPooling1D, Input, \
    LeakyReLU
from keras.metrics import Metric
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.utils import plot_model


class NDCG(Metric):
    def __init__(self, evaluator, name="ndcg", **kwargs):
        super(NDCG, self).__init__(name=name, **kwargs)
        self.evaluator = evaluator
        self.ndcg = np.zeros(3)
        self.mAP = np.zeros(3)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mAP, self.ndcg = self.evaluator(y_pred)

    def result(self):
        return {'mAP@3': self.mAP[0], 'mAP@10': self.mAP[1], 'mAP@100': self.mAP[2],
                'NDCG@3': self.ndcg[0], 'NDCG@10': self.ndcg[1], 'NDCG@100': self.ndcg[2]}

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        pass


class Network:
    def __init__(self, config, evaluator):

        self.config = config
        self.model = None
        self.loadCheckpoint = self.config['training']['loadCheckpoint']
        self.evaluator = evaluator

    def _built_model(self):
        # input image dimensions
        param_length = 1  # output param length
        input_shape = (300, 2)

        # Start Neural Network
        sequence = Input(shape=input_shape)

        x = sequence
        # CNN layer.
        for i in range(len(self.config['CNN']['layers'])):

            x = Conv1D(filters=self.config['CNN']['layers'][i],
                       kernel_size=self.config['CNN']['kernels'][i],
                       strides=self.config['CNN']['stride'][i],
                       padding='same')(x)
            if self.config['CNN']['batchNorm'][i] == 1:
                x = BatchNormalization(axis=-1)(x)
            if self.config['CNN']['activation'] == 'leaky_relu':
                x = LeakyReLU(0.2)(x)
            else:
                x = Activation(self.config['CNN']['activation'])(x)

            if self.config['CNN']['maxPool'][i] == 1:
                x = MaxPooling1D(pool_size=2)(x)

        flatten_x = Flatten()(x)

        dense_layer = Dense(self.config['CNN']['denseUnit'], activation='linear')(flatten_x)
        dense_layer = LeakyReLU(0.2)(dense_layer)
        dense_layer = Dropout(self.config['training']['dropRate'])(dense_layer)
        decision_layer = Dense(param_length, activation='linear')(dense_layer)
        return sequence, decision_layer

    def compile_model(self, checkpoint_dir=None, cv=0, ):
        sequence, decision_layer = self._built_model()

        if self.loadCheckpoint > 0:
            # checkpoint_dir = self.config['general']['checkpoint_dir']
            # cv = self.config['training']['cv']
            self.model = self.load_model(checkpoint_dir + f'ckt/checkpt_{cv}_{self.loadCheckpoint:03d}.h5')
        else:
            self.model = Model(inputs=sequence, outputs=decision_layer)
            self.model.compile(loss=self.config['training']['lossFn'],
                               optimizer=Adam(learning_rate=0.001,
                                              decay=10 ** self.config['training']['decay']),
                               metrics=[NDCG(evaluator=self.evaluator)]
                               )
        self.model.summary()
        return self.model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=64, checkpoint_dir='./',
                    cv_order=0):
        print('training begins')
        # make sure no prior graph exists
        tf.keras.backend.clear_session()

        # create log folder and checkpoint folder
        os.makedirs(os.path.join(checkpoint_dir, 'history'), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'ckt'), exist_ok=True)

        # callback
        # initialise model checkpoint
        filename = f'ckt/checkpt_{cv_order}' + '_{epoch:03d}.h5'
        model_cktpt = ModelCheckpoint(os.path.join(checkpoint_dir, filename),
                                      monitor='val_loss',
                                      mode='min',
                                      verbose=0,
                                      save_freq=1)
        tbCallBack = TensorBoard(log_dir=f'./runs/{cv_order}',
                                 write_graph=True,
                                 write_images=True,
                                 update_freq='epoch',
                                 profile_batch=32,
                                 embeddings_freq=10)
        callbacks = [model_cktpt, tbCallBack]
        self.compile_model(checkpoint_dir, cv=cv_order)

        # display network
        if self.config['general']['displayNet']:
            plot_model(self.model, to_file=os.path.join(
                checkpoint_dir, 'model.png'), show_shapes=True)

        # ensure they have the right shape

        X_train = X_train.reshape(-1, 300, 2)
        y_train = y_train.reshape(-1)

        # trainings
        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       initial_epoch=self.loadCheckpoint if self.loadCheckpoint > 0 else 0,
                       shuffle=True,

                       validation_data=(X_val, y_val),
                       validation_steps=1,

                       verbose=1,
                       workers=4,
                       use_multiprocessing=True,

                       callbacks=callbacks,
                       )

    def load_model(self, checkpoint_dir_path):
        print(f'loading model from {checkpoint_dir_path}')
        self.model = load_model(checkpoint_dir_path)
        return self.model
