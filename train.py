import tensorflow as tf
import datetime
import yaml
import json
import time
from mymodel import create_model
import os
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from dvc.api import make_checkpoint

import dvclive
from dvclive.keras import DvcLiveCallback

weights_file = "model.h5"
summary = "summary.json"

params = yaml.safe_load(open('params.yaml'))
epochs = params['epochs']
log_file = params['log_file']
dropout = params['dropout']
lr = params['lr']

class MyCallback(Callback):
    def __init__(self, file):
        self.file = file
    def on_epoch_end(self, epoch, logs={}):
        self.model.save(self.file)
        json.dump(logs, open(summary, 'w'))
        make_checkpoint()

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = create_model(dropout)
opt = keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

if os.path.exists(weights_file):
    model.load_weights(weights_file)

start_real = time.time()
start_process = time.process_time()
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[
                        MyCallback(weights_file),
                        DvcLiveCallback()
                    ])
end_real = time.time()
end_process = time.process_time()

model.save(weights_file)

