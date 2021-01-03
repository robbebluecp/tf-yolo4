"""
training part.
this is the entrance for training
"""

import loss
import config
import models
from tensorflow import keras
from generator import data_generator


class_mapping = dict(enumerate(config.classes_names))
class_mapping = {class_mapping[key]: key for key in class_mapping}


model_yolo = models.YOLO(pre_train=None)()

f = open(config.label_path)
label_lines = f.readlines()

train_lines = label_lines[:-int(len(label_lines) * config.validation_split)]
valid_lines = label_lines[-int(len(label_lines) * config.validation_split):]

h, w = config.image_input_shape
y_true = [keras.layers.Input(shape=(h // config.scale_size[l], w // config.scale_size[l], config.num_anchors, config.num_classes + 5)) for l
          in range(3)]

model_loss = keras.layers.Lambda(function=loss.yolo4_loss, output_shape=(1,), name='yolo_loss')([*model_yolo.output, *y_true])

tensorboard = keras.callbacks.TensorBoard()
checkpoint = keras.callbacks.ModelCheckpoint(filepath='model_train/ep{epoch:03d}-loss{loss:.3f}-valloss{val_loss:.3f}.h5',
                                             monitor='val_loss',
                                             save_weights_only=True,
                                             save_best_only=True,
                                             period=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)

model = keras.models.Model([model_yolo.input, *y_true], model_loss)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})


g_train = data_generator(label_lines=train_lines,
                         batch_size=config.batch_size,
                         input_shape=config.image_input_shape,
                         anchors=config.anchors,
                         num_classes=config.num_classes)

g_valid = data_generator(label_lines=valid_lines,
                         batch_size=config.batch_size,
                         input_shape=config.image_input_shape,
                         anchors=config.anchors,
                         num_classes=config.num_classes)
print('And so it begins..... To train!')
model.fit(g_train,
          validation_data=g_valid,
          steps_per_epoch=len(label_lines) // config.batch_size,
          validation_steps=int(len(label_lines) * config.validation_split * 0.2),
          epochs=config.epochs,
          callbacks=[tensorboard, checkpoint, reduce_lr])

model_yolo.save_weights('model_train/model_train_final.weights')
