import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers

def conv2D_block(n_filters, inputs):
  # Conv2D followed by a Relu activation
  x = layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer = "he_normal" )(inputs)
  x = layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer = "he_normal" )(x)
  return x

def build_unet(input_size=(128,128,3), dropout=0.1):
# Contracting path; downsampling to extract high-level abstract features (encoder)
    inputs = layers.Input(shape=input_size) # The input image shape is 128x128x3
    c1 = conv2D_block(64, inputs)
    p1 = layers.Dropout(dropout)(c1)
    p1 = layers.MaxPooling2D((2, 2))(p1)

    c2 = conv2D_block(128, p1)
    p2 = layers.Dropout(dropout)(c2)
    p2 = layers.MaxPooling2D((2, 2))(p2)

    c3 = conv2D_block(256, p2)
    p3 = layers.Dropout(dropout)(c3)
    p3 = layers.MaxPooling2D((2, 2))(p3)

    c4 = conv2D_block(512, p3)
    p4 = layers.Dropout(dropout)(c4)
    p4 = layers.MaxPooling2D((2, 2))(p4)

    c5 = conv2D_block(1024, p4)

    # Expansive path; upsampling to extract low-level abstract features (decoder)
    u6 = layers.Conv2DTranspose(512, 3, strides=(2,2), padding='same')(c5) # Upsample
    u6 = layers.concatenate([u6, c4]) # Add skip connection
    u6 = layers.Dropout(dropout)(u6)
    c6 = conv2D_block(512, u6)

    u7 = layers.Conv2DTranspose(256, 3, strides= (2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(dropout)(u7)
    c7 = conv2D_block(256, u7)

    u8 = layers.Conv2DTranspose(128,3,strides=(2,2),padding='same')(c7)
    u8 = layers.concatenate([u8,c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = conv2D_block(128,u8)

    u9 = layers.Conv2DTranspose(64,3,strides=(2,2),padding='same')(c8)
    u9 = layers.concatenate([u9,c1])
    u9 = layers.Dropout(dropout)(u9)
    c9 = conv2D_block(64,u9)

    outputs = layers.Conv2D(3, 1, activation='softmax')(c9) # Output layer

    # The output has three channels corresponding to the three classes that the model will classify 
    # each pixel for: background, foreground object, and object outline.

    model = Model(inputs, outputs, name="U-Net")
    return model

def train(model, model_weights, train_batches, val_batches, n_epochs, steps_per_epoch):

    checkpoint = keras.callbacks.ModelCheckpoint(model_weights,
                                                    monitor='accuracy',
                                                    verbose=1,
                                                    mode='max',
                                                    save_best_only=True)
    
    early = tf.keras.callbacks.EarlyStopping(monitor="accuracy", mode="max", restore_best_weights=True, patience=3)
    callbacks_list = [checkpoint, early]
    
    model_history = model.fit(train_batches,
                                epochs=n_epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_batches,
                                callbacks=callbacks_list)
    return model_history