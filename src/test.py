import tensorflow as tf
from data_preprocessing import *

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask[0]

def show_predictions(model, dataset=None, num=2):
 if dataset:
   for image, mask in dataset.take(num):
     pred_mask = model.predict(image)
     display([image[0], mask[0], create_mask(pred_mask)])
   
def test(model, data):
    model_history = model.evaluate(data)
    show_predictions(model, data)
    return model_history
     
