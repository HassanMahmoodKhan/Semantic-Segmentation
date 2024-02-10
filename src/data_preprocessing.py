import tensorflow as tf
import matplotlib.pyplot as plt
import os

def resize(input_image, input_mask, shape=(128,128)):
    input_image = tf.image.resize(input_image, shape, method="nearest")
    input_mask = tf.image.resize(input_mask, shape, method="nearest")
    return input_image, input_mask

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask

def load_image_train(datapoint):
    image = datapoint['image']
    mask = datapoint['segmentation_mask']
    image, mask = resize(image, mask)
    image, mask = augment(image, mask)
    image, mask = normalize(image, mask)
    return image, mask

def load_image_test(datapoint):
    image = datapoint['image']
    mask = datapoint['segmentation_mask']
    image, mask = resize(image, mask)
    image, mask = normalize(image, mask)
    return image, mask

def display(items):
    plt.figure(figsize=(8,6))
    titles = ['Original Image', 'True Mask', 'Predicted Mask']
    for i in range(len(items)):
        plt.subplot(1,len(items), i+1)
        plt.imshow(items[i])
        plt.title(titles[i])
    plt.show()

def sample_batch(batches):
    sample_batch = next(iter(batches))
    image, mask = sample_batch[0][0], sample_batch[1][0]
    display([image,mask])

def visualize_results(results, output_path):
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('Training Vs validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid('On')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.subplot(2,1,2)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Training Vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('On')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(os.path.join(output_path, 'accuracy-loss.jpg'))