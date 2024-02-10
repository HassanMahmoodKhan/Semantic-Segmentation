import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import tensorflow_datasets as tfds
import argparse
import os
import logging
import time
from data_preprocessing import *
from train import *
from test import *

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

working_dir = os.getcwd()
os.chdir(working_dir)
data_dir = os.path.join(working_dir, 'data')
save_dir = os.path.join(working_dir,'models')
fig_path = os.path.join(working_dir,'assets')

if __name__=='__main__':
    # Execute main block if script run directly
    print("Start")
    parser = argparse.ArgumentParser(description="A script for semantic segmentation of images")
    parser.add_argument("--output", type=str, help="Path to the output file", default=".\output")
    parser.add_argument("--verbose", help="Enable verbose mode")
    parser.add_argument("--shape", type=tuple, help="Shape of the image/mask e.g. (128,128) or (256,256), etc.", default=(128,128))
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument("--batch_size", type=int, help="The batch size for model building", default=64)
    parser.add_argument("--learning_rate", type=float, help="The optimizer learing rate [0,1]", default=0.001)
    parser.add_argument("--dropout", type=float, help="The model dropout rate [0,1]", default=0.1)

    args = parser.parse_args()

    output_path = args.output
    verbose = args.verbose
    input_shape = args.shape
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    dropout = args.dropout
    
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, 'output.log')
    logging.basicConfig(filename=log_path,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
 
    logging.info(f"Creating data directory at {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    logging.info("Dowloading and loading dataset")
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=data_dir)
    assert isinstance(dataset, dict)
    assert isinstance(info, tfds.core.DatasetInfo)
    print(info)  

    logging.info("Loading train and test datasets")
    train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

    logging.info("Splitting dataset into training, validation and test batches")
    train_batches = train_dataset.batch(batch_size).repeat()
    val_batches = test_dataset.take(3000).batch(batch_size)
    test_batches = test_dataset.skip(3000).take(669).batch(batch_size)

    logging.info("Displaying sample image and its corressponding mask")
    # sample_batch(train_batches)

    logging.info("Building/Creating Unet model")
    model = build_unet()
    model.summary()

    os.makedirs(save_dir, exist_ok=True)
    model_weights = os.path.join(save_dir, "unet_best.keras")
    print(model_weights)
    if not os.path.exists(model_weights):
        logging.info(f"Loading model weights from {model_weights}")
        loaded_model = load_model(model_weights)
        logging.info(" Performing model inference with test set")
        model_history = test(loaded_model, test_batches)
        logging.info(f"Metrics: {loaded_model.metrics_names}")
        logging.info(f"Evaluation results: {model_history}")

    else:
        logging.info("Compiling model")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
        
        train_length = info.splits["train"].num_examples
        steps_per_epoch = train_length // batch_size
        logging.info(f"Training model for {epochs} epochs.")
        start_time = time.perf_counter()
        model_results = train(model, model_weights, train_batches, val_batches, epochs, steps_per_epoch)
        end_time = time.perf_counter()
        logging.info(f"Model training complete in {end_time-start_time} secs")
        os.makedirs(fig_path, exist_ok=True)
        logging.info(f"Saving accuracy and loss plots to {fig_path}")
        visualize_results(model_results, fig_path)
        logging.info(" Performing model inference with test set")
        model_history = test(model, test_batches)
        logging.info(f"Metrics: {model.metrics_names}")
        logging.info(f"Evaluation results: {model_history}")

    logging.info("Script executed successfully!")
    print("End")
