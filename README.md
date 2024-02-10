# Semantic Segmentation with Fully Convolutional Neural Network (U-Net)

A TensorFlow based project for semantic segmentation with the `Oxford-IIIT Pet Dataset`; a 37 category pet dataset with roughly 200 images for each class with pixel level trimap segmentation.

![Model Architecture](<assets/U-Net Architecture.png>)

## Features

- Load and transform the dataset
- Resize rgb images and single channel segmentation masks to 128x128 pixels
- Build the U-Net model with the encoder and decoder
- Train the model and monitor evaluation metrics i.e., accuracy and categorical cross-entropy loss
- Save model weights for the best results
- Load and evaluate model using the test set

## Getting Started

1) Open a terminal or comand prompt and create a conda virtual environment using the `environment.yml` file:
    ```
    conda env create -f environment.yml 
    ```
2) Activate the environment:
    ```
    conda activate semantic_segmentation
    ```
3) Clone the git repository:
    ```
    git clone https://github.com/HassanMahmoodKhan/Semantic-Segmentation.git
    ```
4) Run the project:
    ```
    python src/main.py
    ```
5) If you wish to install the project dependencies using the `requirements.txt` file. Use the `pip` package manager:
    ```
    pip install -r requirements.txt
    ```

To view the output log file, refer to the `output.log` in the output folder.

To view the training and validation accuracy and loss figure, refer to the assets folder.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Acknowledgements

- Novikov, A. A., Lenis, D., Major, D., Hladůvka, J., Wimmer, M., & Bühler, K. (2017). Fully Convolutional Architectures for Multi-Class Segmentation in Chest Radiographs. ArXiv. /abs/1701.08816
