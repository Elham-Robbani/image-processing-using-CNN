# image-processing-using-CNN


This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras for image classification. The model is designed to classify images into predefined categories, leveraging deep learning techniques.

## Features
- Utilizes TensorFlow and Keras for model creation and training.
- Implements key CNN layers such as Conv2D, MaxPooling2D, and Dense.
- Includes dropout and batch normalization to improve model performance and prevent overfitting.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- scikit-learn (optional, for performance evaluation)

Install dependencies with:
bash
pip install tensorflow numpy matplotlib scikit-learn


## Dataset
The project expects an image dataset organized in a directory structure:

- dataset/
  - train/
    - class1/
    - class2/
  - test/
    - class1/
    - class2/

Replace class1 and class2 with your actual class names. Update the dataset paths in the notebook accordingly.

## Model Architecture
The CNN is built using the following layers:
1. Convolutional layers with ReLU activation
2. MaxPooling layers for downsampling
3. Flatten layer to convert 2D features into a 1D vector
4. Dense layers for classification
5. Dropout layers to reduce overfitting
6. Batch normalization for faster convergence

## Usage
1. Clone the repository and navigate to the project directory.
2. Prepare your dataset in the required format.
3. Open the Jupyter Notebook file CNN for Image Classification.ipynb.
4. Execute the cells sequentially to:
   - Load and preprocess the dataset.
   - Build and compile the CNN model.
   - Train the model and evaluate its performance.

## Testing
- We have tested the model with 2 images, which we have them in the single_prediction folder. The test is passed, it recognized the dog and the cat

## Results
Add your model's performance metrics here, such as:
- Accuracy: 85.5%

Include confusion matrices, loss graphs, or example predictions as applicable.

## Links
- [Dateset URl link](https://drive.google.com/drive/folders/18kJQaZ7fjqd9CWchJXKu8jys7Pj0fN4T?usp=sharing)
- [My Medium Blog for this project](https://medium.com/@mdelhamrobbani/image-classification-using-cnn-d9f8271db4ec)

## Acknowledgments
- TensorFlow/Keras for providing a robust deep learning framework.
- The dataset providers (add details here if using an external dataset).
