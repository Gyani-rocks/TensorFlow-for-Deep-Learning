# TensorFlow-for-Deep-Learning
**TensorFlow** is a Google-created software library that's used for deep learning and machine learning. It's a popular choice for deep learning because of its ability to combine various machine learning and deep learning models and algorithms into a single interface. TensorFlow also offers a high-level API, which means that complex coding isn't required to prepare a neural network.

## 00_tensorflow_fundamentals (What we're going to cover broadly):
- TensorFlow basics and fundamentals
- Preprocessing data (getting it into tensors)
- Building and using pretrained deep learning models
- Fitting a model to the data (learning patterns)
- Making predictions with a model (using patterns)
- Evaluating model predictions
- Saving and loading models
- Using a trained model to make predictions on our custom data

## 01_neural_network_regression_in_tensorflow (What we're going to cover broadly): 
- Architecture of a neural network regression model
- Input shapes and output shapes of a regression model (features and labels)
- Creating custom data to view and fit
- Steps in modelling :
  Creating a model, compiling it, fitting it and evaluating the model
- Different evaluation methods
- Saving and loading models

## 02_neural_network_classification_in_tensorflow (What we're going to cover broadly):
- Architecture of a neural network classification model
- Input shapes and output shapes of a classification model (features and labels)
- Creating custom data to view and fit
- Steps in modelling :
  Creating a model, compiling it, fitting it and evaluating the model
- Different classification evaluation methods
- Saving and loading models

## 03_introduction_to_computer_vision_with_tensorflow (What we're going to cover broadly):
- Getting a dataset to work with
- Architecture of a convolutional neural network (CNN) with TensorFlow
- An end-to-end binary image classification problem
- Steps in modelling with CNNs :
  Creating a CNN, compiling it, fitting it and evaluating the model
- An end-to-end multi-class image classification problem
- Making predictions on our own custom images

## 04_transfer_learning_in_tensorflow_part_1_feature_extraction (What we're going to cover broadly):
- Introducing transfer learning with TensorFlow
- Using a small dataset to experiment faster (10% of training samples)
- Building a transfer learning feature extraction model using TensorFlow Hub
- Using TensorBoard to track modelling experiments and results

## 05_transfer_learning_in_tensorflow_part_2_fine_tuning (What we're going to cover broadly):
- Introducing fine-tuning transfer learning with TensorFlow
- Introducing the Keras Functional API to build models
- Using a smaller dataset to experiment faster
- Data augmentation (making our training set more diverse by adding not other samples , yet modifying present data)
- Running a series of experiments on our food vision data
- Introducing the ModelCheckpoint callback to save intermediate training results

## 06_transfer_learning_in_tensorflow_part_3_scaling_up (What we're going to cover broadly):
- Downloading and preparing 10% of all Food101 classes (7500+ training images)
- Training a transfer learning feature extraction model
- Fine-tuning our feature extraction model to beat the original Food101 paper with only 10% of the data
- Evaluating Food Vision mini's predictions
- Finding most wrong predictions (on the test dataset)
- Making predictions with Food Vision mini on our own custom images

## 07_introduction_to_NLP_in_tensorflow (What we're going to cover broadly):
- Downloading and preparing a text dataset
- How to prepare a text data for modelling (tokenization and embedding)
- Setting up multiple modelling experiments with recurrent neural networks (RNNs)
- Building a text feature extraction model using TensorFlow Hub
- Finding the most wrong prediction examples
- Using a model we've built to make predictions on text from the wild
