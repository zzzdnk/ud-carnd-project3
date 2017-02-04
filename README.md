# Behavioral Cloning Project
This project aims to develop a deep learning model for predecting the steering angle of an autonomous car. The model takes an image of the scene as input and outputs a number representing the predicted steering angle.

## Model Design
The model architecture is based on a convolutional neural network consisting of both convolutional layers as well as densely connected layers. The final model is a result of a series of gradually changing models starting with a simple neural network with one hidden layer  to verify correct functioning of the input-output pipeline, and ending in the current model implementation. Every model was tested in the simulator to observe its performance in the autonomous mode.


## Model Architecture
The final model consists of the input layer, three hidden convolutional layers, two densely connected hidden layers and the final output layer. The input layer uses a lambda function to normalize the input images into a (-1, 1) range. The resolution of input images is 25x160x3 pixels. The first convolutional layer using a 1x1 kernel with depth 3 is meant as a color selection tool. The second convolutional layer uses a 12x24 kernel with depth 16 and is followed by the third convolutional layer with 4x12 kernel size and depth 24. Each convolutional layer is followed by an exponential linear unit (ELU) activation, 30% dropout, and maxpooling operation with stride 2x2. Two densely connected hidden layers use 512 and 256 units with ELU and 50% dropout. The final output layer has one unit corresponding to the predected steering angle.

The model uses the mean squared error loss function, and Adam optimizer with its default settings.


## Dataset Characteristics

The model was trained on the dataset provided by Udacity. Only images from the center camera were used for the training. The resolution of each input image is 160x320x3 pixels. 

<p align="Center">
     <img src=img/sample.png />
</p>

To reduce the size of the input layer and hence the complexity of the model, inpute images were cropped and resized. The upper and lower part of each image was cropped out and the cropped image was further resized by 50% in both axis. Hence the resolution of every image used by the network is 25x160x3.

<p align="Center">
     <img src=img/fig2.png />
</p>

The steering angle of input images falls into the (-1, 1) range. As we can see in the following figure, 

<p align="Center">
     <img src=img/fig3.png />
</p>

the majority of input images correspond to zero value of the steering angle. If we plot count only images with non-zero values of the steering angle we get the following figure.

<p align="Center">
     <img src=img/fig4.png />
</p>

To account for uneven distribution of different steering angles in the dataset, a new dataset was created by randomly dropping 50% of images with zero steering values and by further adding four copies of each image with nonzero steering value. The following figure shows the distribution of values in the final dataset.

<p align="Center">
     <img src=img/fig5.png />
</p>

Hence, the final dataset contains 44783 samples. All samples were used for model training, and a subset of images was selected for validation purposes. The final validation is done by running the simulator in autonomous mode.

