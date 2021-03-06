# About
Image denoising is a classical yet active topic in low level vision since it is an indispensable step in many practical applications of image processing. This project aims to provide a generic image denoising model to solve the image denoising problem. The model is designed as a deep Convolutional Neural Network and is supported by formulations of Residual Learning and Batch Normalization. Different from the existing discriminative denoising models which usually train a specific model for additive white Gaussian noise of a certain level (known σ), the DnCNN model to be implemented in this project would be able to handle Gaussian denoising with unknown noise level (blind Gaussian denoising). Moreover, the formulation of a residual learning approach allows the generalization of the model to solve several other “general” image denoising tasks. 

## Model 1 – DnCNN-S
The DnCNN-S model solves the problem of denoising images with gaussian noise of a known specific noise level. The model is trained with noisy images that are set to a predefined noise value. Naturally, this model is limited to denoising images with noise profiles that match the training images and performs poorly when given an image with an alternate noise profile.

## Model 2 - DnCNN-B
The DnCNN – B model is a more efficient variation of the previous denoising model. This model is not restricted by an image’s noise profile. Instead it is capable of denoising images with any type of gaussian noise profile. Although theoretically the range of noise profile can be infinite, for the purposes of this project, the noise values are limited to the range σ=[0-55].

**Denoising Results**

![not found](https://github.com/iamVarunAnand/Residual-Image-Denoising/blob/master/results/DnCNN-S(Set12)/Denoised4.png) 
![not found](https://github.com/iamVarunAnand/Residual-Image-Denoising/blob/master/results/DnCNN-S(Set12)/Denoised3.png)
![not found](https://github.com/iamVarunAnand/Residual-Image-Denoising/blob/master/results/DnCNN-S(Set12)/Denoised1.png)

## Model 3 - DnCNN-3
This model is the most generic and also the most superior model when compared to the other two. The DnCNN-3 isn’t restricted to the Gaussian Denoising task, but is capable of denoising other noise varieties as well. As mentioned previously, in this project two other denoising applications are explored, namely, Single Image Super Resolution (SISR) and JPEG Image Deblocking. 

**Denoising Results**

![not found](https://github.com/iamVarunAnand/Residual-Image-Denoising/blob/master/results/DnCNN-3/input.png)
![not found](https://github.com/iamVarunAnand/Residual-Image-Denoising/blob/master/results/DnCNN-3/output.png)
