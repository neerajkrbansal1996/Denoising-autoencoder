# Denoising autoencoder Using Keras
A denoising autoencoder is a feed forward neural network that learns to denoise images. By doing so the neural network learns interesting features on the images used to train it. Then it can be used to extract features from similar images to the training set.

![Denoising autoencoder](https://cdn-images-1.medium.com/max/1800/1*G0V4dz4RKTKGpebeoSWB0A.png)

# Dependencies
1. Keras
2. numpy
3. matplotlib

# Usage
1. First Download this repository onto your local system.
2. (Optional) if you want to train your neural network onto yourself run <b>python auto_encoder.py</b> after opening the project Directory and make sure you delete autoencoder.h5(In Trainded_Model Folder) before training your neural network.
3. If you want to check what output does your neural network produce run <b>python test_model.py</b>

<b>Note:</b> The Pretrained model is trained on only 20 epochs and if you want to see your neural network work better it is wise to train on 100 epochs.
