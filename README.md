# AudioDenoising

To train the model the program uses U-Net Architecture.
The U-Net is a convolutional neural network architecture originally designed for biomedical image segmentation. It has since been adapted for various applications, including audio denoising. The U-Net's architecture is characterized by an encoder-decoder structure with skip connections between the corresponding layers of the encoder and decoder. This design enables the network to learn both low-level and high-level features, which allows it to accurately capture the structure and patterns in the input data.

# Training_model.py
The first code snippet defines a training and validation loop for a deep learning model (UNetGenerator) used for audio denoising. It first sets up the optimizer, loss function, and device for the model. Then, it trains the model on the training data and validates it on the validation data for a specified number of epochs. Finally, it saves the trained model to a file.

# Denoising.py
The second code snippet is a script that loads a noisy audio file, preprocesses it, applies the trained UNetGenerator model to denoise the audio, and saves the denoised audio to a file. 

# Data Used
Valentini-Botinhao, Cassia. (2017). Noisy speech database for training speech enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2117.
