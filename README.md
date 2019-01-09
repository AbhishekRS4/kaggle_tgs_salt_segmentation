# Salt Identification Kaggle Challenge

## Notes
* The link to the competition - [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)
* This implementation uses ideas from multiple papers 
    - [DenseNet-121 Encoder](https://arxiv.org/pdf/1608.06993.pdf)
    - [Squeeze and excitation block](https://arxiv.org/pdf/1709.01507)
    - [UNet Decoder](https://arxiv.org/pdf/1505.04597)
* Lovasz softmax loss function is used to train the network
* Code for lovasz loss function is cloned from [lovasz softmax loss function](https://github.com/bermanmaxim/LovaszSoftmax)
* This implementation finished in the **top 53%** of the leaderboard

## Improvement might be achieved by using
* Pyramid pooling module
* Bilinear upsampling instead of transposed convolution
* Momentum optimizer instead of adam

## Pretrained weights
* [Densenet-121](https://github.com/fchollet/deep-learning-models/releases/)
