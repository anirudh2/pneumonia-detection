"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(1, self.num_channels, 26, stride=4, padding=1)
        #
        # 1 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(self.num_channels*25*25, 1)

        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        # we apply the convolution layers
        out1 = self.conv1(s)
#         pdb.set_trace()
        batch_size = out1.size()[0]
        out2 = F.relu(F.max_pool2d(out1, 2))

        # flatten the output for each image
#         pdb.set_trace()
        out3 = out2.view(batch_size, self.num_channels*25*25)

        out4 = self.fc1(out3)
#         pdb.set_trace()
        out5 = F.relu(out4)
        out6 = F.dropout(out5)

        # apply 1 fully connected layer with dropout
#         s = F.dropout(F.relu())
#         pdb.set_trace()
        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return  out6.view(batch_size) #.log_softmax(out6, dim=1)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 2 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
#     # Manually calculating loss for comparision
#     tmpFn = nn.Sigmoid()
#     loss_manual = np.mean(np.log(1-tmpFn(outputs).data.numpy()))
#     pdb.set_trace()
#     num_examples = outputs.size()[0]
#     return -torch.sum(outputs[range(num_examples), labels])/num_examples

    # pdb.set_trace()
    weights = torch.ones(labels.size())
    weights[labels==1] = 100
    weights[labels==0] = 0
    loss = nn.BCEWithLogitsLoss(weight=weights)


    return loss(outputs, labels.float())


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    # pdb.set_trace()
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
