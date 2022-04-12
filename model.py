"""
Contains implementations of all the models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM implementation by Andrea Palazzi and Davide Abati
    https://github.com/ndrplz/ConvLSTM_pytorch

    ConvLSTM is a type of recurrent neural network
    for spatio-temporal prediction that has convolutional structures
    in both the input-to-state and state-to-state transitions.

    This class implements a single cell of the ConvLSTM model.

    Parameters
    ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.

    Example
    -------
    convlstmcell = ConvLSTMCell(input_size=(height,width),
                                          input_dim,
                                          hidden_dim,
                                          kernel_size,
                                          bias)

    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        """
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Returns the output of the ConvLSTM cell.
        """
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state and cell state for the ConvLSTM cell.
        """
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                self.height,
                self.width).to(config.device),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                self.height,
                self.width).to(config.device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM implementation by Andrea Palazzi and Davide Abati
    https://github.com/ndrplz/ConvLSTM_pytorch

    ConvLSTM is a type of recurrent neural network
    for spatio-temporal prediction that has convolutional structures
    in both the input-to-state and state-to-state transitions.

    This class implements the ConvLSTM model.

    Parameters
    ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        num_layers: int
            Number of layers in the model.
        batch_first: bool, optional
            Whether or not the batches are processed with the batch dim first.
        bias: bool, optional
            Whether or not to add the bias.
        return_all_layers: bool, optional
            Whether or not to return all the outputs and hidden layers

    Example
    -------
    # (t, b, c, h, w) or (b, t, c, h, w)
    inp_tensor = torch.rand([10, 5, 3, 28, 28])

    convlstm = ConvLSTM(
                              input_size=(4,8),
                              input_dim=512,
                              hidden_dim=[512, 512],
                              kernel_size=(3,3),
                              num_layers=2,
                              batch_first=False,
                              bias=True,
                              return_all_layers=False
                            )
    output = convlstm(inp_tensor)
    """

    def __init__(
            self,
            input_size,
            input_dim,
            hidden_dim,
            kernel_size,
            num_layers,
            batch_first=False,
            bias=True,
            return_all_layers=False):
        """
        Initialize ConvLSTM model.
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having
        # len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            if i == 0:
                cur_input_dim = self.input_dim
            else:
                cur_input_dim = self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape
            (time_steps, batch_size, inp_channels, height, width)
            or (batch_size, time_steps, inp_channels, height, width)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c])

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden layers for all cells in the model.
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Checks whether the kernel_size is in the correct format.
        """
        if not (isinstance(kernel_size, tuple)
           or (isinstance(kernel_size, list)
           and all(
                [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Make sure that both `kernel_size` and `hidden_dim`
        are lists having len == num_layers
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class VGG16_SegNet(nn.Module):
    """
    References used when implementing SegNet:
    https://github.com/vinceecws/SegNet_PyTorch/blob/master/SegNet.py
    https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/model.py
    https://github.com/say4n/pytorch-segnet/blob/master/src/model.py

    This class implements the SegNet model with the encoder as the same
    topology as VGG16. This implementation does not use pretrained weights
    for the encoder.

    Example
    -------
    model = VGG16_SegNet()
    # (batch_size, inp_channels, width, height)
    input = torch.rand([100, 3, 128, 256])
    output = model(input)
    """

    def __init__(self):
        """
        Initializes the VGG11_SegNet model.
        """
        super(VGG16_SegNet, self).__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=(3, 3), padding="same"),
        )

    def forward(self, x):
        """
        Returns the output of the model.
        Parameters
        ----------
        input : torch.Tensor((batch_size, inp_channels, width, height))
            Input image.

        Returns
        -------
        output: torch.Tensor((batch_size, 2, width, height))
            Segmented binary image.

        """
        x, idx1 = self.maxpool(self.encoder1(x))
        x, idx2 = self.maxpool(self.encoder2(x))
        x, idx3 = self.maxpool(self.encoder3(x))
        x, idx4 = self.maxpool(self.encoder4(x))
        x, idx5 = self.maxpool(self.encoder5(x))

        x = self.unpool(x, idx5)
        x = self.unpool(self.decoder5(x), idx4)
        x = self.unpool(self.decoder4(x), idx3)
        x = self.unpool(self.decoder3(x), idx2)
        x = self.unpool(self.decoder2(x), idx1)
        x = self.decoder1(x)

        return F.log_softmax(x, dim=1)


class VGG16_SegNet_pretrained(nn.Module):
    """
    This class implements the SegNet model with the encoder as the
    same topology as VGG16. This implementation uses pretrained weights
    for the encoder.

    Example
    -------
    model = VGG16_SegNet()
    # (batch_size, inp_channels, width, height)
    input = torch.rand([100, 3, 128, 256])
    output = model(input)
    """

    def __init__(self):
        """
        Initializes the VGG11_SegNet model.
        """
        super(VGG16_SegNet_pretrained, self).__init__()
        self.vgg16_bn = models.vgg16_bn(pretrained=True).features
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.encoder1 = nn.Sequential(
            self.vgg16_bn[0],
            self.vgg16_bn[1],
            self.vgg16_bn[2],
            self.vgg16_bn[3],
            self.vgg16_bn[4],
            self.vgg16_bn[5]
        )
        self.encoder2 = nn.Sequential(
            self.vgg16_bn[7],
            self.vgg16_bn[8],
            self.vgg16_bn[9],
            self.vgg16_bn[10],
            self.vgg16_bn[11],
            self.vgg16_bn[12]
        )
        self.encoder3 = nn.Sequential(
            self.vgg16_bn[14],
            self.vgg16_bn[15],
            self.vgg16_bn[16],
            self.vgg16_bn[17],
            self.vgg16_bn[18],
            self.vgg16_bn[19],
            self.vgg16_bn[20],
            self.vgg16_bn[21],
            self.vgg16_bn[22]
        )
        self.encoder4 = nn.Sequential(
            self.vgg16_bn[24],
            self.vgg16_bn[25],
            self.vgg16_bn[26],
            self.vgg16_bn[27],
            self.vgg16_bn[28],
            self.vgg16_bn[29],
            self.vgg16_bn[30],
            self.vgg16_bn[31],
            self.vgg16_bn[32]
        )
        self.encoder5 = nn.Sequential(
            self.vgg16_bn[34],
            self.vgg16_bn[35],
            self.vgg16_bn[36],
            self.vgg16_bn[37],
            self.vgg16_bn[38],
            self.vgg16_bn[39],
            self.vgg16_bn[40],
            self.vgg16_bn[41],
            self.vgg16_bn[42]
        )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=(3, 3), padding="same"),
        )

    def forward(self, x):
        """
        Returns the output of the model.
        Parameters
        ----------
        input : torch.Tensor((batch_size, inp_channels, width, height))
            Input image.

        Returns
        -------
        output: torch.Tensor((batch_size, 2, width, height))
            Segmented binary image.

        """
        x, idx1 = self.maxpool(self.encoder1(x))
        x, idx2 = self.maxpool(self.encoder2(x))
        x, idx3 = self.maxpool(self.encoder3(x))
        x, idx4 = self.maxpool(self.encoder4(x))
        x, idx5 = self.maxpool(self.encoder5(x))

        x = self.unpool(x, idx5)
        x = self.unpool(self.decoder5(x), idx4)
        x = self.unpool(self.decoder4(x), idx3)
        x = self.unpool(self.decoder3(x), idx2)
        x = self.unpool(self.decoder2(x), idx1)
        x = self.decoder1(x)

        return F.log_softmax(x, dim=1)


class VGG11_SegNet_pretrained(nn.Module):
    """
    This class implements a semantic segmentation model similar to
    SegNet but instead of VGG16, it used VGG11 topology as the encoder. This
    model uses pretrained weights for the encoder.

    Example
    -------
    model = VGG11_SegNet_pretrained()
    # (batch_size, inp_channels, width, height)
    input = torch.rand([100, 3, 128, 256])
    output = model(input)
    """

    def __init__(self):
        """
        Initializes the VGG11_SegNet_pretrained model.
        """
        super(VGG11_SegNet_pretrained, self).__init__()
        self.vgg11_bn = models.vgg11_bn(pretrained=True).features
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.encoder1 = nn.Sequential(
            self.vgg11_bn[0],
            self.vgg11_bn[1],
            self.vgg11_bn[2]
        )
        self.encoder2 = nn.Sequential(
            self.vgg11_bn[4],
            self.vgg11_bn[5],
            self.vgg11_bn[6]
        )
        self.encoder3 = nn.Sequential(
            self.vgg11_bn[8],
            self.vgg11_bn[9],
            self.vgg11_bn[10],
            self.vgg11_bn[11],
            self.vgg11_bn[12],
            self.vgg11_bn[13]
        )
        self.encoder4 = nn.Sequential(
            self.vgg11_bn[15],
            self.vgg11_bn[16],
            self.vgg11_bn[17],
            self.vgg11_bn[18],
            self.vgg11_bn[19],
            self.vgg11_bn[20]
        )
        self.encoder5 = nn.Sequential(
            self.vgg11_bn[22],
            self.vgg11_bn[23],
            self.vgg11_bn[24],
            self.vgg11_bn[25],
            self.vgg11_bn[26],
            self.vgg11_bn[27]
        )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        """
        Returns the output of the model.
        Parameters
        ----------
        input : torch.Tensor((batch_size, inp_channels, width, height))
            Input image.

        Returns
        -------
        output: torch.Tensor((batch_size, 2, width, height))
            Segmented binary image.

        """
        x, idx1 = self.maxpool(self.encoder1(x))
        x, idx2 = self.maxpool(self.encoder2(x))
        x, idx3 = self.maxpool(self.encoder3(x))
        x, idx4 = self.maxpool(self.encoder4(x))
        x, idx5 = self.maxpool(self.encoder5(x))

        x = self.unpool(x, idx5)
        x = self.unpool(self.decoder5(x), idx4)
        x = self.unpool(self.decoder4(x), idx3)
        x = self.unpool(self.decoder3(x), idx2)
        x = self.unpool(self.decoder2(x), idx1)
        x = self.decoder1(x)

        return F.log_softmax(x, dim=1)


class VGG11_SegNet_ConvLSTM_pretrained(nn.Module):
    """
    This class implements a semantic segmentation model similar to
    VGG11_SegNet but has a Convolutional LSTM layer between the encoder and
    the decoder.

    Example
    -------
    model = VGG11_SegNet_ConvLSTM_pretrained()
    # (batch_size, time_steps, inp_channels, width, height)
    input = torch.rand([100, 5, 3, 128, 256])
    output = model(input)
    """

    def __init__(self):
        """
        Initializes the VGG11_SegNet_ConvLSTM_pretrained model.
        """
        super(VGG11_SegNet_ConvLSTM_pretrained, self).__init__()
        self.vgg11_bn = models.vgg11_bn(pretrained=True).features
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.encoder1 = nn.Sequential(
            self.vgg11_bn[0],
            self.vgg11_bn[1],
            self.vgg11_bn[2]
        )
        self.encoder2 = nn.Sequential(
            self.vgg11_bn[4],
            self.vgg11_bn[5],
            self.vgg11_bn[6],
        )
        self.encoder3 = nn.Sequential(
            self.vgg11_bn[8],
            self.vgg11_bn[9],
            self.vgg11_bn[10],
            self.vgg11_bn[11],
            self.vgg11_bn[12],
            self.vgg11_bn[13],
        )
        self.encoder4 = nn.Sequential(
            self.vgg11_bn[15],
            self.vgg11_bn[16],
            self.vgg11_bn[17],
            self.vgg11_bn[18],
            self.vgg11_bn[19],
            self.vgg11_bn[20],
        )
        self.encoder5 = nn.Sequential(
            self.vgg11_bn[22],
            self.vgg11_bn[23],
            self.vgg11_bn[24],
            self.vgg11_bn[25],
            self.vgg11_bn[26],
            self.vgg11_bn[27],
        )
        self.convlstm = ConvLSTM(
            input_size=(4, 8),
            input_dim=512,
            hidden_dim=[512, 512],
            kernel_size=(3, 3),
            num_layers=2,
            batch_first=False,
            bias=True,
            return_all_layers=False
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        """
        Returns the output of the model.
        Parameters
        ----------
        input : torch.Tensor((batch_size,
                              time_steps,
                              inp_channels,
                              width,
                              height))
            Input image.

        Returns
        -------
        output: torch.Tensor((batch_size, 2, width, height))
            Segmented binary image.

        """
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            item, idx1 = self.maxpool(self.encoder1(item))
            item, idx2 = self.maxpool(self.encoder2(item))
            item, idx3 = self.maxpool(self.encoder3(item))
            item, idx4 = self.maxpool(self.encoder4(item))
            item, idx5 = self.maxpool(self.encoder5(item))
            data.append(item.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        x = lstm[0][-1, :, :, :, :]
        x = self.unpool(x, idx5)
        x = self.unpool(self.decoder5(x), idx4)
        x = self.unpool(self.decoder4(x), idx3)
        x = self.unpool(self.decoder3(x), idx2)
        x = self.unpool(self.decoder2(x), idx1)
        x = self.decoder1(x)
        return F.log_softmax(x, dim=1)
