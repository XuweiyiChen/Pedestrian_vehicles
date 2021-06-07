import torch
import torch.nn as nn

"""
Each tuple is a sample of a convolutional block (filter, kernel_size, stride)
tuple: (out_channels, kernel_size, stride)

Each convolutional layer is same convolution
"B" indicates a residual block
"S" indicates a prediction block and carries responsibility of computing loss
"U" is up-sampling for the features detection
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    # explicit model for YOLO_v3
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    """
    The most common building blocks for architecture as separate classes
    to avoid repeating code.
    Each block is consisted of a convolutional block, batch normalization
    and leaky relu.
    """
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        """
        Add lays for convolutional block, batch normalization and leaky relu

        Args:
            in_channels:
            out_channels:
            bn_act:
            **kwargs:
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        """
        the additional forward methods can toggle the bn_act to false and
        skip the batch normalization for the last layer

        Args:
            x: boolean

        Returns:
            Default output batch normalization and activation function
            or output convolutional layer
        """
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    """
    The residual block is two convolutional blocks with a residual
    layer
    The number of channels will be halved in the first convolutional
    layer and then doubled again in the second
    """
    def __init__(self, channels, use_residual=True, num_repeats=1):
        """
        Construct a sequential pair of two convolutional layer with
        repeats

        Args:
            channels:
            use_residual:
            num_repeats:
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        """
        This methods provides options of whether connect two layers
        with residual

        Args:
            x: boolean

        Returns:
            Default by returning a combination of two layers connected by residual
            or return the layer without residual
        """
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    """
    Prediction block is th last two convolutional layers for
    each scale.
    Here is the shape: (batch size, anchors per scale, grid size,
    grid size, 5 + number of classes)
    """
    def __init__(self, in_channels, num_classes):
        """
        As the paper we have three archers for each ceil and
        we are make one predictions for each ceil and 5 is
        the output here.
        Here is the model: [po, x, y, w, h]

        Args:
            in_channels:
            num_classes:
        """
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        """
        reshape is very important step here.
        Making the long list into two dimensions.
        Args:
            x: integer

        Returns:
            two dimensional array
        """
        return (
            self.pred(x)
                .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    """
    We can put it all together based on on the config list
    """

    def __init__(self, in_channels=3, num_classes=80):
        """
        We will initialize YOLOv3 based on channels and
        input number of classes.

        Args:
            in_channels:
            num_classes:
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        """
        One output for each scale

        Args:
            x: Array

        Returns:
             Array for different cases
        """
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        """
        track all layers using pytorch
        Construct Yolo networks based on config

        Returns:
            pytorch object (layer)
        """
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats, ))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2), )
                    # up sampling is the time we want to concatenate
                    in_channels = in_channels * 3

        return layers


def test():
    """
    test whether we have a good model with num_class is 20
    """
    num_classes = 20
    model = YOLOv3(num_classes=num_classes)
    img_size = 416
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes)
    assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes)
    assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes)
    print("Perform correctly")
