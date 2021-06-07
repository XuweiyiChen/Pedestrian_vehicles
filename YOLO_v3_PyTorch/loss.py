import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    This is the implementation of Yolo Loss Function.
    We may want to use BinaryCrossEntropy, but we utilized CrossEntropy here.
    """

    def __init__(self):
        """
        Initialize the YoloLoss object.

        Constants signifying how much to pay for each respective part of the loss
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        """
        This function has four different parts: NO OBJECT LOSS,
        OBJECT LOSS, BOX COORDINATES and CLASS LOSS.

        NO OBJECT LOSS calculates binary CrossEntropy loss applying
        sigmoid function

        OBJECT LOSS calculates Intersection of Union

        BOX COORDINATES calculates mean squared error loss between
        the targets and predictions

        CLASS LOSS calculates CrossEntropy between labels

        We will also print out CLASS LOSS.

        Args:
            predictions: array
            target: array
            anchors: array

        Returns:
            array of losses
        """
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),)

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        print("__________________________________")
        print(self.lambda_box * box_loss)
        print(self.lambda_obj * object_loss)
        print(self.lambda_noobj * no_object_loss)
        print(self.lambda_class * class_loss)
        print("\n")

        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
        )
