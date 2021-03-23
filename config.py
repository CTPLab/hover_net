import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"
        model_mode = "fast"  # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception(
                "Must use either `original` or `fast` as model mode")

        nr_type = 6  # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information -
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        # patch shape used as input to network - central crop performed after augmentation
        #act_shape = [256, 256]
        #out_shape = [164, 164]  # patch shape at output of network
        if model_mode == "original":
            act_shape = [270, 270]
            out_shape = [80, 80]
            #if act_shape != [270, 270] or out_shape != [80, 80]:
            #    raise Exception(
            #        "If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            act_shape = [256, 256]
            out_shape = [164, 164]
            #if act_shape != [256, 256] or out_shape != [164, 164]:
            #    raise Exception(
            #        "If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")
        
        # patch shape used during augmentation (larger patch may have less border artefacts)
        aug_shape = [540, 540]
        
        self.dataset_name = "consep"  # extracts dataset info from dataset.py
        self.log_dir = "logs/"  # where checkpoints will be saved

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape, },
            "valid": {"input_shape": act_shape, "mask_shape": out_shape, },
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.{}.opt".format(model_name)
        )
        self.model_config = module.get_config(nr_type, model_mode)
