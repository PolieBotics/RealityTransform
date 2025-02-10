#!/usr/bin/env python3
"""
Pix2PixHD inference module, renamed to avoid circular imports.

This module defines a Pix2PixHDInference class that:
- Initializes a pix2pixHD model using the original repository's create_model().
- Provides a run_inference(pil_image) method to run single-image forward passes.
"""

import os
import numpy as np
import torch
from PIL import Image

# Import from your existing pix2pixHD code:
# Adjust these imports if your "models" or "util" are in a different path
from models.models import create_model
import util.util as util

class Pix2PixHDInference:
    def __init__(
        self,
        # Basic or commonly used defaults
        model="pix2pixHD",
        name="reality_transform_rec2emis",
        checkpoints_dir="./checkpoints",
        which_epoch="latest",
        gpu_ids="0",

        # Some typical pix2pixHD network settings
        netG="local",
        ngf=32,
        n_downsample_global=4,
        n_blocks_global=9,
        n_blocks_local=3,
        n_local_enhancers=1,
        norm="instance",
        label_nc=0,
        no_instance=True,
        fp16=False,
        use_dropout=False,

        # Additional features
        instance_feat=False,
        label_feat=False,
        feat_num=3,
        load_features=False,
        use_encoded_image=False,
        n_downsample_E=4,
        nef=16,
        n_clusters=10,

        # Input/Output channels
        input_nc=3,
        output_nc=3,

        # Data settings
        resize_or_crop='none',
        no_flip=True,
        continue_train=False,
        verbose=False,
        isTrain=False,

        # Possibly needed by create_model():
        phase='test',
        batchSize=1,
        serial_batches=True,
    ):
        """
        Initialize the pix2pixHD model with all relevant flags (defaults suitable for inference).
        """

        # We'll mimic argparse Namespace by creating a small class:
        class TempOptions:
            pass

        opt = TempOptions()

        # Basic required fields
        opt.model = model
        opt.name = name
        opt.checkpoints_dir = checkpoints_dir
        opt.which_epoch = which_epoch
        opt.gpu_ids = gpu_ids

        opt.netG = netG
        opt.ngf = ngf
        opt.n_downsample_global = n_downsample_global
        opt.n_blocks_global = n_blocks_global
        opt.n_blocks_local = n_blocks_local
        opt.n_local_enhancers = n_local_enhancers
        opt.norm = norm
        opt.label_nc = label_nc
        opt.no_instance = no_instance
        opt.fp16 = fp16
        opt.use_dropout = use_dropout

        opt.instance_feat = instance_feat
        opt.label_feat = label_feat
        opt.feat_num = feat_num
        opt.load_features = load_features
        opt.use_encoded_image = use_encoded_image
        opt.n_downsample_E = n_downsample_E
        opt.nef = nef
        opt.n_clusters = n_clusters

        opt.input_nc = input_nc
        opt.output_nc = output_nc

        opt.resize_or_crop = resize_or_crop
        opt.no_flip = no_flip
        opt.continue_train = continue_train
        opt.verbose = verbose
        opt.isTrain = isTrain

        # Additional options often used by pix2pixHD:
        opt.phase = phase
        opt.batchSize = batchSize
        opt.serial_batches = serial_batches

        # Convert gpu_ids from string to list of ints
        str_ids = opt.gpu_ids.split(',')
        gpu_ids_parsed = []
        for s in str_ids:
            s = s.strip()
            if s:
                gid = int(s)
                if gid >= 0:
                    gpu_ids_parsed.append(gid)
        opt.gpu_ids = gpu_ids_parsed

        # Create & initialize the model
        self.model = create_model(opt)
        self.model.initialize(opt)
        self.model.eval()

        # Move model to GPU if available
        if len(opt.gpu_ids) > 0:
            self.model.cuda(opt.gpu_ids[0])

        self.gpu_ids = opt.gpu_ids
        self.opt = opt

    def run_inference(self, pil_image):
        """
        Run a single forward pass through the pix2pixHD model.
        Input:  PIL image (RGB)
        Output: PIL image (RGB) in [0..255], or None if unexpected output shape.
        """
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert [0..255] -> [-1..1]
        arr = np.array(pil_image, dtype=np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)

        tensor_in = torch.from_numpy(arr).unsqueeze(0)  # shape (1,C,H,W)

        if len(self.gpu_ids) > 0:
            tensor_in = tensor_in.cuda(self.gpu_ids[0])

        with torch.no_grad():
            fake = self.model.inference(tensor_in, None)

        # Some pix2pixHD nets return a list of outputs
        if isinstance(fake, list) and len(fake) > 0:
            fake = fake[0]

        # Possibly remove batch dimension
        if fake.dim() == 4 and fake.size(0) == 1:
            fake = fake.squeeze(0)  # shape => (3,H,W)

        # Convert back to a PIL
        if fake.dim() == 3 and fake.size(0) == 3:
            out_np = util.tensor2im(fake)  # (H,W,3) in uint8
            return Image.fromarray(out_np)
        else:
            print("[WARN] Unexpected pix2pixHD output shape:", fake.size())
            return None
