import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from torchvision import transforms, models

from PIL import Image

from opensoundscape.helpers import binarize


def predict(
    model, img_paths, img_shape, batch_size=1, num_workers=12, apply_softmax=True
):
    """ get multi-class model predictions from a pytorch model for a set of images
    
    model: a pytorch model object (not path to weights)
    img_paths: a list of paths to RGB png spectrograms
    
    returns: df of predictions indexed by file (columns=class names? #TODO)"""

    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, df, height=img_shape[0], width=img_shape[1]):
            self.data = df

            self.height = height
            self.width = width

            self.mean = torch.tensor([0.8013 for _ in range(3)])
            self.std_dev = [0.1576 for _ in range(3)]

            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(self.mean, self.std_dev)]
            )

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            row = self.data.iloc[idx, :]
            image = Image.open(row["filename"])
            image = image.convert("RGB")
            image = image.resize((self.width, self.height))

            return self.transform(image)  # , torch.from_numpy(labels)

    # turn list of files into a df (this maintains parallelism with training format)
    file_df = pd.DataFrame(columns=["filename"], data=img_paths)

    # create pytorch dataset and dataloader objects
    dataset = PredictionDataset(file_df)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # run prediction
    all_predictions = []
    for i, inputs in enumerate(dataloader):
        predictions = model(inputs)
        if apply_softmax:
            softmax_val = softmax(predictions, 1).detach().cpu().numpy()[0]
            all_predictions.append(softmax_val)
        else:
            all_predictions.append(
                list(predictions.detach().numpy()[0])
            )  # .astype('float64')

    # how do we get class names? are they saved in the file?
    # the column names should be class names
    return pd.DataFrame(index=img_paths, data=all_predictions)


##### a set of functions for gradcam event detection (maybe these become a separate module) ####


def activation_region_limits(gcam, threshold=0.2):
    """get min and max column, row indices of a gradcam map that exceed threshold"""
    img_shape = np.shape(gcam)
    arr = np.array(gcam)
    binary_activation = np.array([binarize(row, threshold) for row in arr])
    non_zero_rows, non_zero_cols = np.nonzero(binary_activation)

    # handle corner case: no rows / cols pass threshold
    row_lims = (
        [min(non_zero_rows), max(non_zero_rows)]
        if len(non_zero_rows) > 0
        else [0, img_shape[0]]
    )
    col_lims = (
        [min(non_zero_cols), max(non_zero_cols)]
        if len(non_zero_cols) > 0
        else [0, img_shape[1]]
    )
    box_lims = np.array([row_lims, col_lims])

    return box_lims


def in_box(x, y, box_lims):
    """check if an x, y position falls within a set of limits [[xl,xh], [yl,yh]]"""
    if x > box_lims[0, 0] and y > box_lims[1, 0]:
        if x < box_lims[0, 1] and y < box_lims[1, 1]:
            return True
    return False


def activation_region_to_box(activation_region, threshold=0.2):
    """draw a rectangle of the activation box as a boolean array
    (useful for plotting a mask over a spectrogram)"""
    img_shape = np.shape(activation_region)
    box_lims = activation_region_limits(activation_region, threshold)
    box_mask_arr = [
        [1 if in_box(xi, yi, box_lims) else 0 for yi in range(img_shape[0])]
        for xi in range(img_shape[1])
    ]
    return box_mask_arr


def save_gradcam(filename, gcam, raw_image):
    """save spectrogram + gradcam to image file. currently not used."""
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


def gradcam_region(
    model, img_paths, img_shape, predictions=None, save_gcams=True, box_threshold=0.2
):
    """
    Compute the GradCam activation region (the area of an image that was most important for classification in the CNN)
    
    parameters:
        model: a pytorch model object
        img_paths: list of paths to image files
        predictions = None: [list of float] optionally, provide model predictions per file to avoid re-computing
        save_gcams = True: bool, if False only box regions around gcams are saved
    
    returns:
        boxes: limits of the box surrounding the gcam activation region, as indices: [ [min row, max row], [min col, max col] ] 
        gcams: (only returned if save_gcams == True) arrays with gcam activation values, shape = shape of image
    """
    from grad_cam import BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation
    import cv2

    gcams = [None] * len(img_paths)
    boxes = [None] * len(img_paths)

    # establish a transformation function for normalizing images
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.8013 for _ in range(3)], std=[0.1576 for _ in range(3)]
            ),
        ]
    )

    # TODO: parallelize
    for i, img in enumerate(img_paths):
        raw_image = Image.open(img)
        raw_image = raw_image.resize(size=img_shape)
        image = transform(raw_image).unsqueeze(0)

        # generate model predictions, if they weren't provided
        if predictions is None:
            with torch.set_grad_enabled(False):
                logits = model(image)  # .cuda())
                softmax_num = softmax(logits, 1).detach().cpu().numpy()[0]
        else:
            logits = predictions[i]

        # gradcam and guided back propogation
        gcam = GradCAM(model=model)
        gcam.forward(image)  # .cuda());

        gbp = GuidedBackPropagation(model=model)
        probs, idx = gbp.forward(image)  # .cuda())

        gcam.backward(idx=idx[0])
        region = gcam.generate(target_layer="layer4.1.conv2")
        gcam = cv2.resize(region, img_shape)

        # find min/max indices of rows/columns that were activated
        box_bounds = activation_region_limits(gcam, threshold=box_threshold)

        if save_gcams:
            gcams[i] = gcam
        boxes[i] = box_bounds

    if save_gcams:
        return boxes, gcams
    return boxes
