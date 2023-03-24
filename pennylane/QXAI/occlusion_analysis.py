import pennylane as qml
from pennylane import numpy as np
import torch
from sklearn.preprocessing import normalize
import math
import matplotlib.pyplot as plt

# import numpy as np
import cv2

from .utils.display import grid_display, heatmap_display
from .utils.image import apply_grey_patch
from .utils.saver import save_rgb

from .explainable_methods import ExplainableMethod


class OcclusionAnalysis(ExplainableMethod):
    """
    Perform Occlusion Sensitivity for a given input
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.name = 'OcclusionAnalysis'

    def explain(
        self,
        qnode,
        params,
        image,
        class_observable,
        patch_size=1,
        colormap=cv2.COLORMAP_VIRIDIS,
    ):
        """
        Compute Occlusion Sensitivity maps for a specific class index.

        Args:
            qnode (__callable__): a function with title qnode(inputs, params, obs=None)
            params (_type_): _description_
            image (_type_): _description_
            class_observable (qml.Hermition): Observable of targeted class
            patch_size (int): Size of patch to apply on the image. Defaults to 1.
            colormap (int): OpenCV Colormap to use for heatmap visualization. Defaults to cv2.COLORMAP_VIRIDIS.

        Returns:
            np.ndarray: Grid of all the sensitivity maps with shape (batch_size, H, W, 3)
        """

        # images, _ = validation_data
        sensitivity_map = self.get_sensitivity_map(qnode, params, image, class_observable, patch_size)

        heatmaps = heatmap_display(sensitivity_map, image, colormap, image_weight=0)

        # grid = grid_display(heatmaps)
        # return grid

        return heatmaps

    def get_sensitivity_map(self, qnode, params, image, class_observable, patch_size=1):
        """
        Compute sensitivity map on a given image for a specific class index.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            image: a (hight, width) shape adarray
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image

        Returns:
            np.ndarray: Sensitivity map with shape (H, W)
        """
        sensitivity_map = np.zeros((
            math.ceil(image.shape[0] / patch_size),
            math.ceil(image.shape[1] / patch_size),
        ))

        patched_images = [
            apply_grey_patch(image, top_left_x, top_left_y, patch_size) for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size))
            for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size))
        ]

        coordinates = [
            (index_y, index_x) for index_x in range(sensitivity_map.shape[1]  # pylint: disable=unsubscriptable-object
                                                    ) for index_y in range(sensitivity_map.shape[0]  # pylint: disable=unsubscriptable-object
                                                                           )
        ]

        target_class_predictions = [qnode(patched_image, params, class_observable) for patched_image in patched_images]

        for (index_y, index_x), confidence in zip(coordinates, target_class_predictions):
            sensitivity_map[index_y, index_x] = 1 - confidence

        return cv2.resize(sensitivity_map, image.shape[0:2])
