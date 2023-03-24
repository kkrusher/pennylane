import pennylane as qml
from pennylane import numpy as np
import torch
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import cv2

from .utils.display import grid_display, heatmap_display
from .utils.image import apply_grey_patch
from .utils.saver import save_rgb, save_grayscale
from .explainable_methods import ExplainableMethod

class IntegratedGradients(ExplainableMethod):
    """
    Perform Integrated Gradients algorithm for a given input

    Paper: [Axiomatic Attribution for Deep Networks](https://arxiv.org/pdf/1703.01365.pdf)
    """

    def __init__(self,):
        self.name = 'IntegratedGradients'

    def explain(self, qnode, params, image, class_observable, n_steps=10, colormap=cv2.COLORMAP_VIRIDIS):
        """
        Compute Integrated Gradients for a specific class index

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_observable (int): Index of targeted class
            n_steps (int): Number of steps in the path

        Returns:
            np.ndarray: Grid of all the integrated gradients
        """
        # images, _ = validation_data

        interpolated_images = IntegratedGradients.generate_interpolations(np.array(image), n_steps)

        integrated_gradients = IntegratedGradients.get_integrated_gradients(qnode, params, interpolated_images, class_observable, n_steps)

        heatmaps = heatmap_display(integrated_gradients, image, colormap, image_weight=0)
        # grid = grid_display(heatmaps)
        # return grid
        return heatmaps

    @staticmethod
    def get_integrated_gradients(qnode, params, interpolated_images, class_observable, n_steps):
        """
        Perform backpropagation to compute integrated gradients.

        Args:
            interpolated_images (numpy.ndarray): 4D-Tensor of shape (n_steps, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            n_steps (int): Number of steps in the path

        Returns:
            tf.Tensor: 3D-Tensor of shape (H, W, 3) with integrated gradients
        """

        params = np.array(params, requires_grad=False)
        cost_fn = lambda image: qnode(image, params=params, obs=class_observable)
        grad_fn = qml.grad(cost_fn)

        gradient_ls = []
        for interpolated_image in interpolated_images:
            gradient = grad_fn(interpolated_image)
            gradient_ls.append(gradient)

        integrated_gradients = np.mean(gradient_ls, axis=0)
        return integrated_gradients

    @staticmethod
    def generate_interpolations(image, n_steps):
        """
        Generate interpolation paths for batch of images.

        Args:
            image (numpy.ndarray): 3D-Tensor of images with shape (H, W, 3)
            n_steps (int): Number of steps in the path

        Returns:
            numpy.ndarray: Interpolation paths for each image with shape (n_steps, H, W, 3)
        """
        baseline = np.zeros(image.shape)

        return IntegratedGradients.generate_linear_path(baseline, image, n_steps)

    @staticmethod
    def generate_linear_path(baseline, target, n_steps):
        """
        Generate the interpolation path between the baseline image and the target image.
        There is no corresponding quantum state to all zero amplitude, so the baseline should not be included.
        TODO should also consider the influence of normalization of path points.

        Args:
            baseline (numpy.ndarray): Reference image
            target (numpy.ndarray): Target image
            n_steps (int): Number of steps in the path

        Returns:
            List(np.ndarray): List of images for each step
        """
        return [np.array(baseline + (target - baseline) * index / n_steps, requires_grad=True) for index in range(1, 1+n_steps)]
