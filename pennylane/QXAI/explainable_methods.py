import pennylane as qml
from pennylane import numpy as np
import torch
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple

from .utils.display import grid_display, heatmap_display
from .utils.image import apply_grey_patch
from .utils.saver import save_rgb, save_grayscale


class ExplainableMethod:

    def explain(self, ):
        pass

    def save(self, grid: np.ndarray, output_dir: str, output_name: str):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)

    def save_compare(self, input: np.ndarray, label: int, grid: np.ndarray, test_index: int, output_dir='.', show_plot=False):
        """
        Save the original input and heatmap in two places to a specific dir.

        Args:
            input (_type_): (heighth, width)
            sensitivity_maps (_type_): _description_
            grid (_type_): _description_
            output_dir (_type_): _description_
            output_name (_type_): _description_
        """

        # Plot the first six images
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].imshow(input, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title(f'Test_data_{test_index}:{label}')

        axs[1].imshow(grid, cmap='viridis')
        # axs[1].imshow(grid)
        axs[1].axis('off')
        axs[1].set_title('heatmap')

        # axs[2].imshow(grid, cmap='gray')
        # axs[2].axis('off')
        # axs[2].set_title(f'input+sensitivity_map')

        plt.savefig(os.path.join(output_dir, f"{self.name}_{test_index}.png"))
        if show_plot:
            plt.show()
        plt.close()
