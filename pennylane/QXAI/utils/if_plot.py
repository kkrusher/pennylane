'''
functions for ploting results of influence function
'''
import pennylane as qml
from pennylane import numpy as np
import torch.nn as nn
from sklearn import manifold, datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d


def t_SNE_3d(train_data_inputs, train_data_labels, test_data_inputs, test_data_labels, if_top_n_samples_indices, reletIF_top_n_samples_indices, test_index, result_save_dir='./', show_plot=False):
    # '''t-SNE'''
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)

    X = np.concatenate((train_data_inputs, test_data_inputs), axis=0)
    # X = train_data_inputs
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    # Define the color map and markers for each class
    cmap = plt.cm.get_cmap('Dark2', 6)
    markers = [".", ".", ".", '*', 's', '^', 'D', 'p', 'v', 'h', '+', 'o']

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    y_list = [np.argmax(train_data_labels[i]) for i in range(len(train_data_labels))]

    for category in range(3):
        indexes = [i for i, y in enumerate(y_list) if y == category]
        ax.scatter(X_norm[indexes, 0], X_norm[indexes, 1], X_norm[indexes, 2], color=cmap(category), marker=markers[category], label=f'Class {category+1}')

    color_index = 3
    # plot current test data
    cur_index = test_index + len(train_data_labels)
    # plot x and y using blue circle markers
    ax.scatter(X_norm[cur_index, 0], X_norm[cur_index, 1], X_norm[cur_index, 2], color=cmap(color_index), marker=markers[color_index], label='test data')

    color_index += 1
    # mark top_n influence train data
    ax.scatter(X_norm[if_top_n_samples_indices, 0], X_norm[if_top_n_samples_indices, 1], X_norm[if_top_n_samples_indices, 2], color=cmap(color_index), marker=markers[color_index], label='IF-explain')

    color_index += 1
    # mark top_n influence train data
    ax.scatter(X_norm[reletIF_top_n_samples_indices, 0],
               X_norm[reletIF_top_n_samples_indices, 1],
               X_norm[reletIF_top_n_samples_indices, 2],
               color=cmap(color_index),
               marker=markers[color_index],
               label='relatIF-explain')

    # 设置坐标轴范围
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))

    plt.xticks([])
    plt.yticks([])
    # Add a legend and show the plot
    ax.legend(loc='upper left')

    plt.savefig(f'{result_save_dir}/{test_index}_2d.png')
    if show_plot:
        plt.show()
    plt.close()
    pass


def t_SNE_2d(train_data_inputs, train_data_labels, test_data_inputs, test_data_labels, if_top_n_samples_indices, reletIF_top_n_samples_indices, test_index, result_save_dir='./', show_plot=False):
    # '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    X = np.concatenate((train_data_inputs, test_data_inputs), axis=0)
    # X = train_data_inputs
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    # Define the color map and markers for each class
    cmap = plt.cm.get_cmap('Dark2', 6)
    # cmap = plt.cm.get_cmap('viridis', 6)
    markers = [".", ".", ".", '*', 's', '^', 'D', 'p', 'v', 'h', '+', 'o']

    plt.figure(figsize=(8, 8))
    y_list = [np.argmax(train_data_labels[i]) for i in range(len(train_data_labels))]

    for category in range(3):
        indexes = [i for i, y in enumerate(y_list) if y == category]
        plt.scatter(X_norm[indexes, 0], X_norm[indexes, 1], color=cmap(category), marker=markers[category], label=f'Class {category+1}')

    color_index = 3
    # plot current test data
    cur_index = test_index + len(train_data_inputs)
    # plot x and y using blue circle markers
    plt.scatter(X_norm[cur_index, 0], X_norm[cur_index, 1], color=cmap(color_index), marker=markers[color_index], label='test data')

    color_index += 1
    # mark top_n influence train data
    plt.scatter(X_norm[if_top_n_samples_indices, 0], X_norm[if_top_n_samples_indices, 1], color=cmap(color_index), marker=markers[color_index], label='IF-explain')

    color_index += 1
    # mark top_n influence train data
    plt.scatter(X_norm[reletIF_top_n_samples_indices, 0], X_norm[reletIF_top_n_samples_indices, 1], color=cmap(color_index), marker=markers[color_index], label='relatIF-explain')

    # 设置坐标轴范围
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))

    plt.xticks([])
    plt.yticks([])
    # Add a legend and show the plot
    plt.legend(loc='upper left')

    plt.savefig(f'{result_save_dir}/{test_index}_2d.png')
    if show_plot:
        plt.show()
    plt.close()
    pass


def plot_digit_figure(train_data_inputs, train_data_labels, test_data_input, test_data_label, if_top_n_samples_indices, reletIF_top_n_samples_indices, test_index, result_save_dir='./', show_plot=False):

    # train_dataset.input, test_dataset.input

    # Plot the first six images
    fig, axs = plt.subplots(2, 6, figsize=(20, 6))

    axs[0][0].imshow(test_data_input.reshape((8, 8)), cmap='gray')
    axs[0][0].axis('off')
    axs[0][0].set_title(f'Test_data_{test_index}:{np.argmax(test_data_label)}')

    for i in range(5):
        axs[0][i + 1].imshow(train_data_inputs[if_top_n_samples_indices[i]].reshape((8, 8)), cmap='gray')
        axs[0][i + 1].axis('off')
        axs[0][i + 1].set_title(f'IF_top_{i+1}:{np.argmax(train_data_labels[if_top_n_samples_indices[i]])}')
    # plt.show()

    axs[1][0].axis('off')
    for i in range(5):
        axs[1][i + 1].imshow(train_data_inputs[reletIF_top_n_samples_indices[i]].reshape((8, 8)), cmap='gray')
        axs[1][i + 1].axis('off')
        axs[1][i + 1].set_title(f'relatIF_top_{i+1}:{np.argmax(train_data_labels[reletIF_top_n_samples_indices[i]])}')
    # plt.show()

    plt.savefig(f'{result_save_dir}/{test_index}_figure.png')
    if show_plot:
        plt.show()
    plt.close()
