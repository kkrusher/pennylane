import pennylane as qml
from pennylane import numpy as np
import torch
from sklearn.preprocessing import normalize

import time
from if_plot import t_SNE_2d, t_SNE_3d, plot_digit_figure


def get_influence_function(cost_fn, params, train_data_inputs, train_data_labels, test_data_inputs, test_data_labels, top_n=5, hessian_approximate_number=None, log=None, plot_type='t_SNE_2d', show_plot=True, result_save_dir='./'):
    # we assume the cost_fn take a batch of data as input, where the first dimension record data index in a batch
    # test_data should be a batch of test data
    # plot_type should be one of ['t_SNE_2d', 't_SNE_3d', 'figure']
    grad_fn = qml.grad(cost_fn)
    hess_fn = qml.jacobian(grad_fn)
    params_num = np.cumprod(params.shape)[-1]

    hessian_I = get_hessian_I(hess_fn, params, train_data_inputs, train_data_labels, approximate_number=hessian_approximate_number, log=log)
    train_grad_on_epsilon_list, normed_train_grad_on_epsilon_list = get_grad_on_epsilon(grad_fn, hessian_I, params, train_data_inputs, train_data_labels, log=log)

    for test_index, test_data in enumerate(zip(test_data_inputs, test_data_labels)):
        if log is not None:
            log.info(f"test_index: {test_index}, data: {test_data}")

        # cost_fn conduct on a batch of data
        input = np.array(test_data[0], requires_grad=False)
        inputs = np.expand_dims(input, axis=0)
        label = np.array(test_data[1], requires_grad=False)
        labels = np.expand_dims(label, axis=0)

        g_test = grad_fn(params, x=inputs, y=labels)
        g_test = np.reshape(g_test, (params_num, ))

        if_top_n_samples_indices = _get_influence_function_top_n(train_data_inputs, train_data_labels, train_grad_on_epsilon_list, g_test, top_n=top_n, log=log, log_tag='if_top_n_samples_indices')
        relatIF_top_n_samples_indices = _get_influence_function_top_n(train_data_inputs, train_data_labels, normed_train_grad_on_epsilon_list, g_test, top_n=top_n, log=log, log_tag='relatIF_top_n_samples_indices')

        if plot_type == 't_SNE_2d':
            t_SNE_2d(train_data_inputs, train_data_labels, test_data_inputs, test_data_labels, if_top_n_samples_indices, relatIF_top_n_samples_indices, test_index, result_save_dir, show_plot=show_plot)
        elif plot_type == 't_SNE_3d':
            t_SNE_3d(train_data_inputs, train_data_labels, test_data_inputs, test_data_labels, if_top_n_samples_indices, relatIF_top_n_samples_indices, test_index, result_save_dir, show_plot=show_plot)
        elif plot_type == 'figure':
            plot_digit_figure(train_data_inputs, train_data_labels, input, label, if_top_n_samples_indices, relatIF_top_n_samples_indices, test_index, result_save_dir, show_plot=show_plot)
        else:
            raise RuntimeError(f"{plot_type} is not a type of plot.")


def get_hessian_I(hess_fn, params, train_data_inputs, train_data_labels, approximate_number=None, log=None):
    """_summary_

    Args:
        hess_fn (_type_): _description_
        params (_type_): _description_
        train_dataset (_type_): _description_
        approximate_number (int, optional): As calculate Hessian is computationally expensive, when approximate_number is assigned, 
            only approximate_number times of calculation of Hessian is conduct to approximate Hessian.
            When set to None, conduct exact computation. Defaults to None.
        log (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # get hessian
    starttime = time.time()
    if log is not None:
        log.info("Start calculating Hessian....")
    hessian = 0
    params_num = np.cumprod(params.shape)[-1]
    if approximate_number is not None:
        if approximate_number < len(train_data_inputs):
            train_data_inputs = train_data_inputs[:approximate_number]
            train_data_labels = train_data_labels[:approximate_number]
            # train_dataset = torch.utils.data.Subset(train_dataset, list(range(approximate_number)))

    for times, data in enumerate(zip(train_data_inputs, train_data_labels), 1):
        inputs = np.array(data[0], requires_grad=False)
        inputs = np.expand_dims(inputs, axis=0)
        labels = np.array(data[1], requires_grad=False)
        labels = np.expand_dims(labels, axis=0)
        hessian += hess_fn(params, x=inputs, y=labels)
        if log is not None and times < 5:
            log.info(f"times: {times}, hessian: {hessian}")
        # to speedup debug
        if __debug__:
            break
    if not __debug__:
        hessian = hessian / len(train_data_inputs)
    hessian = np.matrix(np.reshape(hessian, (params_num, params_num)))
    hessian_I = np.linalg.pinv(hessian)
    # hessian_I = hessian.I
    if log is not None:
        log.info(f"hessian_I: {hessian_I}")
        log.info(f"Time used for optimization of cur epoch is:{time.time() - starttime:10.3f}")
    return hessian_I


def get_grad_on_epsilon(grad_fn, hessian_I, params, train_data_inputs, train_data_labels, log=None):
    # get grad on epsilon
    params_num = np.cumprod(params.shape)[-1]
    train_grad_on_epsilon_list = []
    normed_train_grad_on_epsilon_list = []
    for train_index, data in enumerate(zip(train_data_inputs, train_data_labels)):
        if log is not None and train_index < 5:
            log.info(f"train_index: {train_index}, data: {data}")
        inputs = np.array(data[0], requires_grad=False)
        inputs = np.expand_dims(inputs, axis=0)
        labels = np.array(data[1], requires_grad=False)
        labels = np.expand_dims(labels, axis=0)
        g_i = grad_fn(params, x=inputs, y=labels)

        grad_on_epsilon = hessian_I @ np.reshape(g_i, (params_num, 1))
        train_grad_on_epsilon_list.append(grad_on_epsilon)

        norm_grad_on_epsilon = normalize(grad_on_epsilon, axis=0)
        normed_train_grad_on_epsilon_list.append(norm_grad_on_epsilon)
        if log is not None and train_index < 5:
            log.info(f"train_index: {train_index}, grad_on_epsilon: {grad_on_epsilon}")
    return train_grad_on_epsilon_list, normed_train_grad_on_epsilon_list


def _get_influence_function_top_n(train_data_inputs, train_data_labels, train_grad_on_epsilon_list, g_test, top_n=5, log=None, log_tag=None):
    if_list = []
    for times in range(len(train_data_inputs)):
        if_list.append(g_test @ train_grad_on_epsilon_list[times])
    if_top_n_samples_indices = _get_indices_with_largest_values(if_list, top_n)
    if log is not None:
        log.info(f"{log_tag}:{if_top_n_samples_indices}")
        for i in if_top_n_samples_indices:
            log.info(f"train_dataset[{i}]: {train_data_inputs[i]}, label: {train_data_labels[i]}")
    return if_top_n_samples_indices


def _get_indices_with_largest_values(my_list, n):
    sorted_indices = sorted(range(len(my_list)), key=lambda i: my_list[i], reverse=True)
    indices_with_largest_values = sorted_indices[:n]
    return indices_with_largest_values
