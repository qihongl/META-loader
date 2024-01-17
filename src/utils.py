import os
import sys
import warnings
import torch
import pickle
import numpy as np
from itertools import product

def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.from_numpy(np_array).type(pth_dtype)

def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))

def to_np(torch_tensor):
    return torch_tensor.data.numpy()

def to_sqnp(torch_tensor, dtype=np.float64):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)


def enumerated_product(*args):
    # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def ignore_warnings():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary

    Parameters
    ----------
    input_dict : type
        Description of parameter `input_dict`.
    save_path : type
        Description of parameter `save_path`.

    """
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """load the dict

    Parameters
    ----------
    fpath : type
        Description of parameter `fpath`.

    Returns
    -------
    type
        Description of returned object.

    """
    return pickle.load(open(fpath, "rb"))


def random_walk(n, step_size=.1):
    x = 0
    X = np.zeros(n)
    for t in range(n):
        dx = np.random.choice([step_size, -step_size])
        x += dx
        X[t] = x
    return X

def binarize(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x


def mkdir(dir_name, verbose=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if verbose:
            print(f'Dir created: {dir_name}')
    else:
        if verbose:
            print(f'Dir exist: {dir_name}')



def one_hot_vector(i, k):
    """
    Create a k-dimensional one-hot vector with the i-th dimension set to 1.

    Parameters:
    - i (int): The index of the dimension to set to 1.
    - k (int): The total number of dimensions.

    Returns:
    - list: A one-hot vector represented as a list.
    """
    if i < 0 or i >= k:
        raise ValueError("Invalid index i for dimension k")

    # Create a list of zeros with length k
    one_hot = [0] * k

    # Set the i-th dimension to 1
    one_hot[i] = 1

    return np.array(one_hot)


def move_legend_outside(ax, box_width_factor=.8, bbox_to_anchor=(1, 0.5)):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * box_width_factor, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=bbox_to_anchor)
    return ax


def find_prev_one(arr, t):
    '''
    # Example usage
    binary_array = [0, 0, 0, 1, 0, 0, 1]

    print(find_prev_one(binary_array, 4)) # Output will be 3
    print(find_prev_one(binary_array, 3)) # Output will be 0
    '''
    # Check if t is within the valid range of the array
    if t >= len(arr) or t < 0:
        raise ValueError("Index t is out of the bounds of the array")
    # Iterate backwards from index t
    for i in range(t - 1, -1, -1):
        if arr[i] == 1:
            return i  # Return the index of the closest 1
    # If no 1 is found on the left side, return 0
    return 0

def reverse_dict(input_dict):
    """
    Reverses the keys and values of the input dictionary.

    Args:
    input_dict (dict): A dictionary to reverse.

    Returns:
    dict: A dictionary with keys and values reversed.

    # Example usage
    example_dict = {'a': 1, 'b': 2, 'c': 3}
    reversed_example_dict = reverse_dict(example_dict)
    reversed_example_dict

    """
    # Reversing the dictionary
    reversed_dict = {value: key for key, value in input_dict.items()}
    return reversed_dict

def apply_mask_2d(x, m):
    '''
    Mask a 2D array and preserve the
    dimension on the resulting array
    ----------
    x: np.array
       2D array on which to apply a mask
    m: np.array
        2D boolean mask
    Returns
    -------
    List of arrays. Each array contains the
    elements from the rows in x once masked.
    If no elements in a row are selected the
    corresponding array will be empty
    '''
    take = m.sum(axis=1)
    return np.split(x[m], np.cumsum(take)[:-1])


if __name__ == "__main__":
    '''how to use'''

    # Example usage:
    i = 2  # Index of the dimension to set to 1
    k = 5  # Total number of dimensions
    result = one_hot_vector(i, k)
    print(result)  # Output: [0, 0, 1, 0, 0]
