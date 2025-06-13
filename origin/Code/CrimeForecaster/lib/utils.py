import os
import pickle
import logging
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def load_graph_data(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
        return sensor_ids, sensor_id_to_ind, adj_mx


def get_logger(log_dir, name, log_file, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def add_simple_summary(writer, names, values, global_step):
    for name, value in zip(names, values):
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=name, simple_value=value)])
        writer.add_summary(summary, global_step)


def get_total_trainable_parameter_size():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters


def calculate_scaled_laplacian(adj_mx, lambda_max=2):
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    if lambda_max is None:
        lambda_max = sp.linalg.eigs(laplacian, k=1, which='LR')[0].real
    laplacian = (2 / lambda_max * laplacian) - sp.eye(adj.shape[0])
    return laplacian.astype(np.float32)


def calculate_random_walk_matrix(adj_mx):
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1)).flatten()
    d_inv = np.power(d, -1)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj).astype(np.float32)


class DataLoader:
    def __init__(self, xs, ys, batch_size):
        self.batch_size = batch_size
        self.xs = xs
        self.ys = ys
        self.size = len(xs)
        self._idx = 0

    def get_iterator(self):
        while self._idx + self.batch_size <= self.size:
            x = self.xs[self._idx:self._idx + self.batch_size]
            y = self.ys[self._idx:self._idx + self.batch_size]
            self._idx += self.batch_size
            yield x, y
        self._idx = 0


def load_dataset(dataset_dir, batch_size, val_batch_size=None, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, f'{category}.npz')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing dataset file: {file_path}")
        cat_data = np.load(file_path)
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['x_offsets'] = cat_data['x_offsets']
        data['y_offsets'] = cat_data['y_offsets']

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], val_batch_size or batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size or batch_size)
    data['y_test'] = data['y_test']
    print("Loaded test set shape:", data['x_test'].shape, data['y_test'].shape)


    class DummyScaler:
        def transform(self, x): return x
        def inverse_transform(self, x): return x

    data['scaler'] = DummyScaler()
    
    return data