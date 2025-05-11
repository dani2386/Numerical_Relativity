import h5py
import os


class HDF5File:
    def __init__(self, filename):
        self.filename = filename

        if not os.path.exists(self.filename):
            with h5py.File(self.filename, 'w') as file:
                    pass

    def dataset(self, dataset):
        with h5py.File(self.filename, 'r') as file:
            return dataset in file

    def create_dataset(self, dataset, shape):
        with h5py.File(self.filename, 'a') as file:
            if dataset in file: del file[dataset]
            file.create_dataset(dataset, shape=shape, maxshape=(None,) * len(shape), compression="gzip")

    def resize(self, dataset, shape):
        with h5py.File(self.filename, 'a') as file:
            file[dataset].resize(shape)

    def copy_metadata(self, source_dataset, target_dataset):
        with h5py.File(self.filename, "a") as file:
            for key, value in file[source_dataset].attrs.items():
                file[target_dataset].attrs[key] = value

    def save_metadata(self, dataset, **kwargs):
        with h5py.File(self.filename, 'a') as file:
            for key, value in kwargs.items():
                file[dataset].attrs[key] = value

    def load_metadata(self, dataset, keys):
        with h5py.File(self.filename, 'r') as file:
            if len(keys) == 1:
                return file[dataset].attrs[keys[0]]
            else:
                return tuple(file[dataset].attrs[key] for key in keys)

    def save(self, dataset, index, data):
        with h5py.File(self.filename, 'a') as file:
            file[dataset][index] = data

    def load(self, dataset, index):
        with h5py.File(self.filename, 'r') as file:
            return file[dataset][index]

    def delete(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)