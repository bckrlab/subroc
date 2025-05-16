from ucimlrepo import fetch_ucirepo
import openml


class DatasetLoader:
    def load():
        raise NotImplementedError()


class UCIDatasetLoader:
    def __init__(self, id):
        self.id = id
    
    def load(self):
        return fetch_ucirepo(id=self.id).data.original


class OpenMLDatasetLoader:
    def __init__(self, id):
        self.id = id

    def load(self, cache_path=None):
        if cache_path is not None:
            openml.config.cache_directory = cache_path + "/openml"

        return openml.datasets.get_dataset(self.id)

