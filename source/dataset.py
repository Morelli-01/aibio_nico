import os, sys

from cv2 import sqrt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision.transforms import v2
from sklearn.preprocessing import LabelEncoder


class Rxrx1(Dataset):

    def __init__(self, root_dir=None, metadata_path: str = None, dataframe: pd.DataFrame = None, dataset_norm=False):
        super().__init__()

        if metadata_path is None and dataframe is None:
            raise RuntimeError('Rxrx1 dataset needs either a metadata absolute path or a pd dataframe containing the metadata.\n \
                               Not both!!!')
        if metadata_path is not None and dataframe is not None:
            raise RuntimeError('Rxrx1 dataset only need ONE of: metadata_path of dataframe. NOT BOTH!!!')

        if root_dir is None:
            raise RuntimeError('Rxrx1 dataset needs to be explicitly initialized with a root_dir')

        self.dataset_norm = dataset_norm
        self.root_dir = os.path.join(root_dir, "rxrx1_v1.0")
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f'Rxrx1 dataset was initialized with a non-existing root_dir: {self.root_dir}')
        self.imgs_dir = os.path.join(self.root_dir, "images")
        if metadata_path is not None:
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = dataframe.copy(deep=True)

        self.cell_type_idx = self.metadata.columns.get_loc("cell_type")
        self.le_cell_type = LabelEncoder()
        self.le_cell_type.fit(self.metadata['cell_type'].unique())
        self.metadata['cell_type'] = self.le_cell_type.transform(self.metadata['cell_type'])

        self.experiment_idx = self.metadata.columns.get_loc("experiment")
        self.le_experiment = LabelEncoder()
        self.le_experiment.fit(self.metadata['experiment'].unique())
        # self.metadata['experiment'] = self.le_experiment.transform(self.metadata['experiment'])

        self.items = [(os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' +
                                    str(item.site) + '.png'), item.sirna_id, list(item)) for item in self.metadata.itertuples(index=False)]
        for _, _, meta in self.items:
            meta[self.experiment_idx] = self.le_experiment.transform([meta[self.experiment_idx]])
        self.num_classes = torch.tensor([cls for _, cls, _ in self.items]).unique().shape[0]
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=(not self.dataset_norm))
        ])

    def normalize_(self, img, meta):
        mean = torch.tensor(eval(meta[-2]))
        var = torch.tensor(eval(meta[-1]))
        img = ((img.permute(2, 1, 0) - mean) / torch.sqrt(var)).permute(2, 1, 0)
        return img

    def __getitem__(self, index):
        img_path, sirna_id, metadata = self.items[index]
        img = self.transforms(Image.open(img_path))
        if self.dataset_norm:
            return (self.normalize_(img, metadata), sirna_id, metadata)
        return (img, sirna_id, metadata)

    def __len__(self):
        return len(self.items)
