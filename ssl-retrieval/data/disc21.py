from typing import Optional

from torch.utils.data import Dataset


class DISC21Dataset(Dataset):

    def __init__(
        self,
        img_dir: str,
        annotation_file: str ='train_clean.csv',
        transform: Optional[str] = None,
        label_transform: Optional[str] = None,
    ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass