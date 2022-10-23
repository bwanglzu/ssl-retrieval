from typing import Literal

from torchvision import transforms

_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


class BaseTransform:
    def __init__(self, transform, phase: Literal['train', 'eval'] = 'train'):
        if transform:
            self._transform = transform
        else:
            if phase == 'train':
                self._transform = self.default_train_transform
            else:
                self._transform = self.default_eval_transform

    def __call__(self, x):
        return self._transform(x)

    @property
    def default_train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_DEFAULT_MEAN, _DEFAULT_STD),
            ]
        )

    @property
    def default_eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_DEFAULT_MEAN, _DEFAULT_STD),
            ]
        )


class ContrastiveTransform(BaseTransform):
    def __init__(self, transform, num_views):
        self._num_views = num_views
        super().__init__(transform=transform)

    def __call__(self, x):
        return [self._transform(x) for _ in range(self._num_views)]
