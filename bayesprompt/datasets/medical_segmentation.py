from .base import BaseMedicalDataset
from ..core.decorators import register_dataset

@register_dataset("AMOS")
class AMOSDataset(BaseMedicalDataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None, modality: str = 'CT', fast_dev_run: bool = False, image_size: int = 512, split_file: str = "splits.json"):
        super().__init__(root_dir, split, transform, synthetic_fallback=synthetic, split_file=split_file, image_size=image_size)
        self.modality = modality

@register_dataset("BraTS")
class BraTSDataset(BaseMedicalDataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None, modality: str = 'MRI', fast_dev_run: bool = False, image_size: int = 512, split_file: str = "splits.json"):
        super().__init__(root_dir, split, transform, synthetic_fallback=synthetic, split_file=split_file, image_size=image_size)
        self.modality = modality

@register_dataset("RotatorCuff")
class RotatorCuffDataset(BaseMedicalDataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None, modality: str = 'US', fast_dev_run: bool = False, image_size: int = 512, split_file: str = "splits.json"):
        super().__init__(root_dir, split, transform, synthetic_fallback=synthetic, split_file=split_file, image_size=image_size)
        self.modality = modality
