import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image

class BaseMedicalDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None, synthetic_fallback: bool = False, split_file: str = "splits.json", image_size: int = 512):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.synthetic_fallback = synthetic_fallback
        self.image_size = image_size
        
        self.items = self._load_items(split_file)

    def _load_items(self, split_file):
        """
        Loads items from a JSON split file.
        Expected format:
        {
            "train": [{"image": "path/to/img", "mask": "path/to/mask", "patient_id": "p1"}, ...],
            "test": [...]
        }
        """
        if self.synthetic_fallback:
            return [{"image": f"synthetic_{i}.png", "mask": f"synthetic_mask_{i}.png", "patient_id": f"p{i//5}"} for i in range(100)]
            
        split_path = os.path.join(self.root_dir, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file {split_path} not found. Please provide valid datasets or enable synthetic_fallback.")
            
        with open(split_path, 'r') as f:
            splits = json.load(f)
            
        return splits.get(self.split, [])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        if self.synthetic_fallback:
            img = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
            mask[100:200, 100:200] = 1 
            if idx % 2 == 0:
                mask[300:400, 300:400] = 2
        else:
            img_path = os.path.join(self.root_dir, item['image'])
            mask_path = os.path.join(self.root_dir, item['mask'])
            
            # Load images
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path)
            
            # Resize
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            
            img = np.array(img)
            mask = np.array(mask)

        sample = {'image': img, 'mask': mask, 'patient_id': item['patient_id']}

        if self.transform:
            sample = self.transform(sample)
        
        if not isinstance(sample['image'], torch.Tensor):
            sample['image'] = torch.from_numpy(sample['image'].transpose(2, 0, 1)).float() / 255.0
        if not isinstance(sample['mask'], torch.Tensor):
            sample['mask'] = torch.from_numpy(sample['mask']).long()

        return sample
