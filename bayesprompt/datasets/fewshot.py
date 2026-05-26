import torch
from torch.utils.data import Dataset, Subset
import numpy as np

class FewShotSampler:
    """
    Samples k-shot support sets ensuring patient-wise separation.
    """
    def __init__(self, dataset: Dataset, k: int, num_classes: int, seed: int = 42, patient_level: bool = True):
        self.dataset = dataset
        self.k = k
        self.num_classes = num_classes
        self.seed = seed
        self.patient_level = patient_level
        self.patient_groups = self._group_by_patient()

    def _group_by_patient(self):
        groups = {}
        for idx in range(len(self.dataset)):
            if hasattr(self.dataset, 'items'):
                patient_id = self.dataset.items[idx]['patient_id']
            else:
                item = self.dataset[idx]
                patient_id = item['patient_id']
                
            if patient_id not in groups:
                groups[patient_id] = []
            groups[patient_id].append(idx)
        return groups

    def sample_support_set(self) -> Subset:
        """
        Samples k examples. If patient_level is True, samples 1 image from k different patients.
        """
        rng = np.random.RandomState(self.seed)
        
        patient_ids = list(self.patient_groups.keys())
        
        if self.patient_level:
            if len(patient_ids) < self.k:
                raise ValueError(f"Cannot sample {self.k} patients from pool of {len(patient_ids)} patients.")
                
            sampled_patients = rng.choice(patient_ids, size=self.k, replace=False)
            
            sampled_indices = []
            for p in sampled_patients:
                valid_indices = self.patient_groups[p]
                # Try to find a positive mask if require_positive is True
                # In real scenario, we'd need to inspect the masks, but to avoid loading everything,
                # we assume the base dataset filter or a separate metadata file marks positives.
                # For this implementation, we just sample a random slice.
                idx = rng.choice(valid_indices)
                sampled_indices.append(idx)
        else:
            all_indices = []
            for indices in self.patient_groups.values():
                all_indices.extend(indices)
            sampled_indices = rng.choice(all_indices, size=self.k, replace=False).tolist()
            
        return Subset(self.dataset, sampled_indices)
