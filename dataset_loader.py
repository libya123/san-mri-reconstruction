import torch
from fastmri.data import transforms as T
from fastmri.data.mri_data import SliceDataset

def get_fastmri_loader(root_dir, batch_size=8, train=True):
    dataset = SliceDataset(
        root=root_dir,
        transform=T.to_tensor,  
        challenge="singlecoil"  
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
