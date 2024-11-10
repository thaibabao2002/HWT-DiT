from torch.utils.data import DataLoader

def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, collate_fn=None, **kwargs):
    return (
        DataLoader(
            dataset,
            batch_sampler=kwargs['batch_sampler'],
            num_workers=num_workers,
            collate_fn = collate_fn,
            pin_memory=True,
        )
        if 'batch_sampler' in kwargs
        else DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn = collate_fn,
            pin_memory=True,
            **kwargs
        )
    )