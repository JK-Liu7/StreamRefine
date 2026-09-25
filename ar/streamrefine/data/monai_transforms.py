"""Concrete, spawn-pickleable MONAI transforms, imported only by data builders."""

from monai.transforms import Transform
from monai.data import PersistentDataset
from .persistent_pair_dataset import DeterministicPairTransform


class LoadLatentPairTransform(DeterministicPairTransform, Transform):
    """Load and validate continuous KL latents before PersistentDataset caches them.

    Image preprocessing is optional. The separate module keeps lightweight CLI
    and manifest utilities usable without importing MONAI or PyTorch.
    """


class StreamRefinePersistentDataset(PersistentDataset):
    """Load the MONAI metadata enums stored with locally generated cache tensors."""

    def _cachecheck(self, item_transformed):
        import torch
        from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys

        with torch.serialization.safe_globals([MetaKeys, SpaceKeys, TraceKeys]):
            return super()._cachecheck(item_transformed)
