import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    #from data.aligned_dataset import AlignedDataset

    if opt.dataset_name == 'semantic_mpv':
        from data.semantic_mpv_dataset import MpvDataset
    elif opt.dataset_name == 'viton_semantic':
        from data.viton_semantic_dataset import VitonDataset
    elif opt.dataset_name == 'content_fusion_mpv':
        from data.content_fusion_dataset import MpvDataset
    elif opt.dataset_name == 'content_fusion_viton':
        from data.viton_content_fusion_dataset import MpvDataset
    elif opt.dataset_name == 'gmm_tps_content_fusion_mpv':
        from data.gmm_tps_content_fusion_dataset import MpvDataset
    elif opt.dataset_name == 'full_semantic__mpv':
        from data.full_semantic_mpv_dataset import MpvDataset
    else:
        raise ValueError("dataset name error")
    #dataset = AlignedDataset()
    if opt.dataset_name == 'viton_semantic':
        dataset = VitonDataset()
    else:
        dataset = MpvDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
