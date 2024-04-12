"""
This file contains the code for loading the TSDF datasets.

Code taken from https://github.com/openai/improved-diffusion, and modifed by Peter Zdravecký.
"""

import numpy as np
from torch.utils.data import DataLoader, Dataset
from . import logger
import blobfile as bf
import torch
import os
import glob
import copy
import scipy.ndimage


def load_data(
    *,
    data_path,
    file_path,
    dataset_name,
    batch_size,
    deterministic=False,
    drop_last=True,
    missing_volumes=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    use_roi=False,
    num_workers=8,
):
    """
    For a dataset, create a generator.

    :param data_path: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param missing_volumes: list of missing volumes to use.
    :param use_roi: if True, use region of interest.
    :param num_workers: number of workers for the DataLoader.
    :param drop_last: if True, drop the last batch if it is smaller than the batch size.
    :param file_path: a file containing the names of the models to be used.
    """

    if not data_path:
        raise ValueError("unspecified data directory")

    if dataset_name == "complete":
        dataset = TSDFDataset(file_path, data_path, missing_volumes, use_roi)
    elif dataset_name == "sr":
        dataset = SRDataset(data_path)
    elif "complete" in dataset_name:
        res = dataset_name.split("_")[-2:]
        if len(res) != 2:
            raise ValueError(f"Unknown dataset name {dataset_name}")
        lr_res = res[0]
        hr_res = res[1]
        dataset = TSDFDataset(
            file_path, data_path, missing_volumes, use_roi, lr_res, hr_res
        )
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    while True:
        yield from loader


def load_data_loader(
    *,
    data_path,
    file_path,
    dataset_name,
    batch_size,
    deterministic=False,
    drop_last=True,
    missing_volumes=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    use_roi=False,
    num_workers=8,
):
    """
    For a dataset, create a generator.

    :param data_path: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param missing_volumes: list of missing volumes to use.
    :param use_roi: if True, use region of interest.
    :param num_workers: number of workers for the DataLoader.
    :param drop_last: if True, drop the last batch if it is smaller than the batch size.
    :param file_path: a file containing the names of the models to be used.
    """

    if not data_path:
        raise ValueError("unspecified data directory")

    if dataset_name == "complete":
        dataset = TSDFDataset(file_path, data_path, missing_volumes, use_roi)
    elif dataset_name == "sr":
        dataset = SRDataset(data_path)
    elif "complete" in dataset_name:
        res = dataset_name.split("_")[-2:]
        if len(res) != 2:
            raise ValueError(f"Unknown dataset name {dataset_name}")
        lr_res = res[0]
        hr_res = res[1]
        dataset = TSDFDataset(
            file_path, data_path, missing_volumes, use_roi, lr_res, hr_res
        )
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return loader


class TSDFDataset(Dataset):
    """
    A dataset of TSDFs.

    :param data_path: a directory containing TSDFs.
    :param file_path: a file containing the names of the models to be used.
    """

    def __init__(
        self,
        file_path,
        data_path,
        missing_volumes=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
        use_roi=False,
        lr_res="",
        gt_sr="",
    ):
        super().__init__()
        self._file = file_path
        self._data_path = data_path
        self._missing_volumes = missing_volumes
        self._use_roi = use_roi
        self._gt_sr = gt_sr
        self._lr_res = lr_res
        self.data = []

        if not os.path.exists(self._file):
            raise ValueError(f"file {self._file} does not exist")

        if not os.path.exists(self._data_path):
            raise ValueError(f"directory {self._data_path} does not exist")

        self.get_data()

    def get_data(self):
        logger.log(
            f"Loading data from {self._data_path} and filtering with {self._file}"
        )

        filter_ids = {}
        with open(self._file, "r") as data:
            filter_ids = data.readlines()
            filter_ids = {f.split("\n")[0] for f in filter_ids}

        folders = [
            f
            for f in bf.listdir(self._data_path)
            if bf.isdir(bf.join(self._data_path, f))
        ]
        folders = [f for f in folders if f != "gt"]
        folders = [f for f in folders if f in self._missing_volumes]

        holes = []
        for folder in folders:
            files = bf.listdir(bf.join(self._data_path, folder))
            holes.extend([bf.join(self._data_path, folder, f) for f in files])

        holes_and_gt = [
            (
                hole,
                bf.join(
                    self._data_path,
                    "gt",
                    "_".join(hole.split("/")[-1].split("_")[:-1]) + ".npy",
                ),
            )
            for hole in holes
        ]

        # Filter files
        holes_and_gt = [
            f for f in holes_and_gt if f[0].split("/")[-1].split(".")[0] in filter_ids
        ]

        if self._gt_sr:
            holes_and_gt = [
                (f[0], f[1].replace(f"{self._lr_res}/gt", f"{self._gt_sr}/gt"))
                for f in holes_and_gt
            ]

        missing_count = 0
        corrected = []
        missing = False
        for hole, gt in holes_and_gt:
            missing = False
            if not bf.exists(hole):
                missing_count += 1
                missing = True
            if not bf.exists(gt):
                missing_count += 1
                missing = True

            if not missing:
                corrected.append((hole, gt))

        logger.log(f"Missing counterpats: {missing_count}")
        holes_and_gt = corrected

        logger.log(f"Final dataset size: {len(holes_and_gt)}")
        self.data = holes_and_gt

    def normalize(self, sdf):
        sdf = sdf.clip(-1, 1)
        return sdf

    def get_roi(self, gt, hole):
        gt_mask = np.zeros(gt.shape)
        gt_mask[np.where(gt <= 1e-10)] = 1

        hole_mask = np.zeros(hole.shape)
        hole_mask[np.where(hole <= 1e-10)] = 1

        final_mask = gt_mask - hole_mask

        nonzero_indices = np.nonzero(final_mask)

        min_indices = np.min(nonzero_indices, axis=1)
        max_indices = np.max(nonzero_indices, axis=1)

        bbox_mask = np.ones_like(final_mask)
        bbox_mask[
            min_indices[0] : max_indices[0] + 1,
            min_indices[1] : max_indices[1] + 1,
            min_indices[2] : max_indices[2] + 1,
        ] = -1

        return bbox_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hole_path, gt_path = self.data[idx]

        with bf.BlobFile(hole_path, "rb") as f:
            hole = np.load(f)
        with bf.BlobFile(gt_path, "rb") as f:
            gt = np.load(f)

        hole = self.normalize(hole)
        gt = self.normalize(gt)

        if self._use_roi:
            roi_mask = self.get_roi(gt, hole)
            roi_mask = torch.from_numpy(roi_mask).unsqueeze(0).float()

        gt = torch.from_numpy(gt).unsqueeze(0).float()
        hole = torch.from_numpy(hole).unsqueeze(0).float()

        if self._use_roi:
            hole = torch.cat([hole, roi_mask], dim=0)

        gt_name = gt_path.split("/")[-1].split(".")[0]

        return gt, hole, gt_name


class SRDataset(Dataset):
    """
    A dataset of TSDFs for super resolution.

    :param data_path: a directory containing TSDFs.
    """

    def __init__(self, data_path):
        super().__init__()
        self._data_path = data_path
        self.scale_factor = 0
        self._low_res = 0
        self._high_res = 0
        self.data = []

        if not os.path.exists(self._data_path):
            raise ValueError(f"directory {self._data_path} does not exist")

        self.get_data()

    def get_data(self):
        logger.log(f"Loading data from {self._data_path}")
        files = glob.glob(self._data_path + "/**/*.npz", recursive=True)
        data = np.load(files[0])
        self._low_res = data["lr"].shape[0]
        self._high_res = data["hr"].shape[0]

        self.scale_factor = self._high_res // self._low_res

        # Check all files are consistent
        for file in files:
            data = np.load(file)
            assert data["lr"].shape[0] == self._low_res, f"File {file} has wrong shape"
            assert data["hr"].shape[0] == self._high_res, f"File {file} has wrong shape"
            lr_model = self.upscale(data["lr"])
            hr_model = data["hr"]
            name = file.split("/")[-1].split(".")[0]
            lr_model = self.normalize(lr_model)
            hr_model = self.normalize(hr_model)
            self.data.append((lr_model, hr_model, name))

        logger.log(f"Final dataset size: {len(self.data)}")

    def normalize(self, sdf):
        sdf = sdf.clip(-1, 1)
        return sdf

    def upscale(self, sdf):
        return scipy.ndimage.zoom(sdf, self.scale_factor, order=3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lr, hr, name = self.data[idx]

        low_res = copy.deepcopy(lr)
        high_res = copy.deepcopy(hr)

        low_res = torch.from_numpy(low_res).unsqueeze(0).float()
        high_res = torch.from_numpy(high_res).unsqueeze(0).float()

        return high_res, low_res, name
