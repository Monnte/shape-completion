import os
import logging
import csv
import numpy as np
import multiprocessing as mp
import glob
import objaverse
import argparse


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


class ObjaverseModels:
    def __init__(self, category_names, tag_names, dataset_folder):
        self.category_names = category_names
        self.tag_names = tag_names
        self.dataset_folder = dataset_folder
        self.tmp_ids_file = os.path.join(dataset_folder, "objaverse_ids.npy")

        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder, exist_ok=True)

    def filter_models(self, annotations):
        logging.info(
            f"Filtering models with categories {self.category_names} and tags {self.tag_names}"
        )
        f = open(os.path.join(self.dataset_folder, "categories.csv"), "w")
        writer = csv.writer(f)
        writer.writerow(["model_id", "category", "tag"])

        models_ids = []
        for key, value in annotations.items():
            categories = value["categories"]
            tags = value["tags"]
            for category in categories:
                if category["name"] in self.category_names:
                    for tag in tags:
                        if tag["name"] in self.tag_names:
                            models_ids.append(key)
                            writer.writerow([key, category["name"], tag["name"]])

        f.close()
        return models_ids

    def get_models(self):
        logging.info("Getting models from objaverse")
        model_ids = []
        if os.path.exists(self.tmp_ids_file):
            logging.info("Loading model ids from file")
            model_ids = np.load(self.tmp_ids_file)
        else:
            logging.info("Downloading model ids from objaverse")
            annotations = objaverse.load_annotations()
            model_ids = self.filter_models(annotations)
            np.save(self.tmp_ids_file, model_ids)

        logging.info(f"Found {len(model_ids)} models")
        logging.info("Downloading models from objaverse")
        models = objaverse.load_objects(model_ids, download_processes=mp.cpu_count())
        return list(models.values())


class ModelNetModels:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

        assert os.path.exists(dataset_folder)

        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder, exist_ok=True)

    def get_models(self):
        files = glob.glob(self.dataset_folder + "/**/*.off", recursive=True)
        return files


class ShapeNetModels:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

        assert os.path.exists(dataset_folder)

        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder, exist_ok=True)

    def get_models(self):
        files = glob.glob(self.dataset_folder + "/**/*.off", recursive=True)
        return files
