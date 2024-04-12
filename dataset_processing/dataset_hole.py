"""
Author: Peter Zdravecký
"""

import multiprocessing as mp
import trimesh
import numpy as np
import os
import mesh2sdf
import argparse
import glob
import logging
import time

from datasets import (
    ObjaverseModels,
    ModelNetModels,
    ShapeNetModels,
    add_dict_to_argparser,
)


class DatasetGenerator:
    """
    Generate TSDF dataset for shape completion

    :param dataset_folder: folder to save the dataset
    :param model_paths: list of paths to the models
    :param N: resolution of the TSDF grid
    :param per_model: number of counteparts to create per model
    :param max_holes_per_model: maximum number of holes to create per model
    :param mising_volume_threshold: range of missing volume to consider
    :param primitive_size_range: range of sizes for the primitive objects
    :param object_list: list of primitive objects to use
    :param multiprocessing: use multiprocessing
    :param cpu_count: number of cpus to use
    :param timeout: timeout for each model
    """

    def __init__(
        self,
        dataset_folder,
        model_paths,
        N=32,
        per_model=5,
        max_holes_per_model=5,
        mising_volume_threshold=(0.05, 0.9),
        primitive_size_range=(0.5, 0.9),
        object_list=[],
        multiprocessing=True,
        cpu_count=mp.cpu_count(),
        timeout=60 * 3,
    ):
        self.dataset_folder = dataset_folder
        self.models = model_paths
        self.N = N
        self.level = 2 / N
        self.object_list = object_list
        self.mising_volume_threshold = mising_volume_threshold
        self.per_model = per_model
        self.primitive_size_range = primitive_size_range
        self.max_holes_per_model = max_holes_per_model
        self.multiprocessing = multiprocessing
        self.object_list = (
            [
                "box",
                "cylinder",
                "cone",
                "capsule",
                "uv_sphere",
                "annulus",
                "icosahedron",
            ]
            if not object_list
            else object_list
        )
        self.cpu_count = cpu_count
        self.timeout = timeout
        self.checkpoint_file = os.path.join(self.dataset_folder, "checkpoint.npy")
        self.last_entry = 0

        os.makedirs(self.dataset_folder, exist_ok=True)
        if not multiprocessing:
            if os.path.exists(self.checkpoint_file):
                self.last_entry = np.load(self.checkpoint_file)
                logging.info(f"Loaded checkpoint {self.last_entry}")
            else:
                np.save(self.checkpoint_file, self.last_entry)

        self.models = self.models[self.last_entry :]

    def create_folder_structure(self):
        os.makedirs(self.dataset_folder, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_folder, f"{self.N}"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_folder, f"{self.N}", "gt"), exist_ok=True)

        for i in range(
            int(self.mising_volume_threshold[0] * 10),
            int(self.mising_volume_threshold[1] * 10) + 1,
        ):
            os.makedirs(
                os.path.join(self.dataset_folder, f"{self.N}", f"{i/10}"), exist_ok=True
            )

    def load_model(self, model_path):
        new_mesh = trimesh.load(model_path, force="mesh")

        # scale and center the mesh ( The vertices of the input mesh, the vertices MUST be in range [-1, 1])
        mesh_scale = 0.8
        vertices = new_mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale
        new_mesh.vertices = vertices

        return new_mesh

    def subdivide_mesh(self, mesh):
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return new_mesh

    def convert_mesh_to_tsdf(self, mesh):
        sdf = mesh2sdf.compute(
            mesh.vertices, mesh.faces, self.N, fix=True, level=self.level
        )
        return sdf

    def check_mesh(self, mesh):
        """
        Check if the mesh after difference operation is empty
        """
        obj_mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        obj_mesh.apply_translation([0, 0, 0])
        diff_mesh = mesh.difference(obj_mesh, check_volume=False)
        if len(diff_mesh.vertices) == 0:
            return True
        return False

    def get_primitive_object(self, object_type, size_range=(0.5, 0.9)):
        obj_mesh = None
        if object_type == "box":
            x_size = np.random.uniform(*size_range)
            y_size = np.random.uniform(*size_range)
            z_size = np.random.uniform(*size_range)
            obj_mesh = trimesh.creation.box(extents=[x_size, y_size, z_size])
        elif object_type == "cylinder":
            random_radius = np.random.uniform(*size_range)
            random_height = np.random.uniform(*size_range)
            obj_mesh = trimesh.creation.cylinder(
                radius=random_radius, height=random_height
            )
        elif object_type == "cone":
            random_radius = np.random.uniform(*size_range)
            random_height = np.random.uniform(*size_range)
            obj_mesh = trimesh.creation.cone(radius=random_radius, height=random_height)
        elif object_type == "capsule":
            random_radius = np.random.uniform(*size_range)
            random_height = np.random.uniform(*size_range)
            obj_mesh = trimesh.creation.capsule(
                radius=random_radius, height=random_height
            )
        elif object_type == "uv_sphere":
            random_radius = np.random.uniform(*size_range)
            obj_mesh = trimesh.creation.uv_sphere(radius=random_radius)
        elif object_type == "annulus":
            random_radius = np.random.uniform(*size_range)
            random_height = np.random.uniform(*size_range)
            obj_mesh = trimesh.creation.annulus(
                r_min=random_radius, r_max=2 * random_radius, height=random_height
            )
        elif object_type == "icosahedron":
            obj_mesh = trimesh.creation.icosahedron()
        else:
            raise NotImplementedError
        return obj_mesh

    def random_translate(self, obj_mesh, mesh):
        """
        Trasnlate the object to a random position relative to the mesh size to avoid holes outside the mesh
        """
        bbmin, bbmax = mesh.bounds * 0.8
        x = np.random.uniform(bbmin[0], bbmax[0])
        y = np.random.uniform(bbmin[1], bbmax[1])
        z = np.random.uniform(bbmin[2], bbmax[2])
        obj_mesh.apply_translation([x, y, z])
        return obj_mesh

    def random_rotate(self, mesh):
        mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
        return mesh

    def create_hole(self, mesh, sdf_orig):
        holes_to_make = np.random.randint(1, self.max_holes_per_model + 1)
        diff_mesh = mesh.copy()

        for _ in range(holes_to_make):
            object_type = np.random.choice(self.object_list)
            obj_mesh = self.get_primitive_object(object_type, self.primitive_size_range)
            obj_mesh = self.random_translate(obj_mesh, mesh)
            obj_mesh = self.random_rotate(obj_mesh)

            diff_mesh = diff_mesh.difference(obj_mesh, check_volume=False)

        # skip if mesh is empty
        if len(diff_mesh.vertices) == 0:
            return None

        try:
            sdf_hole = self.convert_mesh_to_tsdf(diff_mesh)
        except:
            return None

        if np.all(sdf_hole == sdf_orig):
            return None

        volume_orig = np.count_nonzero(sdf_orig <= 0)
        volume_hole = np.count_nonzero(sdf_hole <= 0)

        volume_missing = (volume_orig - volume_hole) / volume_orig

        if (
            volume_missing > self.mising_volume_threshold[0]
            and volume_missing < self.mising_volume_threshold[1]
        ):
            return sdf_hole, volume_missing
        else:
            # too much or too little volume missing
            return None

    def save_hole(self, sdf, volume_missing, model_name, count):
        volume_missing = np.round(volume_missing, decimals=1)
        save_path = os.path.join(
            self.dataset_folder,
            f"{self.N}",
            f"{volume_missing}",
            f"{model_name}_{count}.npy",
        )
        np.save(save_path, sdf)

    def save_gt(self, sdf, model_name):
        save_path = os.path.join(
            self.dataset_folder, f"{self.N}", "gt", f"{model_name}.npy"
        )
        np.save(save_path, sdf)

    def process_model(self, model_path):
        logging.info(f"Processing model {model_path}")
        mesh = self.load_model(model_path)

        if len(mesh.faces) > 1_000_000:
            logging.info(f"Too many faces for model {model_path}")
            return None

        # check mesh if eligible
        if self.check_mesh(mesh):
            # try to subdivide
            mesh = self.subdivide_mesh(mesh)
            if self.check_mesh(mesh):
                logging.info(f"Unable to create hole for model {model_path}")
                return None

        if len(mesh.faces) > 1_000_000:
            logging.info(f"Too many faces for model {model_path}")
            return None

        model_name = os.path.basename(model_path)
        try:
            gt_sdf = self.convert_mesh_to_tsdf(mesh)
        except:
            logging.info(f"Unable to convert mesh to tsdf for model {model_path}")
            return None

        self.save_gt(gt_sdf, model_name)

        tries = 0
        count = 0
        while count < self.per_model:
            sdf_hole = self.create_hole(mesh, gt_sdf)
            if sdf_hole:
                sdf, volume_missing = sdf_hole
                self.save_hole(sdf, volume_missing, model_name, count)
                count += 1
            tries += 1
            if tries > self.per_model * 20:
                logging.info(f"Too many tries for model {model_path}")
                return None

        logging.info(f"Finished processing model {model_path}")

    def run(self):
        self.create_folder_structure()
        if self.multiprocessing:
            pool = mp.Pool(self.cpu_count)
            pool.map(self.process_model, self.models)
        else:
            for i, model in enumerate(self.models):
                if i % 100 == 0:
                    logging.info(f"Processing model {i}/{len(self.models)}")
                p = mp.Process(target=self.process_model, args=(model,))
                p.daemon = True
                p.start()
                start = time.time()
                while time.time() - start < self.timeout:
                    if not p.is_alive():
                        break
                    time.sleep(1)
                if p.is_alive():
                    logging.info(f"Timeout for model {model}")
                    p.terminate()
                    p.join()
                p.join()

                self.last_entry += 1
                np.save(self.checkpoint_file, self.last_entry)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    default_dict = {
        "N": 32,
        "source": "",
        "output": "datasets/objaverse_dataset",
        "dataset": "objaverse",  # objaverse, modelnet, shapenet
        "per_model": 5,
        "max_holes_per_model": 5,
        "mising_volume_threshold": (0.05, 0.9),
        "primitive_size_range": (0.5, 0.9),
        "multiprocessing": False,
        "cpu_count": 8,
        "timeout": 60 * 3,
        "filter_path": "",  # .txt file with model names to filter
    }

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_dict)
    parser.add_argument("--category_names", nargs="+", default=["furniture-home"])
    parser.add_argument("--tag_names", nargs="+", default=["furniture"])
    args = parser.parse_args()

    logging.info(f"Arguments: {args}")

    if args.dataset == "objaverse":
        logging.info("Using Objaverse dataset")
        models = ObjaverseModels(
            category_names=args.category_names,
            tag_names=args.tag_names,
            dataset_folder=args.output,
        ).get_models()
    elif args.dataset == "modelnet":
        logging.info("Using ModelNet dataset")
        models = ModelNetModels(
            dataset_folder=args.source,
        ).get_models()
    elif args.dataset == "shapenet":
        logging.info("Using ShapeNet dataset")
        models = ShapeNetModels(dataset_folder=args.source).get_models()
    else:
        ext = trimesh.available_formats()
        models = []
        for e in ext:
            models.extend(glob.glob(os.path.join(args.source, f"*.{e}")))

    if args.filter_path:
        with open(args.filter_path, "r") as f:
            filter_models = f.readlines()
        filter_models = [f.strip() for f in filter_models]
        models = [m for m in models if m.split("/")[-1].split(".")[0] in filter_models]

    if args.filter_path:
        logging.info(f"Found filtered {len(models)} models")
    else:
        logging.info(f"Found {len(models)} models")
    DatasetGenerator(
        dataset_folder=args.output,
        model_paths=models,
        N=args.N,
        per_model=args.per_model,
        max_holes_per_model=args.max_holes_per_model,
        mising_volume_threshold=args.mising_volume_threshold,
        primitive_size_range=args.primitive_size_range,
        multiprocessing=args.multiprocessing,
        cpu_count=args.cpu_count,
        timeout=args.timeout,
    ).run()
