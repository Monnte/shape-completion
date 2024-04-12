"""
Code taken from https://github.com/openai/improved-diffusion, and modifed by Peter Zdravecký.
"""

import argparse
import trimesh
import mesh2sdf
import os
import skimage

import numpy as np
import torch as th

from model import logger, utils
from model.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    diffusion_sample_defaults,
    model_defautls,
    create_gaussian_diffusion,
    create_model,
)

def load_model(model_path):
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

def convert_mesh_to_tsdf(mesh, N):
    level = 2/N
    sdf = mesh2sdf.compute(
        mesh.vertices, mesh.faces, N, fix=True, level=level
    )
    return sdf


def tsdf_to_mesh(tsdf):
    tsdf = np.clip(tsdf, -1, 1)
    tsdf = np.pad(tsdf, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1)
    vertices, faces, _, _ = skimage.measure.marching_cubes(tsdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.log_dir)
    
    logger.log(f"Arguments: {args}")


    logger.log("creating model...")
    model = create_model(**args_to_dict(args, model_defautls().keys()))
    diffusion = create_gaussian_diffusion(**diffusion_sample_defaults())
    model.load_state_dict(
        utils.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(utils.dev())
    model.eval()

    logger.log("loading data...")

    logger.log("creating samples...")
    
    if args.input_mesh:
        mesh = load_model(args.sample_path)
        cond = convert_mesh_to_tsdf(mesh, args.condition_size)
    else:
        cond = np.load(os.path.join(args.sample_path))

    cond = th.from_numpy(cond).to(utils.dev()).unsqueeze(0)
    
    if args.use_roi:
        roi = np.load(os.path.join(args.roi_path))
        roi = th.from_numpy(roi).to(utils.dev()).unsqueeze(0)
        cond = th.cat((cond,roi),0)
        
    cond = cond.unsqueeze(0)
    model_kwargs = {
        "cond":cond,
    }

    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    dim = cond.shape[-1]
    if args.output_size != dim:
        dim = args.output_size
    
    sample = sample_fn(
        model,
        (1,1,dim,dim,dim),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        progress=True,
    )
    sample = sample.squeeze(0).squeeze(0).cpu().numpy()

    logger.log("saving sample...")
    sample_name = args.sample_path.split("/")[-1].split(".")[0]
    np.save(f"{args.log_dir}/{sample_name}_sample.npy", sample)
    
    mesh = tsdf_to_mesh(sample)
    mesh.export(f"{args.log_dir}/{sample_name}_sample.obj")
    
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        model_path="",
        input_mesh=False,
        sample_path="",
        log_dir="logs/sample",
        use_roi=False,
        roi_path="",
        super_res=False,
        output_size=32,
        condition_size=32,
        in_scale_factor=0,
    )
    defaults.update(diffusion_sample_defaults())
    defaults.update(model_defautls())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
