"""
Code taken from https://github.com/openai/improved-diffusion, and modifed by Peter Zdravecký.
"""

import argparse
import os
import copy
import numpy as np
import tqdm
import torch as th
from model import logger, utils
from model.datasets import load_data_loader
from model.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    diffusion_sample_defaults,
    model_defautls,
    create_gaussian_diffusion,
    create_model,
)
import scipy
import scipy.ndimage
from evaluation.evaluation import Evaluation


def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.save_path)
    
    logger.log(f"Arguments: {args}")
    logger.log("creating model...")
    model = create_model(**args_to_dict(args, model_defautls().keys()))
    if args.model_path_sr:
        sr_dict = args_to_dict(args, model_defautls().keys())
        sr_dict["super_res"] = True
        sr_model = create_model(**sr_dict)
        sr_model.load_state_dict(
        utils.load_state_dict(args.model_path_sr, map_location="cpu")
        )
        sr_model.to(utils.dev())
        sr_model.eval()
        
    diffusion = create_gaussian_diffusion(**diffusion_sample_defaults())
    model.load_state_dict(
        utils.load_state_dict(args.model_path, map_location="cpu")
    )
    evaluation = Evaluation()
    
    model.to(utils.dev())
    model.eval()

    logger.log("loading data...")
    os.makedirs(args.save_path, exist_ok=True)
    args.missing_volumes = [x for x in args.missing_volumes.split(",")]
    loader = load_data_loader(
        data_path=args.data_path,
        file_path=args.file_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        deterministic=True,
        drop_last=False,
        missing_volumes=args.missing_volumes,
        use_roi=args.use_roi,
    )
    sample_fn = diffusion.ddim_sample_loop
    
    logger.log("creating samples...")
    
    j = 0
    eval_results = {'cd': [], 'iou': [], 'l1': []}
    with th.no_grad():
        for batch in tqdm.tqdm(loader):
            gt = batch[0].to(utils.dev())
            cond = batch[1].to(utils.dev())
            file_name = batch[2]
            gt_dim = gt.shape[-1]
            cond_dim = cond.shape[-1]
            batch_size= cond.shape[0]
            
            samples = sample_fn(
                model,
                (batch_size,1,gt_dim,gt_dim,gt_dim),
                model_kwargs={"cond": cond}, 
                clip_denoised=True, 
            )
            
            
            if args.model_path_sr:
                samples_cond = copy.deepcopy(samples)
                samples_cond_resized = []
                for i in range(samples.shape[0]):
                    samples_cond_resized.append(th.tensor(scipy.ndimage.zoom(samples[i].squeeze().cpu().numpy(), gt_dim/cond_dim)).unsqueeze(0).unsqueeze(0).to(utils.dev()))
                samples_cond_resized = th.cat(samples_cond_resized, dim=0)
                samples = sample_fn(
                    sr_model,
                    (batch_size,1,gt_dim,gt_dim,gt_dim),
                    model_kwargs={"cond": samples_cond_resized}, 
                    clip_denoised=True, 
                )

            
            for i in range(samples.shape[0]):
                sample_item = samples[i].squeeze().cpu().numpy()
                gt_item = gt[i].squeeze().cpu().numpy()
                cond_item = cond[i].squeeze().cpu().numpy()
                
                if args.model_path_sr:
                    np.savez(f"{args.save_path}/{file_name[i]}_{j}",gt=gt_item, pred=sample_item, cond=cond_item, cond_sr=samples_cond[i].squeeze().cpu().numpy())
                else:
                    np.savez(f"{args.save_path}/{file_name[i]}_{j}",gt=gt_item, pred=sample_item, cond=cond_item)
                    
                j += 1
                cd, iou, l1 = evaluation.evaluate_tsdf(gt_item, sample_item, gt_dim)
                eval_results['cd'].append(cd)
                eval_results['iou'].append(iou)
                eval_results['l1'].append(l1)
        
    # log the results
    logger.logkv("cd", np.mean(eval_results['cd']) * 100)
    logger.logkv("iou", np.mean(eval_results['iou']) * 100)
    logger.logkv("l1", np.mean(eval_results['l1']))
    logger.dumpkvs()
    
    logger.log("cd: ", np.mean(eval_results['cd']) * 100)
    logger.log("iou: ", np.mean(eval_results['iou']) * 100)
    logger.log("l1: ", np.mean(eval_results['l1']))  
    logger.log("evaluation finished")


def create_argparser():
    defaults = dict(
        data_path="",
        file_path="",
        dataset_name="complete",
        batch_size=32,
        clip_denoised=True,
        model_path="",
        model_path_sr="",
        save_path="logs/eval",
        missing_volumes="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        use_roi=False,
    )
    defaults.update(model_defautls())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
