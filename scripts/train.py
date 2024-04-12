"""
Code taken from https://github.com/openai/improved-diffusion, and modifed by Peter Zdravecký.
"""
import argparse

from model import logger, utils
from model.datasets import load_data,load_data_loader
from model.resample import create_named_schedule_sampler
from model.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    diffusion_sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from model.train_util import TrainLoop
from evaluation.evaluation import Evaluation

def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.log_dir)

    logger.log(f"Arguments: {args}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    validation_diffusion = create_gaussian_diffusion(**diffusion_sample_defaults())
    model.to(utils.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    val_data_path = args.val_data_path if args.val_data_path else args.data_path
    
    args.missing_volumes = [x for x in args.missing_volumes.split(",")]
    data = load_data(
        data_path=args.data_path,
        file_path=args.train_file_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        missing_volumes=args.missing_volumes,
        use_roi=args.use_roi,
    )
    
    val_data_loader = load_data_loader(
        data_path=val_data_path,
        file_path=args.val_file_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        missing_volumes=args.missing_volumes,
        use_roi=args.use_roi,
    )

    evaluation = Evaluation()
        

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        evaluation=evaluation,
        data=data,
        val_data=val_data_loader,
        val_diffusion=validation_diffusion,
        val_ddim=args.val_ddim,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_path="",
        val_data_path="",
        train_file_path="",
        val_file_path="",
        dataset_name="complete",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        val_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir="logs/train",
        missing_volumes="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        use_roi=False,
        val_ddim=True,
        super_res=False,
        in_scale_factor=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
