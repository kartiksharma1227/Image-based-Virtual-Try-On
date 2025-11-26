import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img
from logging_config import setup_inference_logger, log_exception, log_gpu_status

# Initialize logger
logger = setup_inference_logger()
logger.info("="*70)
logger.info("Inference Script Started")
logger.info("="*70)

def build_args():
    logger.debug("Parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./DATA/zalando-hd-resized")
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()
    
    logger.info(f"Configuration:")
    logger.info(f"  Config path: {args.config_path}")
    logger.info(f"  Model path: {args.model_load_path}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Data root: {args.data_root_dir}")
    logger.info(f"  Unpaired: {args.unpair}")
    logger.info(f"  Repaint: {args.repaint}")
    logger.info(f"  Denoise steps: {args.denoise_steps}")
    logger.info(f"  Image size: {args.img_H}x{args.img_W}")
    
    return args


@torch.no_grad()
def main(args):
    logger.info("Starting main inference function")
    
    # Clear GPU cache at the start
    if torch.cuda.is_available():
        logger.info("GPU is available")
        log_gpu_status(logger)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("GPU cache cleared at start")
        log_gpu_status(logger)
    else:
        logger.warning("No GPU available - running on CPU")
    
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    logger.info("Loading configuration...")
    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params
    logger.info(f"Configuration loaded: {args.config_path}")

    logger.info("Creating model...")
    model = create_model(config_path=None, config=config)
    
    logger.info(f"Loading model checkpoint: {args.model_load_path}")
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    sd = load_cp.get("state_dict", load_cp)

    # Drop HF vision encoder leftovers (e.g., position_ids) and other harmless extras
    sd = {k: v for k, v in sd.items() if "position_ids" not in k}

    logger.info("Loading state dict into model...")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        logger.warning(f"State dict loading: missing={len(missing)} keys, unexpected={len(unexpected)} keys")
        if missing:
            logger.debug(f"Missing keys: {missing[:5]}...")  # First 5
        if unexpected:
            logger.debug(f"Unexpected keys: {unexpected[:5]}...")  # First 5
    else:
        logger.info("State dict loaded successfully (all keys matched)")

    logger.info("Moving model to CUDA...")
    model = model.cuda()
    model.eval()
    logger.info("Model ready for inference")
    log_gpu_status(logger)

    logger.info("Creating PLMS sampler...")
    sampler = PLMSSampler(model)
    logger.info("Sampler created")
    
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        is_paired=not args.unpair,
        is_test=True,
        is_sorted=True
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    dataloader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=batch_size, pin_memory=False)
    logger.info(f"DataLoader created: {len(dataloader)} batches")

    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")
    
    logger.info("Starting batch processing...")
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx+1}/{len(dataloader)}")
        logger.debug(f"Batch size: {len(batch['img_fn'])}")
        
        print(f"{batch_idx}/{len(dataloader)}")
        z, c = model.get_input(batch, params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        logger.debug("Creating start code with q_sample...")
        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = model.q_sample(z, ts)     

        logger.info(f"Running PLMS sampler ({args.denoise_steps} steps)...")
        samples, _, _ = sampler.sample(
            args.denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_conditioning=uc_full,
        )
        logger.info("Sampling completed")

        logger.debug("Decoding samples from latent space...")
        x_samples = model.decode_first_stage(samples)
        logger.debug("Decoding completed")
        
        # Move samples to CPU and clear cache to free GPU memory
        samples = samples.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared after decoding")
        
        logger.info(f"Saving {len(x_samples)} output images...")
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if args.repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)

            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])
            logger.debug(f"Saved output: {to_path}")
        
        logger.info(f"Batch {batch_idx+1} completed")
        
        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared after batch")
            if (batch_idx + 1) % 5 == 0:  # Log GPU status every 5 batches
                log_gpu_status(logger)
    
    logger.info("All batches processed successfully")
    
    # Final GPU cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("Final GPU cache clear completed")
        log_gpu_status(logger)
    
    logger.info("="*70)
    logger.info("Inference completed successfully!")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        args = build_args()
        main(args)
    except Exception as e:
        log_exception(logger, e, "Fatal error in inference script")
        raise
