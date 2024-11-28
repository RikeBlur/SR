from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision.transforms as transforms


if __name__ == '__main__' :

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        'datasets',
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 100,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True              # whether to calculate fid during training
    )

    trainer.train()

    sampled_images = diffusion.sample(batch_size = 4)

    print(sampled_images.shape) # (4, 3, 128, 128)
    
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    for i in range(0,3) :
        pic = toPIL(sampled_images[i])
        pic.save(f'random_{i}.jpg')
    
