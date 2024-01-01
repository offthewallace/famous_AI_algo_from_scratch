import os 
import torch
import torch.nn as nn
from matplotlib import puplot as plt
from torch import optim 
from tqdm import tqdm 
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self,noise_steps=1000,beta_start=1e-4,beta_end=0.02,img_size=64,device="cuda"):
        self.noise_steps=noise_steps
        self.beta_start=beta_start
        self.beta_end=beta_end
        self.image_size=img_size
        self.device=device

        self.beta=self.prepare_noise_schedule().to(device)
        self.alpha=1-self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,self.noise_steps)
    
    def npise_images(self,x,t):
        sqrt_alpha_hat= torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_mnius_alpha_hat=torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
        eps=torch.rand_like(x)
        return sqrt_alpha_hat*x+sqrt_one_mnius_alpha_hat*eps,eps
    
    def sample_timesteps(self,n):
        return torch.randint(low=1,high=self.noise_steps,size=(n,))
    
    def sample(self,model,n):
        logging.info("Sampling %d images"%n)
        model.eval()
        with torch.no_grad():
            #t=self.sample_timesteps(n)
            x=torch.randn(n,3,self.image_size,self.image_size,device=self.device)
            for i in tqdm(range(self.noise_steps)):
                t=(torch.ones(n)*i).long().to(self.device)
                predicted_noise=model(x,t)
                alpha= self.alpha[t][:,None,None,None]
                alpha_hat=self.alpha_hat[t][:,None,None,None]
                beta=self.beta[t][:,None,None,None]
                if i>1:
                    noise= torch.randn_like(x)
                else:
                    noise=torch.zeros_like(x)

                x=1/torch.sqrt(alpha_hat)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
        model.train()
        x=(x.clap(-1,1)+1)/2
        x=(x*255).type(torch.uint8)
        return x 
    
    