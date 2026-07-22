import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import logging
class TabularDiffusionModel(nn.Module): #
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super(TabularDiffusionModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim + 1, hidden_dim))  # +1 for timestep
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)  # Shape: (batch_size, 1)
        x_t = torch.cat([x, t], dim=-1)  # Shape: (batch_size, input_dim + 1)
        return self.network(x_t) #the noise that is being return
class TabularDiffusion:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.target_column = self.config['data'].get('target_column', 'Outcome')
        self.device = torch.device(config['diffusion']['device'])
        self.num_timesteps = config['diffusion']['num_timesteps'] #defines the no of diffusion steps ( folds taken for syntetic data generation)
        self.beta_start = config['diffusion']['beta_start'] #(influences later noise dynamics/intial val to noise generation)
        self.beta_end = config['diffusion']['beta_end'] #timestamps-1
        
        self.hidden_dim = config['diffusion']['hidden_dim'] #layers middle
        self.num_layers = config['diffusion']['num_layers'] #count of hidden layers
        
        self.batch_size = config['diffusion']['batch_size'] #no of samples processed per training item
        self.learning_rate = config['diffusion']['learning_rate'] #instantiates model parameters ()
        self.epochs = config['diffusion']['epochs'] #defines the training iterations count(eg. 5/50,10/50...45/50)
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device) #intialise the beta schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.model = None
        self.feature_names = None
        
    def _get_beta_schedule(self) -> torch.Tensor: #returns beta tensor
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps) #no of elements = self.num_timesteps
    def _q_sample(
        self,x_start: torch.Tensor,t: torch.Tensor,noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise #noisy version of input data at the time it was used. noisy_x was produced from diffusion, and noise was implemented intially from xstart
        return noisy_x, noise
    
    def _p_losses(self,x_start: torch.Tensor,t: torch.Tensor,noise: Optional[torch.Tensor] = None) -> torch.Tensor: #loss function for the noise generated
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy, noise = self._q_sample(x_start, t, noise)
       
        predicted_noise = self.model(x_noisy, t)
        loss = nn.functional.mse_loss(predicted_noise, noise) # loss b/w the actual noise added during the diffusion process and losses predicted
        return loss
    def _p_sample(self,x: torch.Tensor,t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predicted_noise = self.model(x, t)
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            alpha_t = alpha_t.view(-1, 1)
            alpha_cumprod_t = alpha_cumprod_t.view(-1, 1)
            beta_t = beta_t.view(-1, 1)

            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)

            if t[0] > 0:
                noise = torch.randn_like(x)
                variance = beta_t
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        return x
    
    def train(self, data: np.ndarray, feature_names: list) -> None:
        self.logger.info("Starting diffusion model training")
        self.feature_names = feature_names

        data_tensor = torch.FloatTensor(data).to(self.device)

        input_dim = data.shape[1]
        self.model = TabularDiffusionModel(input_dim, self.hidden_dim, self.num_layers).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = max(1, int(np.ceil(len(data_tensor) / self.batch_size)))
            for i in range(num_batches):
                # Get batch
                current_batch_size = min(self.batch_size, len(data_tensor))
                idx = torch.randperm(len(data_tensor), device=self.device)[:current_batch_size]
                batch = data_tensor[idx]
                t = torch.randint(0, self.num_timesteps, (current_batch_size,), device=self.device)
                loss = self._p_losses(batch, t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        self.logger.info("Diffusion model training completed")
    def generate(self,num_samples: int,class_label: Optional[int] = None) -> pd.DataFrame:
        self.logger.info(f"Generating {num_samples} synthetic samples")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        x = torch.randn(num_samples, len(self.feature_names)).to(self.device)
        with torch.no_grad():
            for t in tqdm(range(self.num_timesteps - 1, -1, -1), desc="Generating samples"):
                t_tensor = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
                x = self._p_sample(x, t_tensor)
        synthetic_data = x.cpu().numpy()

        df = pd.DataFrame(synthetic_data, columns=self.feature_names)

        if class_label is not None:
            df[self.target_column] = class_label
        self.logger.info(f"Generated {num_samples} synthetic samples")
        return df
    def generate_balanced(self,num_samples_per_class: int) -> pd.DataFrame:
        self.logger.info(f"Generating {num_samples_per_class} samples per class")

        df_class_0 = self.generate(num_samples_per_class, class_label=0)
        df_class_1 = self.generate(num_samples_per_class, class_label=1)
        df_balanced = pd.concat([df_class_0, df_class_1], ignore_index=True)
        
        self.logger.info(f"Generated {len(df_balanced)} total samples")
        return df_balanced
    def save_synthetic_data(self, df: pd.DataFrame, save_path: str) -> None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Saved synthetic data to {save_path}")
