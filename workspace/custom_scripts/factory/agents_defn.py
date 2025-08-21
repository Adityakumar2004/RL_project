import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal



class torchRunningnormalizer(nn.Module):
    def __init__(self, insize, epsilon=1e-5):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        
        in_size = insize
        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, denorm = False, mask=None):

        if self.training:
            mean = input.mean(0)
            var = input.var(0)

            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )


        current_mean = self.running_mean
        current_var = self.running_var

        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-5.0, max=5.0)  

        return y         


class LSTMwithDones(nn.Module):
    """ lstm that handles done flags for rl games"""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.device = next(self.parameters()).device 
            
    def forward(self, inputs, lstm_state, done):
        """
        lstm_state : 2--> cell state and hidden state (num_layers, batch_size, hidden_size)
        done : (seq_len, batch_size)
        input: (seq_len, batch_size, input_size)
        """
        # done = done.to(dtype=inputs.dtype, device= self.device)
        new_hidden = []

        for x, d in zip(inputs, done):
            # print("x shape:", x.shape, "d shape:", d.shape)
            # print("d dtype:", d.dtype, "d device:", d.device, "d min/max:", d.min().item(), d.max().item())
            # assert torch.all((d == 0) | (d == 1)), f"d contains values other than 0 or 1: {d}"
            h, lstm_state = self.lstm(
                x.unsqueeze(0), ## shape: (1, B, input_size)
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h] ## h shape: (1, B, 1024=> hidden_size)

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1) ## shape: (T × B, hidden_size)
        return new_hidden, lstm_state

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, envs, hidden_size = 1024, num_layers = 2):
        super().__init__()
        self.envs = envs
        self.lstm = LSTMwithDones(envs.total_obs_space["policy"].shape[-1], hidden_size, num_layers)

        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.mlp_after_lstm = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ELU(),  # From RL-Games config: activation: elu
            layer_init(nn.Linear(512, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
        )

        self.actor_mean = layer_init(nn.Linear(64, envs.action_space.shape[-1]), std=0.01)
        self.actor_logstd = layer_init(nn.Linear(64, envs.action_space.shape[-1]), std=0.01)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)    

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2    

    def get_states(self, x, lstm_state, done):
        ## x shape: (seq_len × B, total_obs_space["policy"].shape[-1])
        ## lstm_state shape: 2 (num_layers, B, hidden_size)
        ## done shape: (seq_len × B,)

        batch_size = lstm_state[0].shape[1]  ## batch_size = B = num_envs
        
        x = x.reshape((-1, batch_size, self.envs.total_obs_space["policy"].shape[-1])) ## shape: (seq_len, B, input_size=total_obs_space["policy"].shape[-1])

        done = done.reshape((-1, batch_size)) ## shape: (seq_len, B)
        
        new_hidden, new_lstm_state = self.lstm(x, lstm_state, done)

        ## new_hidden shape: (seq_len × B, hidden_size)
        ## new_lstm_state shape: 2 (num_layers, B, hidden_size)

        return new_hidden, new_lstm_state 
        
    def get_action(self, x, lstm_state, done, action=None):
        """
        x : (seq_len * B, total_obs_space["policy"].shape[-1])
        lstm_state : 2 (num_layers, B, hidden_size)
        done : (seq_len * B,)
        action : (seq_len * B, ) or None if we want to sample an action

        """

        hidden, new_lstm_state = self.get_states(x, lstm_state, done) 
        hidden = self.layer_norm(hidden)
        
        ## hidden shape: (seq_len × B, hidden_size)

        mlp_output = self.mlp_after_lstm(hidden)
        ## mlp_output shape: (seq_len × B, 64)

        action_mean = self.actor_mean(mlp_output)
        ## action_mean shape: (seq_len × B, action_space.shape[-1])

        action_logstd = self.actor_logstd(mlp_output)
        action_logstd = torch.clamp(action_logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = action_logstd.exp()
        ## std shape: (seq_len × B, action_space.shape[-1])

        dist = Normal(action_mean, std)
        if action is None:
            action = dist.sample()
        
        a_mu = action_mean.detach().clone()
        a_std = std.detach().clone()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), new_lstm_state, a_mu, a_std


class critic(nn.Module):
    def __init__(self, envs, hidden_size = 1024, num_layers = 2):
        super().__init__()
        self.envs = envs
        self.lstm = LSTMwithDones(envs.total_obs_space["critic"].shape[-1], hidden_size, num_layers)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.mlp_after_lstm = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
        )
        
        self.critic_value = layer_init(nn.Linear(64, 1), std=1.0)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
       
    def get_states(self, x, lstm_state, done):
        ## x shape: (seq_len × B, total_obs_space["critic"].shape[-1])
        ## lstm_state shape: 2 (num_layers, B, hidden_size)
        ## done shape: (seq_len × B,)

        batch_size = lstm_state[0].shape[1]  ## batch_size = B = num_envs
        
        x = x.reshape((-1, batch_size, self.envs.total_obs_space["critic"].shape[-1])) 
        ## x shape: (seq_len, B, input_size=total_obs_space["critic"].shape[-1])

        done = done.reshape((-1, batch_size)) ## shape: (seq_len, B)
        
        new_hidden, new_lstm_state = self.lstm(x, lstm_state, done)

        ## new_hidden shape: (seq_len × B, hidden_size)
        ## new_lstm_state shape: 2 (num_layers, B, hidden_size)

        return new_hidden, new_lstm_state 

    def get_value(self, x, lstm_state, done):
        """
        x : (seq_len * B, total_obs_space["critic"].shape[-1])
        lstm_state : 2 (num_layers, B, hidden_size)
        done : (seq_len * B,)
        
        """
        hidden, new_lstm_state = self.get_states(x, lstm_state, done)
        hidden = self.layer_norm(hidden)
        ## hidden shape: (seq_len × B, hidden_size)

        mlp_output = self.mlp_after_lstm(hidden)
        ## mlp_output shape: (seq_len × B, 64)

        value = self.critic_value(mlp_output)
        ## value shape: (seq_len × B, 1)

        return value, new_lstm_state
        

class Agent(nn.Module):
    def __init__(self, envs, actor_lr=None, critic_lr=None, norm_value = True, value_size = 1, eval=False):
        super().__init__()

        self.hidden_size = 1024
        self.num_layers = 2


        self.actor = Actor(envs, self.hidden_size, self.num_layers)
        self.critic = critic(envs, self.hidden_size, self.num_layers)

        if not eval:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.norm_value = norm_value
        if norm_value:
            self.critic_normalizer = torchRunningnormalizer(value_size)

    def get_action(self, x, lstm_state, done, action=None):

        x = x["policy"]
        action, log_prob, entropy, new_lstm_state, a_mu, a_std = self.actor.get_action(x, lstm_state, done, action)

        return action, log_prob, entropy, new_lstm_state, a_mu, a_std
    
    def get_value(self, x, lstm_state, done, denorm = True, update_running_mean = False):

        x = x["critic"]
        value, new_lstm_state = self.critic.get_value(x, lstm_state, done)
        


        if denorm:
            if self.norm_value:
                if update_running_mean:
                    self.critic_normalizer.train()
                    value = self.critic_normalizer(value, denorm = True)
                    self.critic_normalizer.eval()
                
                else:
                    self.critic_normalizer.eval()
                    value = self.critic_normalizer(value, denorm = True)
            
                
                
    

        return value, new_lstm_state
    

