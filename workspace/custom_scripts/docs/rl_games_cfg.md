## RL_Games_cfg

- [Location](../../isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml)
- ### sigma 
#sigma
- Under Network  
    ***sigma_init***: Initializes sigma to 0 (which becomes 1 when exponentiated as log_std)  
    ***fixed_sigma***: False: Sigma can be learned/updated during training  
    ***sigma_activation***: None: No activation function applied to sigma
- 