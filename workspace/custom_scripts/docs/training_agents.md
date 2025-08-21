## parameters 
- num_actors = num of parallel envs
- horizon_length = length of env roll out steps 128
- num_agents = num agents in the env default 1
- batch_size_envs = `self.horizon_length * self.num_actors`
- seq_length = length of rnn sequence 128
- total_agents = `num_agents * num_actors`
- batch_size = `horizontal_length * num_actors * num_agents`
- self.config = `config = params['config']`
- self.games_num = `config['minibatch_size'] // seq_length`  
    it is used only for current rnn implementation (episodes per minibatch) `512 // 128`

- weight_decay = 0
- num_minibatches = `batch_size // minibatch_size` 
- 


### config
- [ ] normalize_value: True
- [ ] value_bootstrap: True
- [ ] truncate_grads: True
- [x] clip_value: True
- [ ] tau: 0.95
- [ ] gamma: 0.995
- [ ] grad_norm: 1.0
- [x] critic_coef: 2
- [x] mini_epochs = 4   

--- 

- [x] lr_schedule: adaptive
- [x] schedule_type: standard
- [x] kl_threshold: 0.008
    

## variables
### training

- self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
- self.tensor_list = self.update_list + ['obses', 'states', 'dones']
- mb_rnn_states = `[torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]`
size is (1, num_)


- is_train = True for value it gives normalized values (direct output from the network ), False gives denormalized values (.denorm_values --> self.value_mean_std(input, denorm=false))
- model.train or update doesnt alter the central_value_net.training
- self.get_actions, self.get_central_value --> self.get_values --> calls value_network.get_value which sets the value_mean_std (value_mean_std.training = False) into eval so no updates of running mean and std != model.eval

- central_value_net.model.value_mean_std for values
- central_value_net.model.running_mean_std for inputs

## actor loss

```
def actor_loss(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip):
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)

    return a_loss
```


## adaptive scheduler

```
class AdaptiveScheduler(RLScheduler):
    def __init__(self, kl_threshold = 0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr, entropy_coef
```    

[a2c_common(central_value_net)] --> [central_value.py] --> [models.py (ModelCentralValue)]
[a2c_common(model)] --> [model.py(modela2ccontinuous)]

## locations

[location a2c_common.py](../../isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/common/a2c_common.py)   
[location a2c_continuous.py](../../isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/algos_torch/a2c_continuous.py)

[central_value](../../isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/algos_torch/central_value.py)

[network_builder](../../isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/algos_torch/network_builder.py)

[models.py](../../isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/algos_torch/models.py)

# note

- [x] scheduler
- [x] actor loss
- [ ] `self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)`
- [ ] `truncate_gradients_and_step()`
- [ ] `self.scalar.scale(loss).backward`

- [ ] explain what is res_dict so that i get a clear understanding of what are res_dict['values'] (also how are these res_dict['values'] calculated) which are further being passed into the critic loss calculation 
- [ ] adaptive scheduler for policy lr ;  no scheduler for value lr 
- [ ] different optimizers for polciy and critic 