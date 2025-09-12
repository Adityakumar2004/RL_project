
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="ppo on factory env")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if available.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### -----------------------------------------------------------
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from isaaclab_tasks.utils import parse_env_cfg
import wandb
import imageio

from utils_1 import env_wrapper, RunningNormalizer, reward_function, policy_kl, AdaptiveScheduler, bound_loss
from agents_defn import Agent


adaptive_scheduler = AdaptiveScheduler()


### -----------------------------------------------------------------

@dataclass
class Args:
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    num_envs: int = args_cli.num_envs
    """the number of parallel game environments"""
    ## this is the user input 

    total_timesteps: int = 10_00_000 #10_00_000 ------------- 
    ## this is changed in the code runtime: total_timesteps = num_updates * batch_size
    """total timesteps of the experiments"""
    
    num_updates: int = 200 # ----------------------------
    """the number of updates for the entire loop on top of the env roll out (num_steps), update_epochs"""
    ## this is the user input 

    learning_rate: float = 1.0e-4 ##---------
    """the learning rate of the optimizer"""
    num_steps: int = 128 #256 #16 
    """the number of steps to run in each environment per policy rollout"""
    ## this is the user input 

    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gae: bool = True
    """Use GAE for advantage computation"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32 #64 #4 #16 -------------
    """the number of mini-batches"""
    ## this is the user input 

    update_epochs: int = 4 #15 ---------------mini_epochs
    """the K epochs to update the policy"""
    ## this is the user input

    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2 #0.1#0.2 -------------
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0001#0.0
    """coefficient of the entropy"""
    vf_coef: float = 2 #0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1
    """the maximum norm for the gradient clipping"""
    target_kl: float =0.008 # None-------------
    """the target KL divergence threshold"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    ## batch_size  =  num_steps * num_envs

    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    norm_value: bool = True  # Add this flag to control value normalization

    adaptive_scheduler: bool = True
    """Toggle adaptive scheduler"""

    bound_loss_coef: float = 0.0001
    """coefficient of the boundary loss"""
    

def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.factory.factory_env_diff_ik:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"custom_scripts.factory.factory_env_cfg_diff_ik:FactoryTaskPegInsertCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
     
    env = env_wrapper(env, video_folder, output_type=output_type, enable_normalization_rewards=False)
    
    return env



if __name__ == "__main__":

    ## start the env 
    # video_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "videos_lstm_1")
    exp_name = "diff_ik_2"
    checkpoint_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "checkpoints")
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, f"{exp_name}.pt")
    
    args = Args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    args.batch_size = int(args.num_envs * args.num_steps) # 64 * 256
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 64*256 //32 = 512
    # args.num_updates = args.total_timesteps // args.batch_size # 10_00_000 // 64*256 = 625
    args.total_timesteps = args.num_updates * args.batch_size


    # envs = make_env(video_folder, output_type="torch")
    envs = make_env(output_type="torch")
    envs.train()  # set the env to training mode

    agent = Agent(envs, args.learning_rate, args.learning_rate, norm_value = True, value_size = 1).to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    tracking_enabled = True
    if tracking_enabled:
        wandb.init(
            project = "Space_RL",
            name = f"{exp_name}_{int(time.time())}"
            
        )  

## init tensors

    obs = {"policy": torch.zeros((args.num_steps, args_cli.num_envs, envs.total_obs_space["policy"].shape[-1]), device=device),
           "critic": torch.zeros((args.num_steps, args_cli.num_envs, envs.total_obs_space["critic"].shape[-1]), device=device)}
    
    actions = torch.zeros((args.num_steps, args_cli.num_envs, envs.action_space.shape[-1]), device=device)
    log_probs = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    values = torch.zeros((args.num_steps, args_cli.num_envs), device=device)    
    a_mus = torch.zeros((args.num_steps, args_cli.num_envs, envs.action_space.shape[-1]), device=device)
    a_stds = torch.zeros((args.num_steps, args_cli.num_envs, envs.action_space.shape[-1]), device=device)
## ---------------------
    global_step = 0

    start_time = time.time()


    # --- Resume logic ---
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    raw_episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    all_returns = []
    all_raw_returns = []
    all_lengths = []
    # Add success rate tracking
    all_success_rates = []
    all_success_times = []
    start_update = 1
    if args_cli.resume:
        import os
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.load_state_dict(checkpoint["agent"])
            agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            agent.critic_normalizer.load_state_dict(checkpoint["critic_normalizer"])

            # restoring the normalizers
            if hasattr(envs, 'normalizers'):
                for k, state in checkpoint['normalizer_state'].items():
                    if k in envs.normalizers:
                        envs.normalizers[k].mean = state['mean']
                        envs.normalizers[k].var = state['var']
                        envs.normalizers[k].count = state['count']
                        envs.normalizers[k].clip_range = state['clip_range']

            # Restore learning rate
            agent.actor_optimizer.param_groups[0]["lr"] = checkpoint.get("actor_learning_rate", args.learning_rate)
            agent.critic_optimizer.param_groups[0]["lr"] = checkpoint.get("critic_learning_rate", args.learning_rate)
            global_step = checkpoint.get("global_step", 0)
            start_update = checkpoint.get("update", 1) + 1
            all_returns = checkpoint.get("all_returns", [])
            all_lengths = checkpoint.get("all_lengths", [])
            all_success_rates = checkpoint.get("all_success_rates", [])
            all_success_times = checkpoint.get("all_success_times", [])
            print(f"[INFO] Resumed training from checkpoint at update {start_update-1}, global_step {global_step}.")
        else:
            print(f"[WARNING] --resume flag set but checkpoint not found at {checkpoint_path}. Starting from scratch.")

    num_updates = args.total_timesteps // args.batch_size
    # initial_lstm_state_actor = (
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # )

    # initial_lstm_state_critic = (
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # )

    next_lstm_state_actor = (
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    )

    next_lstm_state_critic = (
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    )

    envs.train()
    next_obs, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    
    true_kl_dist = None

    for update in range(start_update, num_updates + 1):

        # envs.train()
        # next_obs, _ = envs.reset()
        # next_lstm_state_actor = (initial_lstm_state_actor[0].clone(), initial_lstm_state_actor[1].clone())
        # next_lstm_state_critic = (initial_lstm_state_critic[0].clone(), initial_lstm_state_critic[1].clone())

        # next_done = torch.zeros(args.num_envs).to(device)
        
        initial_lstm_state_actor = (next_lstm_state_actor[0].clone(), next_lstm_state_actor[1].clone())
        initial_lstm_state_critic = (next_lstm_state_critic[0].clone(), next_lstm_state_critic[1].clone())


        
        ## doubt about initial_lstm_state and next_lstm_state

        # if args.anneal_lr:
        #     if args.adaptive_scheduler:
        #         if true_kl_dist is not None:
        #             lrnow = adaptive_scheduler.update(optimizer.param_groups[0]["lr"], true_kl_dist)
        #             # optimizer.param_groups[0]["lr"] = lrnow
                    
        #             for param_group in optimizer.param_groups:
        #                 param_group["lr"] = lrnow
        #             print(f"Adaptive scheduler: Learning rate updated to {lrnow}")
        #     else:
        #         frac = 1.0 - (update - 1.0) / num_updates
        #         lrnow = frac * args.learning_rate
        #         optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args_cli.num_envs
           
            obs["policy"][step] = next_obs["policy"]
            obs["critic"][step] = next_obs["critic"]

            dones[step] = next_done

            
            with torch.no_grad():
                action, log_prob, _, next_lstm_state_actor, a_mu, a_std = agent.get_action(next_obs, next_lstm_state_actor, next_done)
                value, next_lstm_state_critic = agent.get_value(next_obs, next_lstm_state_critic, next_done, denorm=True, update_running_mean=False)

                values[step] = value.flatten()        

            actions[step] = action
            log_probs[step] = log_prob
            a_mus[step] = a_mu
            a_stds[step] = a_std

            next_obs, reward, terminated, truncated, info_custom = envs.step(action)
            next_done = (terminated | truncated).float()
            rewards[step] = reward

            ### for calculating the episodic returns --------------------------------
            raw_reward = info_custom.get('org_reward', None)
            reward_np = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
            done_np = (terminated | truncated).cpu().numpy() if isinstance(terminated, torch.Tensor) else (terminated | truncated)

            episode_returns += reward_np
            raw_episode_returns += raw_reward
            episode_lengths += 1

            # if step % 10 == 0 and hasattr(envs, 'normalizers'):
            #     norm = envs.normalizers["rewards"]
            #     print(f"Normalizer mean: {norm.mean}, var: {norm.var}, count: {norm.count}")
            #     print(f"Raw reward: {info_custom.get('org_reward', None)}")
            #     print(f"Normalized reward: {reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward}")
           
            # print(done_np)

            if np.any(done_np):
                all_returns.extend(episode_returns[done_np == 1])
                all_raw_returns.extend(raw_episode_returns[done_np == 1])
                all_lengths.extend(episode_lengths[done_np == 1])
                
                # Collect success rates for completed episodes
                if 'success_rate' in info_custom:
                    success_rate = info_custom['success_rate']
                    if isinstance(success_rate, torch.Tensor):
                        success_rate = success_rate.cpu().numpy()
                    # Add the success rate for each completed episode
                    for _ in range(done_np.sum()):
                        all_success_rates.append(success_rate)
                
                if 'success_times' in info_custom:
                    success_times = info_custom['success_times']
                    if isinstance(success_times, torch.Tensor):
                        success_times = success_times.cpu().numpy()
                    # Add the success time for each completed episode
                    for _ in range(done_np.sum()):
                        all_success_times.append(success_times)
                
                episode_returns[done_np == 1] = 0
                raw_episode_returns[done_np == 1] = 0
                episode_lengths[done_np == 1] = 0
                # print("here 1")


        # bootstrap value if not done
        with torch.no_grad():
            ## doubt need to verify this segment
            next_value, _ = agent.get_value(
                next_obs,
                next_lstm_state_critic,
                next_done,
                denorm = True,
                update_running_mean = False
            )
            next_value = next_value.reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values


        b_obs = {
            "policy": obs["policy"].reshape(-1, envs.total_obs_space["policy"].shape[-1]),
            "critic": obs["critic"].reshape(-1, envs.total_obs_space["critic"].shape[-1]),
        }
        
        b_logprobs = log_probs.reshape(-1)
        b_a_mus = a_mus.reshape(-1, envs.action_space.shape[-1])
        b_a_stds = a_stds.reshape(-1, envs.action_space.shape[-1])
        b_actions = actions.reshape(-1, envs.action_space.shape[-1])
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        ## doubt about the sequence order of the data .ravel, .reshape,
        ## optmizing the policy and value nn
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches # 64 // 32 = 2
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []


        # print("b dones shape:", b_dones.shape, "b_dones dtype :", b_dones.dtype, "b_dones device:", b_dones.device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                mb_inds = torch.as_tensor(mb_inds, dtype=torch.long, device=device)

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]
                # --- Value normalization (per minibatch) ---
                if args.norm_value:
                    # mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
                    agent.critic_normalizer.train()
                    mb_values = agent.critic_normalizer(mb_values, denorm = False)
                    mb_returns = agent.critic_normalizer(mb_returns, denorm = False)
                    agent.critic_normalizer.eval()

                # print("b_actions.shape:", b_actions.shape)
                # print("mb_inds.shape:", mb_inds.shape)
                # print("mb_inds min/max:", mb_inds.min().item(), mb_inds.max().item())
                # print("b_actions length:", len(b_actions))
                assert mb_inds.max() < len(b_actions), f"mb_inds contains out-of-bounds indices! max: {mb_inds.max().item()}, len(b_actions): {len(b_actions)}"
                _, new_logprob, entropy, _, a_mu, a_std = agent.get_action(
                 mb_obs,
                 (initial_lstm_state_actor[0][:, mbenvinds], initial_lstm_state_actor[1][:, mbenvinds]),
                 b_dones[mb_inds],
                 b_actions[mb_inds]
                 )

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                newvalue, _ = agent.get_value(
                    mb_obs,
                    (initial_lstm_state_critic[0][:, mbenvinds],initial_lstm_state_critic[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    denorm = False,
                    update_running_mean = False
                )


                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                ## calculating actual KL
                with torch.no_grad():
                    true_kl_dist = policy_kl(b_a_mus[mb_inds], b_a_stds[mb_inds], a_mu, a_std)

                if args.anneal_lr:
                    if args.adaptive_scheduler:
                        if true_kl_dist is not None:
                            lrnow = adaptive_scheduler.update(agent.actor_optimizer.param_groups[0]["lr"], true_kl_dist)
                            # optimizer.param_groups[0]["lr"] = lrnow
                            
                            for param_group in agent.actor_optimizer.param_groups:
                                param_group["lr"] = lrnow
                            # print(f"Adaptive scheduler: Learning rate updated to {lrnow}")
                    else:
                        frac = 1.0 - (update - 1.0) / num_updates
                        lrnow = frac * args.learning_rate
                        optimizer.param_groups[0]["lr"] = lrnow


                # mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                ## Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                ## value loss

                newvalue = newvalue.view(-1)
                # Use mb_returns in value loss below
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    
                entropy_loss = entropy.mean()

                boundary_loss = bound_loss(a_mu)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + boundary_loss * args.bound_loss_coef

                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()

            # if args.target_kl is not None:
            #     if approx_kl > args.target_kl:
            #         break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                
        if tracking_enabled:
            wandb.log({

                "actor_learning_rate": agent.actor_optimizer.param_groups[0]["lr"],
                "critic_learning_rate": agent.critic_optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "true_kl_dist": true_kl_dist.item(),
                "explained_variance": explained_var,
                "clipfrac": np.mean(clipfracs),
                "boundary_loss": boundary_loss.item()
                }, step=global_step)


        ## inference 
        # avg_reward = TestingAgent(envs, agent, num_episodes=2, recording_enabled=envs.enable_recording)
        # checkpoint = {
        #     "agent": agent.state_dict(),
        #     "optimizer": optimizer.state_dict()
        # }

        # if tracking_enabled:
        #     wandb.log({"avg_reward": avg_reward}, step=global_step)
        # print(f"Iteration/update {update + 1}/{args.num_updates}, Global Step: {global_step}, Avg Reward: {avg_reward:.2f}, Time: {time.time() - start_time:.2f}s")
        
        ## logging the episodic returns

            

        # print("this is the length of all_returns ", len(all_returns))
        # print("this is all returns \n " ,all_returns)
        # print("--"*20)

        if len(all_returns) > 0:
            avg_return = np.mean(all_returns[-100:])
            avg_raw_return = np.mean(all_raw_returns[-100:])
            avg_length = np.mean(all_lengths[-100:])
            
            # Calculate success rate metrics
            avg_success_rate = 0.0
            avg_success_time = 0.0
            current_success_rate = 0.0
            if len(all_success_rates) > 0:
                avg_success_rate = np.mean(all_success_rates[-100:])
                current_success_rate = all_success_rates[-1] if len(all_success_rates) > 0 else 0.0
            if len(all_success_times) > 0:
                avg_success_time = np.mean(all_success_times[-100:])
            
            if tracking_enabled:
                wandb.log({
                    "avg_return": avg_return,
                    "avg_length": avg_length,
                    "avg_raw_return": avg_raw_return,
                    "avg_success_rate": avg_success_rate,
                    "avg_success_time": avg_success_time,
                    "current_success_rate": current_success_rate,
                    "total_episodes_completed": len(all_returns),
                }, step=global_step)
            print(f"Iteration/update {update}/{num_updates}, Global Step: {global_step}, Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.1f}, Success Rate: {avg_success_rate:.3f}, Current Success: {current_success_rate:.3f}, Time: {time.time() - start_time:.2f}s")

        # Save checkpoint with all necessary information for resuming training
        checkpoint = {
            "agent": agent.state_dict(),
            "actor_optimizer": agent.actor_optimizer.state_dict(),
            "critic_optimizer":agent.critic_optimizer.state_dict(),
            "critic_normalizer":agent.critic_normalizer.state_dict(),
            "actor_learning_rate": agent.actor_optimizer.param_groups[0]["lr"],
            "critic_learning_rate": agent.critic_optimizer.param_groups[0]["lr"],
            "global_step": global_step,
            "update": update,
            "all_returns": all_returns,
            "all_lengths": all_lengths,
            "all_success_rates": all_success_rates,
            "all_success_times": all_success_times,
            # Add any other stateful variables you want to resume
        }

        normalizer_state = {}
        if hasattr(envs, 'normalizers'):
            for k, norm in envs.normalizers.items():
                normalizer_state[k] = {
                    'mean': norm.mean,
                    'var': norm.var,
                    'count': norm.count,
                    'clip_range': norm.clip_range,
                }
        checkpoint['normalizer_state'] = normalizer_state

        torch.save(checkpoint, checkpoint_path)



    envs.close()
    wandb.finish()





        








        
