## Brief overview of train.py
- [location](../../isaaclab/scripts/reinforcement_learning/rl_games/train.py)
- RlGamesVecEnvWrapper(env, rldevice, ... )

- [runner](#runner)

    ```
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)

    ```

    ```
        if args_cli.checkpoint is not None:
            runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
        else:
            runner.run({"train": True, "play": False, "sigma": train_sigma})
    ```

### Runner
- [location](../../isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/torch_runner.py)
   
- ```def _override_sigma(agent, args):```
  - [sigma docs](rl_games_cfg.md#sigma)
  - just overwrites the sigma if the sigma value is provided in args and Fixed_sigma is True

- [Builder](#Builder) `__init__` creates two instances of this builder
- [training and playing](#run) `def run(self, args):` args : 'train' and 'play'

#### Builder
- [location](../../isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/common/object_factory.py)  

- `a2c_continuous` builder:   
**Function**: ```lambda **kwargs : a2c_continuous.A2CAgent(**kwargs)```  
**Creates**: An instance of A2CAgent for continuous action spaces   
**Used for**: PPO, A2C with continuous actions

- In this file(runner) we have two instances `algo_factory` and `player_factory` #builder
- `algo_factory` for training and  `player_factory` for inference mode i.e., testing 
  ```
  self.algo_factory = object_factory.ObjectFactory()
  self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))     
  ```
  ```
  self.player_factory = object_factory.ObjectFactory()
  self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
  ```

- `def load_config(self, params):` takes the algo_name
    ```
    self.algo_params = params['algo']
    self.algo_name = self.algo_params['name']   
    ```
    > note this function also loads the params like multiple gpu, seed

#### run

- `def run(self, args):`
- arg 'train' for `run_train` and 'play' for `run_play`
- `run_train` creates agent using builder.create  
   and then `agent.train()`
- `run_play` creates player using builder.create  
   and then `player.run()`


#### a2c_continous and players
- [builder](#builder) uses them to create the agents and is later trained or played in [run](#run)
- [a2c_continuous](#training_agents.md)
- [refer for players]()