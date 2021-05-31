# I-SEE

High-Quality Diversification for Task-Oriented Dialogue Systems

We follow the code structure of [GDPL](https://github.com/truthless11/GDPL), but modified files for our needs.

### Requirements
python 3.6
```shell
pip install -r requirements.txt
```

### Pre-train dialogue policy
```shell
python main.py --pretrain --save_dir model
```

### Pre-train world models
```shell
python main.py --pretrain_world --save_dir model
```

### RL training 
##### DQN
```shell

```

##### PPO
```shell
python main_vanilla_ppo.py --process=8 --load_model=model/best --lr_rl=1e-4 --lr_irl=1e-4 --epoch=16 --ensemble_size=5 --sim_ratio=0.05 --horizon=5 --save_dir=model_rl
```
##### GDPL
```shell
python main.py --process=8 --load_model=model/best --lr_rl=1e-4 --lr_irl=1e-4 --epoch=16 --ensemble_size=5 --sim_ratio=0.2 --horizon=5 --save_dir=model_rl
```


### Evaluation



### Citation
```
@inproceedings{traj_acl_2021,
    title = "High-Quality Dialogue Diversification by Intermittent Short Extension Ensemble",
    author = "Tang, Zhiwen  and
      Kulkarni, Hrishikesh  and
      Hui Yang, Grace",
    booktitle = "Proceedings of The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021) (Findings of ACL).",
    year = "2021",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
}
```