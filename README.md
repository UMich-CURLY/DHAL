# Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding


**Authors**: Hang Liu, Sangli Teng, Ben Liu, Wei Zhang, Maani Ghaffari  
**Website**: https://umich-curly.github.io/DHAL/  
**Paper**: https://arxiv.org/pdf/2503.01842   
**Contact**:  hangliu@umich.edu


### Install


1. Create environment and install torch

   ```bash
   conda create -n dhal python=3.8 
   pip3 install torch torchvision torchaudio 
   ```

   

2. Install Isaac Gym preview 4 release https://developer.nvidia.com/isaac-gym

   unzip files to a folder, then install with pip:

   `cd isaacgym/python && pip install -e .`


3. Clone our repo and install
    
    ```bash
    git clone git@github.com:UMich-CURLY/DHAL.git
    cd DHAL/legged_gym
    pip install -e .
    cd ../rsl_rl
    pip install -e .
    ```


### Training

- go to `legged_gym/legged_gym/scripts`

    ```
    python train.py --exptid=dhal
    ```

### Play

- go to `legged_gym/legged_gym`
  
    ```
    python play.py --exptid=dhal
    ```

### Arguments
- --exptid: string, can be `WHATEVER` and the weights would be saved in corresponding directory 
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume training from previous checkpoint, need to use with `--exptid`.
- --wandb: with wandb logging.
- --debug: debug mode, less agents.

## Acknowledgement
Our code are based on these previous outstanding repo:
- https://github.com/leggedrobotics/rsl_rl  
- https://github.com/leggedrobotics/legged_gym  
- https://github.com/chengxuxin/extreme-parkour  
- https://github.com/MarkFzp/Deep-Whole-Body-Control

## Citation
If our work does help you, please consider citing us and the following works:
```bibtex
@misc{liu2025discretetimehybridautomatalearning,
      title={Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding}, 
      author={Hang Liu and Sangli Teng and Ben Liu and Wei Zhang and Maani Ghaffari},
      year={2025},
      eprint={2503.01842},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.01842}, 
}
```
We used codes in [Legged Gym](training/legged_gym) and [RSL RL](training/rsl_rl), based on the paper:
  + Rudin, Nikita, et al. "Learning to walk in minutes using massively parallel deep reinforcement learning." CoRL 2022.

