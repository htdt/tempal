## Proximal Policy Optimization
- Python 3.7, PyTorch 1.2
- Neat, simple and efficient code
- `atari pacman` score ≈4200 after 24h training on T4 GPU 

## Start
```
pip install -r requirements.txt
tensorboard --logdir runs
python -m train cartpole
```

## Dependencies
```
git clone https://github.com/openai/baselines.git
pip install -e baselines
```
