import sys, re

# 1. Update train.py
with open('src/mech390/ml/train.py', 'r') as f:
    train_content = f.read()

train_content = train_content.replace(
    "def _train_one_trial(",
    "global_history = []\ndef _train_one_trial("
)
train_content = train_content.replace(
    "device:      torch.device,",
    "device:      torch.device,\ntrial_id: int=0,"
)
train_content = train_content.replace(
    "patience   = int(cfg['training']['patience'])\n",
    "patience   = int(cfg['training']['patience'])\n    trial_history = []\n"
)
train_content = train_content.replace(
    "best_epoch    = 0",
    "best_epoch    = 0\n    train_loss = 0.0"
)

with open('src/mech390/ml/train.py', 'w') as f:
    f.write(train_content)

print('Patched files')
