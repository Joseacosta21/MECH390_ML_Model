import re

with open('src/mech390/ml/train.py', 'r') as f:
    text = f.read()

# Make sure we import json and os at the top
if 'import json' not in text:
    text = text.replace('import torch', 'import json\nimport os\nimport torch')
if 'import numpy as np' not in text:
    text = text.replace('import torch', 'import numpy as np\nimport torch')

# Collect epoch history inside _train_one_trial
# Find where train_loss is calculated
if 'train_loss =' in text and 'trial_history.append' not in text:
    pass # Needs finer Regex patching

with open('src/mech390/ml/train.py', 'w') as f:
    f.write(text)
