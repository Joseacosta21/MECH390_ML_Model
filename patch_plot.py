import sys

with open('scripts/generate_report.py', 'r') as f:
    text = f.read()

imports = """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mech390.ml import features as F
"""
if 'import matplotlib.pyplot' not in text:
    text = text.replace('import json', f'import json\n{imports}')

with open('scripts/generate_report.py', 'w') as f:
    f.write(text)

