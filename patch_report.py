import sys, re

with open('scripts/summarize_results.py', 'r') as f:
    text = f.read()

if 'import matplotlib' not in text:
    text = text.replace(
        "import argparse\nimport json",
        "import argparse\nimport json\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns"
    )

with open('scripts/summarize_results.py', 'w') as f:
    f.write(text)
