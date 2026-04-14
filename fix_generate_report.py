import re

with open('scripts/generate_report.py', 'r') as f:
    text = f.read()

# I will find the first occurrence of `def generate_learning_curves():`
# and the first occurrence of `def generate_text_report():` and replace everything between them.
idx_start = text.find('def generate_learning_curves():')
idx_end = text.rfind('def generate_text_report():')

if idx_start != -1 and idx_end != -1:
    text = text[:idx_start] + text[idx_end:]

with open('scripts/generate_report.py', 'w') as f:
    f.write(text)
