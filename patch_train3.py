import sys, re

with open('src/mech390/ml/train.py', 'r') as f:
    text = f.read()

# Make sure train_loss_epoch logic exists
if text.find("train_loss_epoch = 0.0") == -1:
    text = text.replace(
        "for epoch in range(max_epochs):\n        # --- Train ---\n        model.train()",
        "for epoch in range(max_epochs):\n        train_loss_epoch = 0.0\n        batches = 0\n        # --- Train ---\n        model.train()"
    )
    text = text.replace(
        "loss.backward()\n            optimizer.step()",
        "loss.backward()\n            optimizer.step()\n            train_loss_epoch += loss.item()\n            batches += 1"
    )
    text = text.replace(
        "        # --- Validate ---",
        "        train_loss_epoch /= max(1, batches)\n        # --- Validate ---"
    )

with open('src/mech390/ml/train.py', 'w') as f:
    f.write(text)
