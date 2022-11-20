from pathlib import Path
import random

random.seed(0)
images = Path("data/datasets/images")
train = Path("data/datasets/train.txt")
val = Path("data/datasets/val.txt")
n = len(list(images.glob('*')))

idx = random.sample(list(range(n)), int(n*0.7))
i = 0
train_set = ""
val_set = ""

for file in images.iterdir():
    if i in idx:
        train_set += f"{file.absolute()}\n"
    else:
        val_set += f"{file.absolute()}\n"
    i += 1

with open(train, 'w') as f:
    f.write(train_set)
with open(val, 'w') as f:
    f.write(val_set)
