# Installation
Don't remember the steps but i got it working with a conda env with python `3.13`.
Torch version is `2.7.1+cu118`.

For custom training i added few packages
```requirements
pandas
opencv-python
matplotlib
scipy
tensorboard
```

# Training
They do not provide the training script but a training which performs a single iteration in [./src/trainer.y](./src/trainer.py).
There are some [Community Scripts](https://github.com/ZHKKKe/MODNet/issues/200#issuecomment-1403243454) provided which I used to star the [own_train.py](own_train.py).

