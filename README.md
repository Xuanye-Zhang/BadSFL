# BadSFL: Backdoor Attack against Scaffold Federated Learning

This code is the implementation of a NTU Final Year Project [BadSFL](https://hdl.handle.net/10356/174843). The source code is mostly based on [ScaffoldFL] (https://github.com/ongzh/ScaffoldFL) and
"[How To Backdoor Federated Learning (AISTATS'20)](https://arxiv.org/abs/1807.00459)".

### Dependencies

- PyTorch => 1.10.*
- torchvision >= 0.11.*
- numpy >= 1.23.0
- tqdm >= 4.46.0

### USAGE
Install all dependencies
```
pip install -r requirements.txt
```

To run the code, can use existing half-trained model (save in *./saved_models/* folder) or train a new new model (select resume=1/0)


---


Scaffold Federated Learning CIFAR-10 without attack
```
python scaffold_backdoor.py --epochs=100 --atks=-2 --atke=-1 --num_users=20 --gpu=0 --dataset=cifar --frac=0.5 --local_bs=128 --local_ep=10 --lr=0.001 --iid=0 --resume=1 --BadSFL=0 --comment=noattack
```

Scaffold Federated Learning CIFAR-10 baseline
```
python scaffold_backdoor.py --epochs=100 --atks=10 --atke=40 --num_users=20 --gpu=0 --dataset=cifar --frac=0.5 --local_bs=128 --local_ep=10 --lr=0.001 --iid=0 --resume=1 --BadSFL=0 --comment=cifarbaseline
```

Scaffold Federated Learning CIFAR-10 BadSFL
```
python scaffold_backdoor.py --epochs=100 --atks=10 --atke=40 --num_users=20 --gpu=0 --dataset=cifar --frac=0.5 --local_bs=128 --local_ep=10 --lr=0.001 --iid=0 --resume=1 --BadSFL=1 --comment=cifarbadsfl
```

Scaffold Federated Learning CIFAR-10 Neurotoxin
```
python scaffold_backdoor.py --epochs=100 --atks=10 --atke=40 --num_users=20 --gpu=0 --dataset=cifar --frac=0.5 --local_bs=128 --local_ep=10 --lr=0.001 --iid=0 --resume=1 --BadSFL=2 --comment=cifarneurotoxin
```


---


Scaffold Federated Learning MNIST without attack
```
python scaffold_backdoor.py --epochs=100 --atks=-2 --atke=-1 --num_users=20 --gpu=0 --dataset=mnist --frac=0.5 --local_bs=128 --local_ep=2 --lr=0.01 --iid=0 --resume=1 --BadSFL=0 --comment=noattack
```

Scaffold Federated Learning MNIST baseline
```
python scaffold_backdoor.py --epochs=100 --atks=10 --atke=40 --num_users=20 --gpu=0 --dataset=mnist --frac=0.5 --local_bs=128 --local_ep=2 --lr=0.01 --iid=0 --resume=1 --BadSFL=0 --comment=mnistbaseline
```

Scaffold Federated Learning MNIST BadSFL
```
python scaffold_backdoor.py --epochs=100 --atks=10 --atke=40 --num_users=20 --gpu=0 --dataset=mnist --frac=0.5 --local_bs=128 --local_ep=2 --lr=0.01 --iid=0 --resume=1 --BadSFL=0 --comment=mnistbadsfl
```

Scaffold Federated Learning MNIST Neurotoxin
```
python scaffold_backdoor.py --epochs=100 --atks=10 --atke=40 --num_users=20 --gpu=0 --dataset=mnist --frac=0.5 --local_bs=128 --local_ep=2 --lr=0.01 --iid=0 --resume=1 --BadSFL=0 --comment=mnistneurotoxin
```


## Results
All the logs files save in [logs](./logs/).

You can view the results through running ```tensorboard --logdir runs```.