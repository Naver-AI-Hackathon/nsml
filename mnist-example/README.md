# ladder_networks for nsml testing

## Description

ladder network example implemented with Pytorch running on NSML

## How To Run
```bash
# run a session with dataset name mnist
# available options : --mode, --batch, --lr, --epochs, --top, --iteration, --pause, --gpu
$ nsml run -d mnist 

# evaluate model
$ nsml submit SESSION_NANE ITERATION

# infer model
$ nsml infer SESSION_NANE ITERATION
```

## Reference
- https://arxiv.org/abs/1507.02672

PyTorch Implementation by Kenta Iwasaki.