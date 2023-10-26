# Hilbert-Schmidt Independence Criterion (HSIC)

Pytorch version of [Hilbert-Schmidt Independence Criterion](http://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf) (HSIC).

## Prerequisites
* pytorch

We tested the code using **torch 1.8.1 or torch 1.12.0 for python 3**.

## Apply on your data

### Usage

Import HSICLoss using

```
from HSICLoss import HSICLoss
```

Apply HSIC on your data
```
# define loss
HISC_Loss = HISCLoss(alpha=0.05)

# compute loss
hisc_loss = HISC_Loss(x, y)
#backward loss
hisc_loss.backward()

```

### Description
The input to definition for HISCLoss
| Argument  | Description  |
|---|---|
|alph | level of the test |


Forward call require parameters x and y.

| Argument  | Description  |
|---|---|
|x | Data of the first variable. `(n, dim_x)` numpy array.|
|y | Data of the second variable. `(n, dim_y)` numpy array.|


Output of the HISC loss to backward.

## Authors

* **Xiao Li** - mxl1990 [at] gmail [dot] com


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
