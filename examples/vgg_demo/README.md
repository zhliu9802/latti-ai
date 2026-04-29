# 说明

该目录包含端到端加密推理示例。每个示例都需要一个 `task/` 文件夹，其中包含经过适配的模型权重、已编译的加密计算图以及配置文件。我们提供了预先准备好的 `task/` 文件夹（除`test_imagenet`），因此你可以直接运行这些示例；或者，你也可以按照本文件下面的[快速开始](./README.md#quick-start) 部分，从头生成这些文件。

# 快速开始

## 项目构建

参考项目根目录README文件中的[Build & Install](../../README.md#build--install)部分。

## 代码运行（以test\_cifar10为例）

每个步骤具体的作用见根目录README文件中的[Quick Start](../../README.md#quick-start)部分，下面仅列出相关命令。主要注意的是：

- 测试MNIST数据集和ImageNet数据集时，只需要将`test_cifar10`替换为`test_mnist`或`test_imagenet`即可；
- vgg.py文件中定义了VGG11、VGG13、VGG16、VGG19这4个模型，你可以根据需要选择不同的模型。

### 基线训练

```bash
python examples/vgg_demo/test_cifar10/train_vgg.py --epochs 150 --batch-size 128 --lr 0.1 --output-dir ./runs/vgg_demo/cifar10/model --input-shape 3 32 32
```

> 注意：MNIST数据集的输入形状需要为`1 32 32`，而不是`1 16 16`

运行结果：最优准确率为90.79%

```text
[140/150] lr=0.0100  train 0.0753/97.42%  test 0.3876/89.10%    12.4s
[141/150] lr=0.0100  train 0.0767/97.36%  test 0.3665/89.57%    11.8s
[142/150] lr=0.0100  train 0.0707/97.61%  test 0.3548/89.84%    12.3s
[143/150] lr=0.0100  train 0.0726/97.52%  test 0.4004/89.00%    11.8s
[144/150] lr=0.0100  train 0.0793/97.26%  test 0.3654/89.75%    12.1s
[145/150] lr=0.0100  train 0.0714/97.64%  test 0.4097/88.99%    12.2s
[146/150] lr=0.0100  train 0.0747/97.32%  test 0.4028/88.89%    11.8s
[147/150] lr=0.0100  train 0.0720/97.47%  test 0.3933/88.66%    11.8s
[148/150] lr=0.0100  train 0.0789/97.19%  test 0.4451/88.07%    12.7s
[149/150] lr=0.0100  train 0.0760/97.36%  test 0.3827/89.08%    12.3s
[150/150] lr=0.0010  train 0.0812/97.12%  test 0.4141/88.78%    12.2s
Best accuracy: 90.79%  ->  ./runs/cifar10/model/train_vgg_baseline.pth
```

### 算子替换与模型微调

```bash
python examples/vgg_demo/test_cifar10/train_vgg.py \
  --poly_model_convert \
  --pretrained ./runs/vgg_demo/cifar10/model/train_vgg_baseline.pth \
  --epochs 10 \
  --batch-size 36 \
  --lr 0.001 \
  --input-dir ./runs/vgg_demo/cifar10/model \
  --export-dir ./runs/vgg_demo/cifar10/task/server \
  --input-shape 3 32 32 \
  --degree 2 \
  --upper-bound 3.0 \
  --poly-module RangeNormPoly2d
```

运行结果：最优准确率为81.00%

```text
[  5/10] lr=0.0010  train 0.3807/87.13%  test nan/80.04% *  25.1s
[  6/10] lr=0.0010  train 0.3563/88.00%  test nan/79.45%    25.0s
[  7/10] lr=0.0010  train 0.3354/88.59%  test nan/79.39%    25.2s
[  8/10] lr=0.0010  train 0.3187/89.19%  test nan/79.82%    27.9s
[  9/10] lr=0.0010  train 0.2971/89.73%  test nan/81.00% *  24.6s
[ 10/10] lr=0.0010  train 0.2849/90.22%  test nan/80.44%    23.3s
Best accuracy: 81.00%  ->  ./runs/cifar10/model/train_poly.pth
```

### FHE编译

```bash
python training/run_compile.py \
  --input=./runs/vgg_demo/cifar10/model/trained_poly.onnx \
  --output=./runs/vgg_demo/cifar10/ \
  --style=multiplexed
```

### 生成底层指令

```bash
python inference/interface/gen_mega_ag.py --task-dir ./runs/vgg_demo/cifar10/task
```

### 使用CPU或GPU进行加密推理

```bash
./build/examples/inference --task-dir ./runs/vgg_demo/cifar10/task --input ./examples/vgg_demo/test_cifar10/task/client/img.csv --verify
./build/examples/inference --task-dir ./runs/vgg_demo/cifar10/task --input ./examples/vgg_demo/test_cifar10/task/client/img.csv --gpu --verify
```

推理结果为：

![推理结果](./images/inference_result.png)

# 问题记录

## 问题 1：`TypeError: only 0-dimensional arrays can be converted to Python scalars`

开启 `--poly_model_convert` 后，在 Hermite 系数计算阶段报错，调用链：

- [`replace_activation_with_poly()`](../../training/nn_tools/replace.py:50)
- [`get_hermite_coeffs_for_module()`](../../training/nn_tools/eval_fn_hat_for_aespa.py:166)
- [`compute_coefficients()`](../../training/nn_tools/eval_fn_hat_for_aespa.py:84)
- [`quad()`](../../training/nn_tools/eval_fn_hat_for_aespa.py:110)

**原因**：`scipy.integrate.quad` 需要“标量输入 -> 标量输出”的被积函数，但当前实现中：

- [`numpy_func()`](../../training/nn_tools/eval_fn_hat_for_aespa.py:185) 对标量输入也可能返回 shape=(1,) 的数组（见 [`return result.numpy()`](../../training/nn_tools/eval_fn_hat_for_aespa.py:189)）；
- `quad`的[lambda](../../training/nn_tools/eval_fn_hat_for_aespa.py:111) 又显式包了 `np.array(...)`，进一步导致返回 `ndarray` 而非 Python float。

### 建议修复点（`training/nn_tools/eval_fn_hat_for_aespa.py`）

1. **保证积分被积函数返回标量**（核心）
   - 在 [`compute_coefficients()`](../../training/nn_tools/eval_fn_hat_for_aespa.py:84) 中，确保 `quad` 调用的函数最终 `return float(...)`。
   - 不要在 [`quad`](../../training/nn_tools/eval_fn_hat_for_aespa.py:110) [调用处](../../training/nn_tools/eval_fn_hat_for_aespa.py:110) 使用 `np.array(...)` 包装返回值。
2. **保证** **`numpy_func`** **在标量输入时返回标量**
   - 在 [`numpy_func()`](../../training/nn_tools/eval_fn_hat_for_aespa.py:185) 内判断输入是否标量；
   - 标量输入返回 `float`；数组输入继续返回 `ndarray`。
3. **可选增强：在** **`compute_coefficients`** **做统一兜底**
   - 对 `fx` 做 `np.asarray(fx)` 后，若 `size == 1` 则转 `float`，避免传给 `quad` 的是非标量对象。

### 参考修复思路（伪代码）

```python
# training/nn_tools/eval_fn_hat_for_aespa.py

def numpy_func(x):
    is_scalar = np.isscalar(x)
    x_arr = np.array([x], dtype=np.float64) if is_scalar else np.asarray(x, dtype=np.float64)
    with torch.no_grad():
        y = module(torch.as_tensor(x_arr, dtype=torch.float64)).cpu().numpy()
    if is_scalar:
        return float(np.asarray(y).reshape(-1)[0])
    return y


def integrand_scalar(x):
    fx = func(x, **func_kwargs) if func_kwargs else func(x)
    return float(fx * hermite_prob(n, x) * np.exp(-(x**2) / 2))

I_n, _ = quad(integrand_scalar, -limit, limit, epsabs=tol, limit=1000)
```

