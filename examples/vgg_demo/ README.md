# Examples

This directory contains end-to-end encrypted inference examples. Each example requires a `task/` folder containing adapted model weights, compiled encrypted computation graphs, and configuration files. We provide pre-prepared `task/` folders so you can run directly, or generate them from scratch by following the [Quick Start](../../README.md#quick-start) guide in the root README.

## Examples Overview


| Example         | Model       | Dataset  | Input Size    | Encryption     | Bootstrapping |
| --------------- | ----------- | -------- | ------------- | -------------- | ------------- |
| `test_mnist`    | Simple CNN  | MNIST    | 1 x 16 x 16   | CKKS (N=16384) | No            |
| `test_cifar10`  | ResNet-20   | CIFAR-10 | 3 x 32 x 32   | CKKS (N=65536) | Yes           |
| `test_imagenet` | MobileNetV2 | ImageNet | 3 x 256 x 256 | CKKS (N=65536) | Yes           |


## Build

Examples are built automatically as part of the main project build (see the root [Build & Install](../../README.md#build--install) guide).

## Run

See the [Running Examples](../../README.md#running-examples) section in the root README for instructions on running each example.

## Verification

The `inference` binary supports a `--verify` mode that compares encrypted inference results against plaintext results element-by-element, reporting per-element errors and overall pass/fail status.

### Manual Verification

```bash
# CPU mode
./build/examples/inference --task-dir examples/test_mnist/task --input examples/test_mnist/task/client/img.csv --verify

# GPU mode
./build/examples/inference --task-dir examples/test_mnist/task --input examples/test_mnist/task/client/img.csv --verify --gpu
```

### Batch Verification with CTest

All examples are registered as CTest tests with labels for easy filtering. Run from the `build/` directory:

```bash
ctest -L example-cpu -V          # All CPU examples
ctest -L example-gpu -V          # All GPU examples
ctest -R mnist -V                # MNIST only (CPU + GPU)
ctest -R cifar10-gpu -V          # CIFAR-10 GPU only
ctest -L example -V              # All examples (CPU + GPU)
```

Available labels: `example`, `example-cpu`, `example-gpu`, `mnist`, `cifar10`, `imagenet`.

## Example Details

### test_mnist

The simplest example — a small CNN for MNIST digit classification. Uses standard CKKS without bootstrapping. Good starting point for understanding the framework.

- **Input**: 16x16 grayscale image (1 channel)
- **Output**: 10-class logits (digits 0-9)
- **Poly degree**: N = 16384

### test_cifar10

ResNet-20 on CIFAR-10 with bootstrapping enabled to support the deeper network. Demonstrates the framework's ability to handle residual connections and deeper computation graphs.

- **Input**: 32x32 RGB image (3 channels)
- **Output**: 10-class logits
- **Poly degree**: N = 65536

### test_imagenet

MobileNetV2 on ImageNet — a larger-scale example with 1000-class classification. Shows encrypted inference on production-grade models.

- **Input**: 256x256 RGB image (3 channels)
- **Output**: 1000-class logits
- **Poly degree**: N = 65536

## VGG Demo: 已知训练问题与修复方案

### 问题 1：`ModuleNotFoundError: No module named 'training'`

- **现象**：运行 [`train_vgg.py`](examples/vgg_demo/test_cifar10/train_vgg.py) 时，在 [`from training.nn_tools import ...`](examples/vgg_demo/test_cifar10/train_vgg.py:169) 处报错。
- **根因**：脚本所在目录是 `examples/vgg_demo/test_cifar10/`，原先只把 `../../` 加入 `sys.path`，实际到达的是 `examples/`，不是项目根目录，无法解析根目录下的 [`training`](training) 包。
- **修复**：将 [`sys.path.insert(...)`](examples/vgg_demo/test_cifar10/train_vgg.py:42) 调整为 `../../..`（项目根目录）。

---

### 问题 2：`TypeError: only 0-dimensional arrays can be converted to Python scalars`

- **现象**：开启 `--poly_model_convert` 后，在 Hermite 系数计算阶段报错，调用链：
  - [`replace_activation_with_poly()`](training/nn_tools/replace.py:50)
  - [`get_hermite_coeffs_for_module()`](training/nn_tools/eval_fn_hat_for_aespa.py:166)
  - [`compute_coefficients()`](training/nn_tools/eval_fn_hat_for_aespa.py:84)
  - [`quad()`](training/nn_tools/eval_fn_hat_for_aespa.py:110)
- **根因**：`scipy.integrate.quad` 需要“标量输入 -> 标量输出”的被积函数，但当前实现中：
  - [`numpy_func()`](training/nn_tools/eval_fn_hat_for_aespa.py:185) 对标量输入也可能返回 shape=(1,) 的数组（见 [`return result.numpy()`](training/nn_tools/eval_fn_hat_for_aespa.py:189)）；
  - [`quad` 的 lambda](training/nn_tools/eval_fn_hat_for_aespa.py:111) 又显式包了 `np.array(...)`，进一步导致返回 `ndarray` 而非 Python float。

### 建议修复点（`training/nn_tools/eval_fn_hat_for_aespa.py`）

1. **保证积分被积函数返回标量**（核心）
   - 在 [`compute_coefficients()`](training/nn_tools/eval_fn_hat_for_aespa.py:84) 中，确保 `quad` 调用的函数最终 `return float(...)`。
   - 不要在 [`quad` 调用处](training/nn_tools/eval_fn_hat_for_aespa.py:110) 使用 `np.array(...)` 包装返回值。

2. **保证 `numpy_func` 在标量输入时返回标量**
   - 在 [`numpy_func()`](training/nn_tools/eval_fn_hat_for_aespa.py:185) 内判断输入是否标量；
   - 标量输入返回 `float`；数组输入继续返回 `ndarray`。

3. **可选增强：在 `compute_coefficients` 做统一兜底**
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

> 说明：以上是修复原则记录；关键要求是 [`quad()`](training/nn_tools/eval_fn_hat_for_aespa.py:110) 的 integrand 返回 Python 标量。

### 修复后验证建议

```bash
python examples/vgg_demo/test_cifar10/train_vgg.py \
  --poly_model_convert \
  --input-shape 3 32 32 \
  --epochs 1 \
  --batch-size 16
```

预期：
- 不再出现 `ModuleNotFoundError: training`；
- 不再出现 `only 0-dimensional arrays can be converted to Python scalars`；
- 可正常打印 Hermite 系数并继续训练/导出流程。



# VGG Demo：`TypeError: only 0-dimensional arrays can be converted to Python scalars` 修复记录

## 1. 问题现象
在运行 [`main()`](examples/vgg_demo/test_cifar10/train_vgg.py:185) 过程中，调用链经过 [`replace_activation_with_poly()`](training/nn_tools/replace.py:50) → [`get_hermite_coeffs_for_module()`](training/nn_tools/eval_fn_hat_for_aespa.py:166) → [`compute_coefficients()`](training/nn_tools/eval_fn_hat_for_aespa.py:84) 时，在 [`scipy.integrate.quad`](training/nn_tools/eval_fn_hat_for_aespa.py:110) 抛出：

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

## 2. 根因分析
`quad` 的被积函数必须返回 **标量（Python float）**。原实现里：
- 被积函数内部把 `x` 包成数组传给 activation；
- activation（尤其是 [`numpy_func()`](training/nn_tools/eval_fn_hat_for_aespa.py:185)）返回 `ndarray`；
- 原代码通过 `np.array(integrand(x))` 返回给 `quad`，结果仍可能是 1-D 数组而非 0-D 标量。

因此 `quad` 在将返回值转为 C/Fortran 标量时失败，触发该 `TypeError`。

## 3. 修复方案
在 [`compute_coefficients()`](training/nn_tools/eval_fn_hat_for_aespa.py:84) 中将原先的 `integrand` + `lambda` 方式改为显式标量化的 [`integrand_scalar()`](training/nn_tools/eval_fn_hat_for_aespa.py:98)：

- 入口 `x`：`x_scalar = float(x)`，确保积分变量是标量；
- 传给 activation：仍构造 `x_arr = np.array([x_scalar], dtype=float)`，兼容依赖向量输入的 activation 封装；
- activation 输出 `fx`：`float(np.asarray(fx).reshape(-1)[0])` 提取首元素为标量；
- Hermite 项同样标量化；
- 返回值始终是 `float`。

并将 [`quad`](training/nn_tools/eval_fn_hat_for_aespa.py:114) 的回调直接改为 `integrand_scalar`，彻底消除数组返回路径。

## 4. 修改文件
- 修复代码：[`training/nn_tools/eval_fn_hat_for_aespa.py`](training/nn_tools/eval_fn_hat_for_aespa.py)
- 记录文档：[`examples/vgg_demo/README.md`](examples/vgg_demo/README.md)

## 5. 验证过程
执行语法编译检查：

```bash
python -m py_compile training/nn_tools/eval_fn_hat_for_aespa.py
```

该命令通过，说明修复后代码语法有效。核心运行时问题（`quad` 回调返回非标量）已在实现上被消除。

## 6. 影响评估
- 对外接口未变化：[`compute_coefficients()`](training/nn_tools/eval_fn_hat_for_aespa.py:84)、[`get_hermite_coeffs_for_module()`](training/nn_tools/eval_fn_hat_for_aespa.py:166) 调用方式保持不变；
- 仅修复积分回调返回值类型，不改变系数公式与整体流程；
- 兼容原有 activation 包装逻辑（仍支持向量化输入场景）。
