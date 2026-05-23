# PI0.5 训练加速方案建议

本文只讨论当前 PI0.5 训练的加速思路，重点区分两类方案：

1. 不改变训练逻辑的安全加速方案。
2. 会改变训练目标或训练逻辑的加速方案。

当前结论先写在前面：

```text
如果坚持当前 full finetune，不建议预计算 VLM 输出。
当前最稳妥的加速尝试是增加 DataLoader 的 num_workers。
```

## 1. 关于“冻结模块预计算”的判断

你提到的方案是：

```text
如果某些大模块被冻结，不参与反向传播和参数更新，
那么同一批数据经过这些模块后的输出是固定的，
因此可以提前预计算，训练时直接读取缓存，避免重复前向。
```

这个思路本身是正确的，而且在大 VLM / 大视觉编码器场景中通常非常有效。

但是它成立的前提是：

```text
被预计算的模块必须冻结。
```

也就是说，该模块的参数在整个训练过程中不能变化。

如果模块参数会更新，那么同一张图像、同一个 prompt 经过该模块的输出也会随着训练变化。此时提前缓存输出，会导致训练使用“旧参数产生的特征”，这会改变训练逻辑。

## 2. 当前 full finetune 下为什么不能预计算 VLM

当前 full finetune 的核心是：

```python
model=pi0_config.Pi0Config(pi05=True)
```

并且没有设置：

```python
freeze_filter=...
```

这意味着模型中的大部分参数都在训练，包括：

```text
VLM / PaliGemma
vision tower
language tower
action expert
action projection
time MLP
```

因此当前训练中，VLM 不是冻结模块。

所以在当前 full finetune 设定下：

```text
不能预计算 VLM 输出。
```

否则训练就不再是严格的 full finetune，而会变成：

```text
用旧 VLM 参数产生的固定特征训练后续模块
```

这会改变优化目标和最终结果。

## 3. 什么时候可以使用 VLM 预计算

如果后续你接受把训练目标改成“冻结 VLM，只训练后续模块”，那么 VLM 预计算就是一个很有价值的加速方向。

这种训练形式可以设计为：

```text
冻结 VLM / vision-language backbone
只训练 action expert、adapter 或 action head
```

然后离线预计算：

```text
图像特征
prompt 特征
prefix embedding
KV cache
```

训练时直接读取这些缓存特征，跳过大 VLM 的重复前向。

优点：

```text
显著减少每步计算量
显著减少显存压力
训练速度可能明显提升
```

缺点：

```text
不再是 full finetune
VLM 无法适配当前数据分布
最终效果可能和 full finetune 不同
需要新增缓存生成和缓存读取逻辑
```

因此它是一个“新的训练方案”，不是当前 full finetune 的无损加速。

## 4. 不改变训练逻辑的首选方案：增加 num_workers

当前比较安全的加速方案是增加数据加载并行度。

训练数据进入模型前，需要做不少 CPU 侧处理：

```text
读取图像数据
图像 resize 到 224x224
prompt tokenize
state/action transform
normalization
state/action padding
action chunking
```

如果 DataLoader worker 太少，GPU 可能会等 CPU 准备 batch。

当前默认值是：

```python
num_workers=2
```

可以先尝试：

```python
num_workers=8
```

如果 CPU 资源充足，也可以尝试：

```python
num_workers=12
```

推荐第一步只改成：

```python
num_workers=8
```

示例配置片段：

```python
num_train_steps=50_000,
batch_size=32,
num_workers=8,
log_interval=100,
save_interval=10_000,
keep_period=10_000,
fsdp_devices=4,
wandb_enabled=False,
```

这个方案的优点：

```text
不改变训练数据
不改变模型结构
不改变 loss
不改变优化目标
实现成本低
风险低
```

可能的问题：

```text
CPU 占用升高
内存占用升高
磁盘 IO 压力升高
如果瓶颈在 GPU 计算而不是数据加载，提速可能不明显
```

判断是否有效的方法：

```text
比较修改前后的 step time
例如当前约 1.8s/it
如果 num_workers=8 后明显低于 1.8s/it，就说明有效
如果基本不变，说明主要瓶颈在模型计算
```

## 5. 不改变训练逻辑的可选方案：预处理确定性数据变换

另一种理论上安全的方案是预处理确定性数据变换，而不是预计算模型输出。

可以考虑离线预处理：

```text
图像 resize 到 224x224
prompt tokenize
state/action padding 到模型维度
action chunking
normalization
```

这些操作不依赖模型参数，所以理论上不会改变 full finetune 的优化目标。

但是它有一个重要前提：

```text
离线预处理结果必须和当前在线 transform 完全一致。
```

当前数据管线里涉及：

```text
AlohaInputs
DeltaActions
Normalize
ResizeImages
TokenizePrompt
PadStatesAndActions
```

只要其中任何一个细节处理不一致，就会改变训练数据。

优点：

```text
减少每步 CPU 数据处理开销
对训练逻辑理论上无影响
适合数据集固定、长时间训练的场景
```

缺点：

```text
工程量较大
需要额外存储预处理后的数据
需要严格验证和当前在线 transform 一致
如果当前瓶颈主要是 GPU 计算，提速有限
```

因此该方案可以作为后续优化方向，但不建议在当前训练已经正常运行时立刻改。

## 6. 已经做好的加速/减负设置

当前配置已经把 checkpoint 保存间隔调大：

```python
save_interval=10_000
keep_period=10_000
```

这比默认每 1000 步保存更适合 full 模型训练。

原因是 full checkpoint 很大，频繁保存会拖慢训练。

当前设置会在以下步数保存并保留：

```text
10000
20000
30000
40000
50000
```

这个设置是合理的，不建议再改小。

另外训练脚本里已经设置了 JAX 编译缓存：

```python
jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
```

它可以减少相同 shape / 相同配置重复启动时的编译开销。

不过它主要优化启动阶段，不会明显加速 steady-state 每步训练。

## 7. 不建议轻易做的方案

### 7.1 关闭 EMA

当前默认会维护 EMA 参数：

```python
ema_decay=0.99
```

如果改成：

```python
ema_decay=None
```

可能减少显存和更新开销。

但是这会改变训练输出和后续评估行为，不属于严格无损加速。

所以不建议在当前目标下优先尝试。

### 7.2 冻结 VLM 再预计算

这个方案速度潜力很大，但它会改变训练目标。

它适合另开一个实验，而不是直接替代当前 full finetune。

如果要做，可以命名为类似：

```text
pi05_aloha_freeze_vlm_cached
```

而不要和当前 full finetune 混在一起。

## 8. 推荐优先级

当前推荐顺序如下：

```text
1. 先保持当前 full finetune 跑到第一个 checkpoint，也就是 step 10000。
2. 如果速度可以接受，不改。
3. 如果速度太慢，先尝试 num_workers=8。
4. 如果 num_workers=8 无明显提升，再判断瓶颈是否在 GPU 计算。
5. 如果长期训练很多次同一数据集，再考虑离线预处理确定性 transform。
6. 如果愿意改变训练目标，再单独设计冻结 VLM + 特征缓存方案。
```

## 9. 最终建议

针对当前 full finetune，最稳妥的加速建议是：

```python
num_workers=8
```

不建议当前直接做 VLM 预计算，因为 VLM 参数正在训练更新。

如果未来要做大幅加速，可以单独开一个冻结 VLM 的实验，专门设计：

```text
冻结 VLM
预计算 VLM 特征 / prefix embedding / KV cache
只训练 action expert / action head
```

但这应被视为新的训练方案，而不是当前 full finetune 的等价加速。
