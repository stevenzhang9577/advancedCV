# Robustness
[github](https://github.com/newyingyi/Robustness.git)

## 环境配置
- 实验环境记录在env.txt中
- `pip install -r env.txt`

## 运行说明
### 训练模型
- 训练模型相关代码在train文件夹中
- 生成模型
> `python -vgg19`
- 在utils文件中可以修改训练数据（cifar10/cifar100）
### 生成对抗样本
- 生成对抗样本相关代码在adv文件夹中
- 生成BIM对抗样本
> 修改76行代码为`adv = attack(x, y_list)`
> 修改最后一行代码为`generate_adv_sample('cifar10', 'bim')`
- 生成JSMA对抗样本
> 修改76行代码为`adv = attack(x, y_list，theta=2)`
> 修改最后一行代码为`generate_adv_sample('cifar10', 'jsma')`
- 生成CW对抗样本
> 修改76行代码为`adv = attack(x, y_list，confidence=20)`
> 修改最后一行代码为`generate_adv_sample('cifar10', 'cw')`
- 将`cifar10`修改为`cifar100`，修改模型链接及训练数据即可生成cifar100 VGG19的对抗样本
### 测试用例排序
- 进行测试用例排序相关代码在sa文件夹中
- 生成cifar10的LSA排序
> `python run.py -lsa -d cifar -num_classes 10`
- 生成cifar100 的DSA排序
> 修改数据集及模型链接
> `python run.py -dsa -d cifar -num_classes 100`
### 模型微调
- 进行模型微调的相关代码在finetune文件夹中
- 根据不同数据集排序结果修改`finetune_on_combine.py`中的list，并修改模型及数据集
- 运行代码
> `python finetune_on_combine.py`


