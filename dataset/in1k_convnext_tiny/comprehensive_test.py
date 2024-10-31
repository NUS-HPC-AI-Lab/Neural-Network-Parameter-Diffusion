import os
import sys
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime

try:
    from finetune import *
except ImportError:
    from .finetune import *

try:
    test_item = sys.argv[1]
except IndexError:
    assert __name__ == "__main__"
    test_item = "./checkpoint"


def get_test_items(test_item):
    test_items = []
    if os.path.isdir(test_item):
        for item in os.listdir(test_item):
            if item.endswith('.pth'):
                item = os.path.join(test_item, item)
                test_items.append(item)
    elif os.path.isfile(test_item):
        test_items.append(test_item)
    return sorted(test_items)


def calculate_statistics(values):
    values = np.array(values)
    return {
        'max': np.max(values),
        'min': np.min(values),
        'median': np.median(values),
        'mean': np.mean(values),
        'std': np.std(values)
    }


def save_results(test_item, results, stats):
    # 使用test_item命名结果文件
    base_name = os.path.basename(test_item)
    if base_name == '':
        base_name = os.path.basename(os.path.dirname(test_item))
    base_name = os.path.splitext(base_name)[0]

    save_path = os.path.dirname(test_item) if os.path.isfile(test_item) else test_item
    result_file = os.path.join(save_path, f"{base_name}_results.txt")

    with open(result_file, 'w', encoding='utf-8') as f:
        # 基本信息
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试路径: {test_item}\n")
        f.write(f"测试模型数量: {len(results)}\n")
        f.write("\n" + "=" * 50 + "\n\n")

        # 每个模型的结果
        f.write("各模型性能详情:\n")
        f.write("-" * 50 + "\n")
        for model_name, (loss, acc) in results.items():
            f.write(f"模型: {model_name}\n")
            f.write(f"Loss: {loss:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write("-" * 30 + "\n")

        # 统计信息
        f.write("\n性能统计:\n")
        f.write("=" * 50 + "\n")

        # Loss统计
        f.write("\nLoss统计:\n")
        f.write(f"最大值: {stats['loss']['max']:.4f}\n")
        f.write(f"最小值: {stats['loss']['min']:.4f}\n")
        f.write(f"中位数: {stats['loss']['median']:.4f}\n")
        f.write(f"平均值: {stats['loss']['mean']:.4f}\n")
        f.write(f"标准差: {stats['loss']['std']:.4f}\n")
        f.write("-" * 30 + "\n")

        # Accuracy统计
        f.write("\nAccuracy统计:\n")
        f.write(f"最大值: {stats['accuracy']['max']:.4f}\n")
        f.write(f"最小值: {stats['accuracy']['min']:.4f}\n")
        f.write(f"中位数: {stats['accuracy']['median']:.4f}\n")
        f.write(f"平均值: {stats['accuracy']['mean']:.4f}\n")
        f.write(f"标准差: {stats['accuracy']['std']:.4f}\n")

        # 最佳模型信息
        best_acc = stats['accuracy']['max']
        best_model = max(results.items(), key=lambda x: x[1][1])[0]
        f.write(f"\n最佳模型性能:\n")
        f.write(f"模型名称: {best_model}\n")
        f.write(f"准确率: {best_acc:.4f}\n")
        f.write(f"Loss: {results[best_model][0]:.4f}\n")

    return result_file


if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型并加载预训练权重
    model = timm.create_model(config['model_name'], pretrained=True, num_classes=1000)
    model = model.to(device)

    # 获取测试数据加载器
    config['dataset_root'] = "/home/wangkai/data/imagenet"
    _, test_loader = get_data_loaders(config)

    # 存储测试结果
    results = {}  # 存储每个模型的结果
    losses = []  # 存储所有loss值
    accuracies = []  # 存储所有accuracy值

    test_items = get_test_items(test_item)
    print(f"\n开始测试 {len(test_items)} 个模型...\n")

    for item in test_items:
        model_name = os.path.basename(item)
        print(f"测试模型: {model_name}")

        try:
            state = torch.load(item, map_location=device)
            model.load_state_dict({k: v.to(torch.float32).to(device) for k, v in state.items()}, strict=False)
            loss, acc, all_targets, all_predicts = test(model, test_loader, device)

            # 保存结果
            results[model_name] = (loss, acc)
            losses.append(loss)
            accuracies.append(acc)

            print(f"损失 = {loss:.4f}, 准确率 = {acc:.4f}\n")

        except Exception as e:
            print(f"测试模型 {model_name} 时发生错误: {str(e)}\n")
            continue

    if results:
        # 计算统计信息
        stats = {
            'loss': calculate_statistics(losses),
            'accuracy': calculate_statistics(accuracies)
        }

        # 保存结果
        result_file = save_results(test_item, results, stats)
        print(f"\n测试结果已保存到: {result_file}")

        # 打印最佳模型信息
        best_acc = stats['accuracy']['max']
        best_model = max(results.items(), key=lambda x: x[1][1])[0]
        print(f"\n性能最好的模型: {best_model}")
        print(f"最佳准确率: {best_acc:.4f}")
    else:
        print("\n没有成功测试任何模型!")