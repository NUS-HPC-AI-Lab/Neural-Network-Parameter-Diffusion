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


def calculate_statistics(metrics):
    stats = {}
    for metric_name, values in metrics.items():
        values = np.array(values)
        stats[metric_name] = {
            'max': np.max(values),
            'min': np.min(values),
            'median': np.median(values),
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return stats


def save_results(test_item, test_items, metrics, stats):
    # 使用test_item的名称作为基础来命名结果文件
    base_name = os.path.basename(test_item)
    if base_name == '':  # 如果test_item以/结尾
        base_name = os.path.basename(os.path.dirname(test_item))

    # 如果test_item是目录，直接使用目录名
    # 如果是文件，去掉扩展名
    base_name = os.path.splitext(base_name)[0]

    # 创建结果文件路径
    result_filename = f"{base_name}_results.txt"
    save_dir = os.path.dirname(test_item) if os.path.isfile(test_item) else test_item
    full_path = os.path.join(save_dir, result_filename)

    with open(full_path, 'w', encoding='utf-8') as f:
        # 写入测试时间和基本信息
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试路径: {test_item}\n")
        f.write(f"测试模型数量: {len(test_items)}\n")
        f.write("\n" + "=" * 50 + "\n\n")

        # 写入每个模型的具体性能
        f.write("各模型性能详情:\n")
        f.write("-" * 50 + "\n")
        for i, item in enumerate(test_items):
            f.write(f"模型 {i + 1}: {os.path.basename(item)}\n")
            f.write(f"Loss: {metrics['Loss'][i]:.4f}\n")
            f.write(f"Accuracy: {metrics['Accuracy'][i]:.4f}\n")
            f.write("-" * 30 + "\n")

        # 写入统计信息
        f.write("\n性能统计:\n")
        f.write("=" * 50 + "\n")
        for metric_name, metric_stats in stats.items():
            f.write(f"\n{metric_name}统计:\n")
            f.write(f"最大值: {metric_stats['max']:.4f}\n")
            f.write(f"最小值: {metric_stats['min']:.4f}\n")
            f.write(f"中位数: {metric_stats['median']:.4f}\n")
            f.write(f"平均值: {metric_stats['mean']:.4f}\n")
            f.write(f"标准差: {metric_stats['std']:.4f}\n")
            f.write("-" * 30 + "\n")

        # 写入最佳模型信息
        best_acc_idx = np.argmax(metrics['Accuracy'])
        best_model = os.path.basename(test_items[best_acc_idx])
        best_acc = metrics['Accuracy'][best_acc_idx]
        f.write(f"\n最佳模型性能:\n")
        f.write(f"模型名称: {best_model}\n")
        f.write(f"准确率: {best_acc:.4f}\n")
        f.write(f"Loss: {metrics['Loss'][best_acc_idx]:.4f}\n")

    return full_path


if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型并加载预训练权重
    model = create_model(num_classes=100).to(device)
    pretrained_state = torch.load('/home/wangkai/cvpr_pdiff/p-diff-v2/dataset/cifar100_convnext_base/pretrained.pth',
                                  map_location=device)
    model.load_state_dict(pretrained_state)

    # 获取测试数据加载器
    _, test_loader = get_data_loaders(config)

    # 存储所有模型的性能指标
    metrics = defaultdict(list)
    test_items = get_test_items(test_item)

    print(f"\n开始测试 {len(test_items)} 个模型...\n")

    for item in test_items:
        print(f"测试模型: {os.path.basename(item)}")
        try:
            state = torch.load(item, map_location=device)
            model.load_state_dict({k: v.to(torch.float32).to(device) for k, v in state.items()}, strict=False)
            loss, acc, all_targets, all_predicts = test(model, test_loader, device)

            # 存储性能指标
            metrics['Loss'].append(loss)
            metrics['Accuracy'].append(acc)

            print(f"损失 = {loss:.4f}, 准确率 = {acc:.4f}\n")

        except Exception as e:
            print(f"测试模型 {os.path.basename(item)} 时发生错误: {str(e)}\n")
            continue

    if metrics:
        # 计算统计信息
        stats = calculate_statistics(metrics)

        # 保存结果到文件
        result_file = save_results(test_item, test_items, metrics, stats)

        print(f"\n测试结果已保存到: {result_file}")

        # 打印最佳模型信息
        best_acc_idx = np.argmax(metrics['Accuracy'])
        best_model = os.path.basename(test_items[best_acc_idx])
        best_acc = metrics['Accuracy'][best_acc_idx]
        print(f"\n性能最好的模型: {best_model}")
        print(f"最佳准确率: {best_acc:.4f}")
    else:
        print("\n没有成功测试任何模型!")
