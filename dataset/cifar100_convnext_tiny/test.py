import os
import sys
import torch

try:
    from finetune import *
except ImportError:
    from .finetune import *

try:
    test_item = sys.argv[1]
except IndexError:
    assert __name__ == "__main__"
    test_item = "./checkpoint"

test_items = []
if os.path.isdir(test_item):
    for item in os.listdir(test_item):
        if item.endswith('.pth'):
            item = os.path.join(test_item, item)
            test_items.append(item)
elif os.path.isfile(test_item):
    test_items.append(test_item)

if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型并加载预训练权重
    model = create_model(num_classes=100).to(device)
    pretrained_state = torch.load('/home/wangkai/cvpr_pdiff/p-diff-v2/dataset/cifar100_convnext_tiny/pretrained.pth', map_location=device)
    model.load_state_dict(pretrained_state)

    # 获取测试数据加载器
    _, test_loader = get_data_loaders(config)

    for item in test_items:
        print(f"测试模型: {os.path.basename(item)}")
        state = torch.load(item, map_location=device)
        model.load_state_dict({k: v.to(torch.float32).to(device) for k, v in state.items()}, strict=False)
        loss, acc, all_targets, all_predicts = test(model, test_loader, device)
        print(f"损失 = {loss:.4f}, 准确率 = {acc:.4f}\n")
