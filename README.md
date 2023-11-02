# diffnet
# 训练目标： resnet18 最后两层bn
# 训练集： 200个finetune后的参数（--number 200）

step1:
pip3 install -r requirements.txt

Without AE:
bash script/partial_diff_res18_bn.sh --gpu 0 --number 200

With AE:
bash script/partial_AE_diff_res18_bn.sh --gpu 0 --number 200
