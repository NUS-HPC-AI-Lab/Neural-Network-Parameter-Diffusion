#本脚本执行partial diffusion 目前支持：
# conv3 cifar10:  'classifier'
# res18/50 cifar10/100: 'linear'
# res18 cifar10: 'bn'
# res18 cifar10: 最后一层conv＋fc : 'conv_fc'
# bn all 9600
# res18  conv 2359296

dataset=${dataset:-"cifar100"} #cifar
size=${size:-"2048"}
gpu=${gpu:-"0"}
save=${save:-"res18_AE_bn_-c16"}
batch=${batch:-"100"}
number=${number:-"1"}
channel=${channel:-"16"}
diff_lr=${diff_lr:-"0.001"}
vq_lr=${vq_lr:-"0.001"}
weight_decay=${weight_decay:-"0.000002"}
latent_epoch=${latent_epoch:-"2000"}
total_epoch=${total_epoch:-"20000"}


if [ ! -d "./parameters/ResNet18_cifar100/bn2" ];then
    python3 generate_params.py \
        --gpu $gpu \
        --num_model $number
fi


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done


##conv3的fcdiffusion: target layer 'classifier' 
CUDA_VISIBLE_DEVICES=$gpu python partial_main_AE_ddpm_diff.py fit \
--config=config/autoaug_partial.yaml  --trainer.logger.name ${save} \
--model.target_layer='bn-2' \
--data.data_dir=./parameters/ResNet18_cifar100/bn2/ --trainer.max_epochs=${total_epoch} \
--model.dataset ${dataset} \
--model.channel ${channel} \
--model.latent_epoch ${latent_epoch} \
--model.weight_decay=${weight_decay} \
--model.diff_lr=${diff_lr} \
--model.vq_lr=${vq_lr} \
--model.param_num=$size \
--data.num_model=${number} --data.batch_size=$batch --trainer.check_val_every_n_epoch=200 \
--trainer.callbacks.dirpath=checkpoints_partial_ldm_conv_fc/${dataset}_${size}_${latent_epoch}_${total_epoch}_${number}_${save}/ #--ckpt=checkpoints/best-epoch=99-loss_all=0.000-v1.ckpt


# bash script/partial_AE_res18_bn.sh --gpu 0 --number 200

