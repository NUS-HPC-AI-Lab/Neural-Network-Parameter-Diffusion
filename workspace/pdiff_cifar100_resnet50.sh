accelerate launch \
  --main_process_port=29750 \
  --num_processes=1 \
  --gpu_ids='0' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  pdiff_cifar100_resnet50.py \
