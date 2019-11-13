set -ex

CLASS="L2R_aug_good_with_depth"

# models
RESULTS_DIR="./results/${CLASS}"
G_PATH="./checkpoints/"${CLASS}"/"${CLASS}"_bicycle_gan/latest_net_G.pth"
E_PATH="./checkpoints/"${CLASS}"/"${CLASS}"_bicycle_gan/latest_net_E.pth"
CHECKPOINTS_DIR="./checkpoints/"${CLASS}"/"${CLASS}"_bicycle_gan"

# dataset
DIRECTION="AtoB" # from domain A to domain B
LOAD_SIZE_W=512
LOAD_SIZE_H=256
CROP_SIZE_W=512
CROP_SIZE_H=256
INPUT_NC=3 # number of channels in the input image
NZ=96
PREPROCESS="resize_and_crop"

NGF=108
NEF=108
NDF=108

LATEST_DIR=${CHECKPOINTS_DIR}"/"${CLASS}
mkdir -p ${LATEST_DIR}
cp ${G_PATH} ${LATEST_DIR}
cp ${E_PATH} ${LATEST_DIR}

# misc
GPU_ID=0   # gpu id

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./get_encoded_z.py \
  --dataroot ./datasets/${CLASS} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --load_size_w ${LOAD_SIZE_W} \
  --load_size_h ${LOAD_SIZE_H} \
  --crop_size_w ${CROP_SIZE_W} \
  --crop_size_h ${CROP_SIZE_H} \
  --input_nc ${INPUT_NC} \
  --preprocess ${PREPROCESS} \
  --nz ${NZ} \
  --ngf ${NGF} \
  --nef ${NEF} \
  --ndf ${NDF} \
  --center_crop \
  --no_flip \
  --gpu_ids "-1" \
  --serial_batches
