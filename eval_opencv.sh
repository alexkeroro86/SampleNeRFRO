SCENE=glass
EXPERIMENT=refractive-nerf-jax
TRAIN_DIR=/home/cgv839/logs/$EXPERIMENT/${SCENE}
DATA_DIR=/home/cgv839/data/real/$SCENE

python -m eval \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --config=configs/"${SCENE}" \
  --gin_file=configs/${SCENE}.gin \
  --chunk 8192 --skip_frames=1 \
  --stage="radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_w-mod-bd-0.05_blur-5-3.0_uni384" \
  --gin_param="Config.radiance_weight_name='radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_w-mod-bd-0.05_blur-5-3.0_uni384'"
