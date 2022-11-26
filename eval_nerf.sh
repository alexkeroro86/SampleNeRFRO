SCENE=ship_skydome-bkgd_no-partial-reflect_cycles
EXPERIMENT=refractive-nerf-jax
TRAIN_DIR=/home/cgv839/logs/$EXPERIMENT/${SCENE}
DATA_DIR=/home/cgv839/data/synthetic/nerf/${SCENE}

python -m eval \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --config=configs/"${SCENE}" \
  --gin_file=configs/${SCENE}.gin \
  --chunk 8192 --skip_frames=1 \
  --stage="radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_wo-U" \
  --gin_param="Config.radiance_weight_name='radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_wo-U'"
