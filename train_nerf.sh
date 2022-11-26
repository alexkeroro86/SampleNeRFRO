SCENE=ship_skydome-bkgd_no-partial-reflect_cycles
EXPERIMENT=refractive-nerf-jax
TRAIN_DIR=/home/cgv839/logs/$EXPERIMENT/${SCENE}
DATA_DIR=/home/cgv839/data/synthetic/nerf/${SCENE}


python -m train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --config=configs/"${SCENE}" \
  --gin_file=configs/${SCENE}.gin \
  --stage="radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_wo-U"
