SCENE=star-lamp_skydome-bkgd_no-partial-reflect_cycles
EXPERIMENT=refractive-nerf-jax
TRAIN_DIR=/home/cgv839/logs/$EXPERIMENT/${SCENE}
DATA_DIR=/home/cgv839/data/synthetic/nerf/${SCENE}

python -m extract_mesh \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --config=configs/${SCENE} \
  --gin_file=configs/${SCENE}.gin \
  --resolution=256 --range=1.2 --threshold=0.5 \
  --stage="radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128" \
  --gin_param="Config.radiance_weight_name='radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128'"
