SCENE=ship_transparent-skydome-bkgd_cycles
EXPERIMENT=refractive-nerf-jax
TRAIN_DIR=/home/cgv839/logs/$EXPERIMENT/${SCENE}
DATA_DIR=/home/cgv839/data/synthetic/nerf/$SCENE

python -m voxelize_mesh \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --config=configs/${SCENE} \
  --gin_file=configs/${SCENE}.gin \
  --num_samples=4 --num_voxels=512 --extent=1.5 --threshold=1.165
