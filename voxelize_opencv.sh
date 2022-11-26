SCENE=pen
EXPERIMENT=refractive-nerf-jax
TRAIN_DIR=/home/cgv/logs/$EXPERIMENT/${SCENE}
DATA_DIR=/home/cgv/data/real/$SCENE

python -m voxelize_mesh \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --config=configs/${SCENE} \
  --gin_file=configs/${SCENE}.gin \
  --num_samples=4 --num_voxels=384 --threshold=1.165 \
  --extent=0 --min_point=-1.61707 --min_point=1.30378 --min_point=-1.64903 --max_point=1.88293 --max_point=4.80378 --max_point=1.85097  # pen
  # --extent=0 --min_point=-1.79102 --min_point=0.711703 --min_point=-1.75 --max_point=1.70898 --max_point=4.2117 --max_point=1.75  # glass
  # --extent=0 --min_point=-1 --min_point=0.59833 --min_point=-1 --max_point=1 --max_point=2.59833 --max_point=1  # ball (small)
  # --extent=0 --min_point=-1.75 --min_point=-0.15167 --min_point=-1.75 --max_point=1.75 --max_point=3.34833 --max_point=1.75  # ball
  # --extent=0 --min_point=0.196362 --min_point=0.211988 --min_point=0.245435 --max_point=0.596362 --max_point=0.611988 --max_point=0.645435
  # --extent=0 --min_point=0.205134 --min_point=0.211988 --min_point=0.170866 --max_point=0.605134 --max_point=0.611988 --max_point=0.570866
