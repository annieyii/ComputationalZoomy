python main.py \
  --img1      img/raw/img_00014.jpg   \
  --depth1    img/output/img_00014.npz \
  --stack_dir sunglassgirlimg    \
  --out       zoom.mp4             \
  --frames    10                   \
  --dolly_z   10
python pngs_to_mp4.py