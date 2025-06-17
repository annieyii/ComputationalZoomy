python main.py \
  --img1      img/raw/img_00014.jpg   \
  --depth1    img/output/img_00014.npz \
  --stack_dir sunglassgirlimg    \
  --out       zoom.mp4             \
  --frames    20                   \
  --dolly_z   20
python pngs_to_mp4.py