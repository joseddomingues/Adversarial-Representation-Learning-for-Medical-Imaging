# Check saving paths
GENERATORS_MODELS_PATH="test_generators"

if [ -d "$GENERATORS_MODELS_PATH" ]; then
  echo "Test Generators Folder Exists! Skipping Creation..."
else
  echo "Test Generators Folder Didn't Exists! Creating..."
  mkdir ./${GENERATORS_MODELS_PATH}
  echo "Done!"
fi

# Change to model directory
cd ../MedSinGAN

# Combinations
image_name_path="../images/benign.png"
stages=(12 12 16 16 20 20)
depth=(5 7 7 9 9 11)
niter=(1500 1500 1500 1500 1500 1500)

for ((i = 0; i < ${#stages[@]}; i++)); do

  # Construct experiment name
  exp_name="(G)S${stages[i]}D${depth[i]}I${niter[i]}_benign"

  # Run generation train
  python main_train.py --train_mode generation --input_name ${image_name_path} --train_stages "${stages[i]}" --niter "${niter[i]}" --train_depth "${depth[i]}" --experiment_name "${exp_name}" --gpu 0

  # Zip model and mlflows runs
  zip -r ../experiments/${GENERATORS_MODELS_PATH}/"${exp_name}".zip .

  # Delete current trained data
  rm -r /TrainedModels
  rm -r /mlruns
  rm -r /runs
done
