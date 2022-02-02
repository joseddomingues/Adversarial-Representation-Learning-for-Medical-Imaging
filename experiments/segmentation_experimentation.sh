# Check saving paths
SEGMENTATION_MODELS_PATH="test_segmentations"

if [ -d "SEGMENTATION_MODELS_PATH" ]; then
  echo "Test Segmentations Folder Exists! Skipping Creation..."
else
  echo "Test Segmentations Folder Didn't Exists! Creating..."
  mkdir ./${SEGMENTATION_MODELS_PATH}
  echo "Done!"
fi

# Change to model directory
cd ../MedSegmentation

# Combinations
train_folder="../data"
val_folder="../data"
test_images="/test_images"
scheduler="cosine"

iters=(100 200 400 600 800 1000)
b_size=(1 1 1 1 1 1)
l_rate=(0.001 0.001 0.001 0.001 0.001 0.001)

for ((i = 0; i < ${#iters[@]}; i++)); do

  # Construct experiment name
  exp_name="(S)N${iters[i]}B${b_size[i]}LR${l_rate[i]}"
  model_checkpoints="${exp_name}_model_checkpoints"
  optimizer_checkpoints="${exp_name}_optimizer_checkpoints"

  # Run segmentation train
  python train_Unet.py --train_folder ${train_folder} --val_folder ${val_folder} --n_epochs "${iters[i]}" --batch_size "${b_size[i]}" --experiment_name "${exp_name}" --model_checkpoints "${model_checkpoints}" --optimizer_checkpoints "${optimizer_checkpoints}" --l_rate "${l_rate[i]}" --scheduler ${scheduler}

  # Segment target folder
  python api.py --model_dir "${model_checkpoints}" --test_images ${test_images} --no_eval

  # Zip model and mlflows runs
  zip -r ../${SEGMENTATION_MODELS_PATH}/"${exp_name}" .

  # Zip model and mlflows runs
  rm -r /mlruns
  rm -r /${model_checkpoints}
  rm -r /${optimizer_checkpoints}
  rm -r /runs
  rm -r /results
done
