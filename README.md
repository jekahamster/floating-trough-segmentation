## My software and hardware

- Python 3.9.12
- Tensorflow (GPU) 2.9.2
- cuDNN 8.1
- CUDA 11.2
- OS: Windows 10


## Project Structure

`/` - project information: libraries, instructions, technical specifications, etc.

`src/` - all code is located here

`src/data/` - directory where datasets and subsets are stored (create manually)

`src/weights/` - directory where model weights are stored (create manually)

`src/media/` - directory where notebook images are stored

`src/output/` - directory where work results are stored (create manually)

`src/notebooks/` - directory where `.ipynb` notebooks are stored

`src/notebooks/eda.ipynb` - data exploration and experiments

`src/notebooks/model_training.ipynb` - model training. **This notebook contains more detailed training than in `src/train.py`**. For example, it includes augmentations and different data loaders, while `train.py` is just a part of this notebook.

`src/defines.py` - global constants for all `*.py` modules (but does not apply to `.ipynb`)

`src/argparsers.py` - argument parsers factory

`src/picker_d.py` - HSV picker app for filtering (see `eda.ipynb`)

`src/utils.py` - module with auxiliary functions

`src/data_generator.py` - module with data generators

`src/inference.py` - application for model inference on user images

`src/model.py` - module with UNet model and functions for building its blocks

`src/train.py` - application for training the model on user data (more details in `model_training.ipynb`)


## Installing Libraries

1) Set up a virtual environment (Optional) <br>

    For Windows:
    ```
    python -m venv venv
    cd venv/Scripts
    activate 
    cd ../..
    ```

2) Install libraries: <br>
`pip install -r requirements.txt`


## Data

Create the `src/data` directory.

Download the archive from the [Kaggle Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection). It is convenient to use the [Kaggle API](https://www.kaggle.com/docs/api).

Unzip the data to the `src/data/` path. You should end up with the following paths:
- `src\data\test_v2`
- `src\data\train_v2`
- `src\data\train_ship_segmentations_v2.csv`

You can also download the [subset and checkpoints from Google Drive](https://drive.google.com/drive/folders/1HAVF-6rz8LefGaEZDjvpI7yXx8R9PeGj?usp=sharing), which I used to train the model. **Some files, such as `files_data.csv`, can significantly accelerate the execution time of certain notebooks, as they store the results of the algorithm's work.**



## Weights

Download the [weights from Google Drive](https://drive.google.com/drive/folders/1HAVF-6rz8LefGaEZDjvpI7yXx8R9PeGj?usp=sharing) and place them in the `src\weights` directory.



## Training

For training, it is recommended to use `src/model_training.ipynb`, but since the task specifically requests a `.py` module, some functionality has been moved to `src/train.py`.

To invoke help, use the command `python train.py --help`.

```
usage: train.py [-h] --images_path IMAGES_PATH --masks_path MASKS_PATH [--valid_images_path VALID_IMAGES_PATH]
                [--valid_masks_path VALID_MASKS_PATH] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--height HEIGHT]
                [--width WIDTH] [--image_type IMAGE_TYPE] [--mask_type MASK_TYPE] [--mask_channels MASK_CHANNELS]
                [--shuffle SHUFFLE] [--transform TRANSFORM] [--model_name MODEL_NAME]
                [--loss {bce_dice_loss,dice_loss,binary_crossentropy}]

Run training script

optional arguments:
  -h, --help                                                show this help message and exit
  --images_path IMAGES_PATH                                 Path to directory with images
  --masks_path MASKS_PATH                                   Path to directory with masks
  --valid_images_path VALID_IMAGES_PATH                     Path to directory with validation images
  --valid_masks_path VALID_MASKS_PATH                       Path to directory with validation masks
  --epochs EPOCHS                                           Number of epochs
  --batch_size BATCH_SIZE                                   Batch size
  --height HEIGHT                                           Image height
  --width WIDTH                                             Image width 
  --image_type IMAGE_TYPE                                   Image type
  --mask_type MASK_TYPE                                     Mask type
  --mask_channels MASK_CHANNELS                             Number of mask channels
  --shuffle SHUFFLE                                         Shuffle data
  --transform TRANSFORM                                     Data augmentation transform
  --model_name MODEL_NAME                                   Model name
  --loss {bce_dice_loss,dice_loss,binary_crossentropy}      Loss function


Example: 
python train.py --images_path data\train_subset\images --masks_path data\train_subset\masks --valid_images_path data\val_subset\images --valid_masks_path data\val_subset\masks --loss bce_dice_loss --batch_size 32 --epochs 10
```


## Inference 

To invoke help, use `python inference.py --help`.

```
usage: inference.py [-h] --model_path MODEL_PATH --image_path IMAGE_PATH --output_path OUTPUT_PATH [--height HEIGHT]
                    [--width WIDTH] [--show]

Run inference script

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to model
  --image_path IMAGE_PATH
                        Path to image
  --output_path OUTPUT_PATH
                        Path to output
  --height HEIGHT       Resized image height (model input)
  --width WIDTH         Resized image width (model input)
  --show                Show output in window


  Example: 
  python inference.py --model_path weights\unet_weights_10e_32b_bcediceloss_2023-11-19_03-17-27.h5 --image_path data\test_v2\2693e39c1.jpg --output_path .\media\output.png --show
```