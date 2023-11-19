import argparse

from pathlib import Path
from defines import WEIGHTS_DIR
from defines import MEDIA_DIR
from datetime import datetime
from tensorflow import keras
from argparsers import training_script_parser
from data_generators import DataGenerator
from model import unet_model
from utils import *



def train(
        images_path,
        masks_path,

        valid_images_path=None,
        valid_masks_path=None,
        
        batch_size=32,
        height=128,
        width=128,
        image_type="jpg",
        mask_type="png",
        mask_channels=1,
        shuffle=True,
        
        epochs=10,
        optimizer="adam",
        loss=bce_dice_loss,
        transform=None,
        callbacks=None,
        model_name="unet"):
    
    metrics = ["accuracy", dice_score]
    
    images_path = Path(images_path)
    masks_path = Path(masks_path)

    if valid_images_path and not valid_masks_path or valid_masks_path and not valid_images_path:
        raise ValueError("Valid images and masks paths must be specified together")

    if valid_images_path:
        valid_images_path = Path(valid_images_path)
        valid_masks_path = Path(valid_masks_path)


    data_generator = DataGenerator(
        images_path=images_path,
        masks_path=masks_path,
        image_type=image_type,
        mask_type=mask_type,
        batch_size=batch_size,
        shuffle=shuffle,
        height=height,
        width=width,
        mask_channels=mask_channels,
        transform=transform)
    
    valid_data_generator = DataGenerator(
        images_path=valid_images_path,
        masks_path=valid_masks_path,
        image_type=image_type,
        mask_type=mask_type,
        batch_size=batch_size,
        shuffle=False,
        height=height,
        width=width,
        mask_channels=mask_channels,
        transform=None)

    model = unet_model((height, width, 3), n_classes=mask_channels)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    model_history = model.fit(
        data_generator, 
        epochs=epochs,
        validation_data=valid_data_generator,
        callbacks=callbacks    
    )
    
    
    if not WEIGHTS_DIR.exists():
        WEIGHTS_DIR.mkdir()
    
    plot_training_history(model_history, metrics=["accuracy", "dice_score"], save_dir=MEDIA_DIR)

    str_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    model_name = f"{model_name}_{str_time}"
    model.save_weights(WEIGHTS_DIR / f"{model_name}.h5")
    print(f"Model saved to {WEIGHTS_DIR / f'{model_name}.h5'}")
    

def main():
    parser = training_script_parser()
    args = parser.parse_args()

    images_path = args.images_path
    masks_path = args.masks_path

    valid_images_path = args.valid_images_path
    valid_masks_path = args.valid_masks_path

    epochs = args.epochs
    batch_size = args.batch_size
    height = args.height
    width = args.width
    image_type = args.image_type
    mask_type = args.mask_type
    mask_channels = args.mask_channels
    shuffle = args.shuffle
    transform = args.transform
    model_name = args.model_name
    loss = args.loss

    if loss == "bce_dice_loss":
        loss = bce_dice_loss

    elif loss == "dice_loss":
        loss = dice_loss

    elif loss == "binary_crossentropy":
        loss = keras.losses.binary_crossentropy
    
    train(
        images_path,
        masks_path,

        valid_images_path=valid_images_path,
        valid_masks_path=valid_masks_path,
        
        batch_size=batch_size,
        height=height,
        width=width,
        image_type=image_type,
        mask_type=mask_type,
        mask_channels=mask_channels,
        shuffle=shuffle,
        
        epochs=epochs,
        optimizer="adam",
        loss=loss,
        transform=None,
        callbacks=None,
        model_name=model_name
    )



if __name__ == "__main__":
    main()