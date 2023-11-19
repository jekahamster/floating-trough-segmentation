import argparse

def training_script_parser():
    parser = argparse.ArgumentParser(description='Run training script')
    
    parser.add_argument(
        '--images_path', 
        type=str, 
        required=True, 
        help='Path to directory with images'
    )

    parser.add_argument(
        '--masks_path', 
        type=str,
        required=True,
        help='Path to directory with masks'
    )

    parser.add_argument(
        "--valid_images_path",
        type=str,
        default=None,
        help="Path to directory with validation images"
    )

    parser.add_argument(
        "--valid_masks_path",
        type=str,
        default=None,
        help="Path to directory with validation masks"
    )

    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10, 
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        help='Batch size'
    )

    parser.add_argument(
        '--height', 
        type=int, 
        default=128, 
        help='Image height'
    )

    parser.add_argument(
        '--width', 
        type=int, 
        default=128, 
        help='Image width'
    )

    parser.add_argument(
        '--image_type', 
        type=str, 
        default="jpg", 
        help='Image type'
    )

    parser.add_argument(
        '--mask_type', 
        type=str, 
        default="png", 
        help='Mask type'
    )

    parser.add_argument(
        '--mask_channels', 
        type=int, 
        default=1, 
        help='Number of mask channels'
    )

    parser.add_argument(
        '--shuffle', 
        type=bool, 
        default=True, 
        help='Shuffle data'
    )

    parser.add_argument(
        '--transform', 
        type=bool, 
        default=None, 
        help='Data augmentation transform'
    )

    parser.add_argument(
        '--model_name', 
        type=str, 
        default="unet", 
        help='Model name'
    )

    parser.add_argument(
        "--loss",
        choices=["bce_dice_loss", "dice_loss", "binary_crossentropy"],
        default="bce_dice_loss",
        help="Loss function"
    )

    return parser


def inference_script_parser():
    parser = argparse.ArgumentParser(description='Run inference script')

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to image"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Image height"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Image width"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show output"
    )


    return parser