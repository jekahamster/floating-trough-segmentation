import numpy as np
import cv2
import imageio

from model import unet_model
from argparsers import inference_script_parser
from tensorflow import keras
from matplotlib import pyplot as plt


def process_image(image, height, width):
    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32)
    image = image / 255.0
    return image


def inference(model_path, image_path, output_path, height=128, width=128, show=False):
    model = unet_model((height, width, 3), n_filters=32, n_classes=1)
    model.load_weights(model_path)

    image = imageio.imread(image_path)
    image = process_image(image, height=height, width=width)
    batch_image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(batch_image)
    prediction = predictions[0]

    output_image = prediction * 255.0
    output_image = output_image.astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    imageio.imwrite(output_path, output_image)

    if show:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input")

        plt.subplot(1, 3, 2)
        plt.imshow(output_image)
        plt.title("Prediction")
        
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(prediction, alpha=0.5)
        plt.title("Output")
        plt.show()



def main():
    parser = inference_script_parser()
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path
    output_path = args.output_path
    height = args.height
    width = args.width
    show = args.show

    inference(model_path, image_path, output_path, height=height, width=width, show=show)
    


if __name__ == "__main__":
    main()