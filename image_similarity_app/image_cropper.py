"""
This module provides functionality to crop images from the top and bottom of screenshots,
supporting device-specific or custom pixel values.
"""

import os
from PIL import Image

class ImageCropper:
    """Class to crop images from top and bottom using device-specific or custom pixel values."""

    DEVICE_CROP = {
        "Pixel_9": {"top": 152, "bottom": 70}
    }

    def __init__(self, device=None, top_pixels=None, bottom_pixels=None):
        """
        Initialize cropper with either a device or explicit pixel values.

        Args:
            device (str, optional): Device type to use for crop values.
            top_pixels (int, optional): Pixels to crop from top (overrides device).
            bottom_pixels (int, optional): Pixels to crop from bottom (overrides device).
        """
        if device:
            self.top_pixels = self.DEVICE_CROP[device]["top"]
            self.bottom_pixels = self.DEVICE_CROP[device]["bottom"]
        else:
            self.top_pixels = top_pixels if top_pixels else 0
            self.bottom_pixels = bottom_pixels if bottom_pixels else 0

    
    def set_top_pixels(self, top_pixels):
        """
        Set value of top pixels to crop.

        Args:
            top_pixels (int): Number of pixels to crop off top of image
        """
        self.top_pixels = top_pixels

    
    def set_bottom_pixels(self, bottom_pixels):
        """
        Set value of bottom pixels to crop.

        Args:
            bottom_pixels (int): Number of pixels to crop off bottom of image
        """
        self.top_pixels = bottom_pixels


    def crop_image(self, input_path, output_path):
        """
        Crops an image from the top and bottom.

        Args:
            input_path (str): Path to input image.
            top_pixels (int): Number of pixels to crop from the top.
            bottom_pixels (int): Number of pixels to crop from the bottom.
            output_path (str): Path to save cropped output image.
        """
        img = Image.open(input_path)
        width, height = img.size

        crop_box = (0, self.top_pixels, width, height - self.bottom_pixels)
        cropped_img = img.crop(crop_box)

        cropped_img.save(output_path)


    def crop_directory(self, input_dir, output_dir):
        """
        Crops each image in a directory from the top and bottom.

        Args:
            input_dir (str): Path to directory containing images to be cropped.
            top_pixels (int): Number of pixels to crop from the top.
            bottom_pixels (int): Number of pixels to crop from the bottom.
            output_dir (str): Path to save cropped output images.
        """
        image_exts = (".png", ".jpg", ".jpeg")

        img_paths = [
            os.path.join(input_dir, img_name) 
            for img_name in os.listdir(input_dir) 
            if img_name.lower().endswith(image_exts)
        ]

        for img_path in img_paths:
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            self.crop_image(img_path, output_path)


if __name__ == "__main__":
    cropper = ImageCropper("Pixel_9")
    
    cropper.crop_directory("static/screenshots/fullscreen/", "static/screenshots/cropped/")