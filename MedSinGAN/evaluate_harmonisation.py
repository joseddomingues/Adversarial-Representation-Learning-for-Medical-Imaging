import os
import sys

sys.path.append("../")

import cv2
import numpy as np

from evaluate_generation import GenerationEvaluator
from utils.utils import resize_to_dim

BASE_COMPARE_CROPPED_PNG = "base_compare_cropped.png"
CROPPED_PNG = "harmonised_compare_cropped.png"


class HarmonisationEvaluator:
    def __init__(self, base_image, target_image, target_mask):
        # Resize inputs to original base image size
        base = cv2.imread(base_image)
        harmonised = resize_to_dim(target_image, base.shape[1], base.shape[0], None)
        harmonised_mask = resize_to_dim(target_mask, base.shape[1], base.shape[0], None)

        # Get the collage part of original and collage
        base_compare = base.copy()
        base_compare[harmonised_mask == 0] = 0

        harmonised_compare = harmonised.copy()
        harmonised_compare[harmonised_mask == 0] = 0

        # Crop both targets for evaluation
        positions = np.nonzero(base_compare)
        top, bottom, left, right = positions[0].min(), positions[0].max(), positions[1].min(), positions[1].max()
        base_compare_cropped = base_compare[top:bottom, left:right]

        positions = np.nonzero(harmonised_compare)
        top, bottom, left, right = positions[0].min(), positions[0].max(), positions[1].min(), positions[1].max()
        harmonised_compare_cropped = harmonised_compare[top:bottom, left:right]

        # Save images to give as input to Generator Evaluator
        cv2.imwrite(BASE_COMPARE_CROPPED_PNG, base_compare_cropped)
        cv2.imwrite(CROPPED_PNG, harmonised_compare_cropped)

        self.evaluator = GenerationEvaluator(BASE_COMPARE_CROPPED_PNG, padd_input=True)

    def run_lpips(self):
        """
        Run the lpips metric to the given images
        @return: The lpips value
        """
        return self.evaluator.run_lpips_to_image("harmonised_compare_cropped.png", padd=True)

    def run_mssim(self):
        """
        Run the ms-ssim metric to the given images
        @return: The ms-ssim values
        """
        return self.evaluator.run_mssim_to_image("harmonised_compare_cropped.png", padd=True)

    @staticmethod
    def clean_auxiliary_images():
        """
        Deletes the auxiliary images created for this harmonisation evaluation
        @return: 0 if properly deleted or -1 if not
        """
        try:
            os.remove("base_compare_cropped.png")
            os.remove("harmonised_compare_cropped.png")
            return 0
        except OSError:
            return -1
