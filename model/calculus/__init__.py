# Calculus segmentation head for 3DTeethSAM
# Use explicit imports to avoid triggering heavy dependencies (hydra, pytorch3d)
# at package-level import time:
#   from model.calculus.calculus_system import CalculusSegmentationSystem

__all__ = ['CalculusSegmentationSystem']
