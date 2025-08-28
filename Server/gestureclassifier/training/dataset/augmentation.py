import random
import numpy as np
import math

# ----------------------------------------------------------------------------------------------------------------------
class DataAugmenter(object):
    """
    Defines the interface for data augmentation
    """
    def __init__(self, dimensionality, random_seed):
        self.dimensionality = dimensionality
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

    def generate_samples(self, pts):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
class AugRandomScale(DataAugmenter):
    """
    Performs random scaling on a sample with the specified factors
    """
    def __init__(self, dimensionality, random_seed, factor_start, factor_end):
        super(AugRandomScale, self).__init__(dimensionality, random_seed)
        self.factor_start = factor_start
        self.factor_end = factor_end

    def generate_samples(self, pts):
        while True:
            rnd = [self.random.uniform(self.factor_start, self.factor_end) for i in range(self.dimensionality)]
            is_good = False

            for d in range(self.dimensionality):
                is_good = is_good or rnd[d] != 1

                if is_good:
                    break
            if is_good:
                rnd = np.asarray(rnd, dtype=np.float32)
                break


        for frame in range(pts.shape[0]):
            pt = pts[frame, :-15].reshape(-1, self.dimensionality) # 21, 3)
            synth_pt = rnd * pt
            pts[frame, :-15] = synth_pt.flatten()

        return pts


class AugRandomGradualTranslation(DataAugmenter):
    """
    Performs progressive translation that accumulates over frames
    Input shape: [frame_num, feature_num] where feature_num = num_joints * dimensionality
    """
    def __init__(self, dimensionality, random_seed, factor_start, factor_end):
        super(AugRandomGradualTranslation, self).__init__(dimensionality, random_seed)
        self.factor_start = factor_start
        self.factor_end = factor_end

    def generate_samples(self, pts):

        frame_num, feature_num = pts.shape
        while True:
            # Generate random translation vector per frame
            rnd = [self.random.uniform(self.factor_start, self.factor_end) for _ in range(self.dimensionality)]
            is_good = False

            for d in range(self.dimensionality):
                is_good = is_good or rnd[d] != 0
                if is_good:
                    break
            if is_good:
                translation_per_frame = np.asarray(rnd, dtype=np.float32)
                break

        synth_pts = pts.copy()

        # Apply progressive translation for each frame
        for frame_idx in range(frame_num):
            # Calculate cumulative translation (increases linearly with frame index)
            cumulative_translation = translation_per_frame * frame_idx

            # Reshape current frame to [num_joints, dimensionality]
            frame_data = synth_pts[frame_idx].reshape(-1, self.dimensionality)

            # Apply translation to all joints in this frame
            translated_frame = frame_data + cumulative_translation

            # Reshape back and update
            synth_pts[frame_idx] = translated_frame.reshape(-1)

        return synth_pts

class AugRandomRotation(DataAugmenter):
    """
    Performs random rotation on a sample around wrist joint (index 0) with specified angle range
    Only rotates X,Y coordinates in 2D plane while preserving Z values
    """

    def __init__(self, dimensionality, random_seed, angle_range_deg=5):
        super(AugRandomRotation, self).__init__(dimensionality, random_seed)
        self.angle_range_deg = angle_range_deg

    def _get_2d_rotation_matrix(self, angle):
        """Generate 2D rotation matrix for X,Y coordinates"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)

    def generate_samples(self, pts):

        while True:
            # Generate random rotation angle (in degrees) for 2D rotation
            angle = self.random.uniform(-self.angle_range_deg, self.angle_range_deg)

            # Check if angle is non-zero
            is_good = abs(angle) > 0.001

            if is_good:
                # Convert to radians
                angle_rad = math.radians(angle)
                break

        # Get 2D rotation matrix
        R_2d = self._get_2d_rotation_matrix(angle_rad)

        for frame in range(pts.shape[0]):
            synth_pt = pts[frame, :-15].reshape(-1, self.dimensionality) # (21, 3)

            xy_coords = synth_pt[:, :2]  # Extract X,Y coordinates (21, 2)
            rotated_xy = np.dot(xy_coords, R_2d.T)  # Rotate X,Y coordinates
            synth_pt[:, :2] = rotated_xy  # Update X,Y coordinates

            pts[frame, :-15] = synth_pt.flatten()

        return pts