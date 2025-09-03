import os
from pathlib import Path
import numpy as np

from dataset.dataset import Dataset
from dataset.augmentation import AugRandomScale, AugRandomRotation, AugRandomGradualTranslation
from dataset.impl.lowlevel import Sample, LowLevelDataset
from utils.logger import log
import time
import cv2
import copy


gnames_long = [
    'Clock',
    'CClock',
    'Natural'
]
gnames_short = [
    'Up',
    'Down',
    'Left',
    'Right',
    'Tap'
]
seq_len = 10
seq_gap = 2


# training set always include trial0
# ["trial1", "..."] : "trial1, ..." in test set
FOLDS = [
        ["trial4"],
        ["trial5"],
        ["All"],
        ["trial1"],
        ["trial2"],
        ["trial3"],
    ]

partial_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 20]

# ----------------------------------------------------------------------------------------------------------------------
class DatasetUvrHand(Dataset):
    def __init__(self, root="data\\uvrhand\\", num_synth=0):
        super(DatasetUvrHand, self).__init__("UvrHand", root, num_synth)
        self.num_synth = num_synth


    def _load_underlying_dataset(self):
        self.augs = self._get_augmenters(int(time.time())
)
        self.underlying_dataset = self._load_uvrhand_interaction()
        self.num_features = len(self.underlying_dataset.samples[0].pts[0])  # 3D world coordinates joints of single hand (21 joints x 3 dimensions).
        self.num_folds = len(FOLDS)
        self.folds = FOLDS

    def _get_augmenters(self, random_seed):
        return [
            AugRandomScale(3, random_seed, 0.7, 1.3),
            AugRandomRotation(3, random_seed, 15),
            AugRandomGradualTranslation(3, random_seed, -2, 2),
        ]

    def _load_uvrhand_interaction(self):
        """
        Loads the UvrHand dataset. We unnormalize the raw data using the equations
        that are provided in the dataset's documentation.
        """

        # Make sure the dataset exists
        self._check_dataset()

        # Pre-set 5-fold cross validations from dataset's README
        # FOLD[i] means train on every other fold, test on fold i
        # Number of folds
        FOLD_CNT = len(FOLDS)

        # Using 5-fold cross validation as the predefined test
        # (e.g. train[0] test[0] mean test on FOLD[0], train on everything else)
        train_indices = [[] for i in range(FOLD_CNT)]
        test_indices = [[] for i in range(FOLD_CNT)]
        samples = []

        for tname in sorted(os.listdir(self.root)):
            tpath = os.path.join(self.root, tname)  # uvrhand/trial0

            for cname in sorted(os.listdir(tpath)):
                cpath = os.path.join(tpath, cname)  # uvrhand/trial0/CClock_index_Grasp

                for fname in sorted(os.listdir(cpath)):
                    fpath = os.path.join(cpath, fname)

                    trial, label, finger, example = tname, cname.split('_')[0], cname.split('_')[1], fname


                    # Parse the example file
                    raw_data = np.load(fpath)  # (n,78) : last column is zero, 78: 21*3 + 15 (angle label)


                    # sampling the data
                    data_action = []
                    num_seq = len(raw_data) // seq_gap
                    num_seq -= seq_len // seq_gap
                    for seq in range(num_seq):
                        data_action.append(raw_data[seq * seq_gap:seq * seq_gap + seq_len])
                    data_action.append(raw_data[-seq_len:])

                    for single_action in data_action:
                        pts = [single_action[i] for i in range(single_action.shape[0])]


                        if label == 'Natural':
                            label_name = label
                        else:
                            label_name = f"{label}_{finger}"

                        # self._visualize_tips(pts, "origin")

                        # Augment samples
                        for s_idx in range(self.num_synth):
                            pts_aug = np.asarray(copy.deepcopy(pts)) # ndarray (12,78)

                            for aug in self.augs:
                                pts_aug = aug.generate_samples(pts_aug)

                            # self._visualize_tips(pts_aug, "aug")
                            # cv2.waitKey(0)

                            pts_aug_norm = _normalize(pts_aug)
                            pts_aug_norm = _extract_partialhand(pts_aug_norm)
                            samples += [Sample(pts_aug_norm, label_name, trial)]

                        pts_norm = _normalize(pts)
                        pts_norm = _extract_partialhand(pts_norm)
                        samples += [Sample(pts_norm, label_name, trial)]


        for s_idx, sample in enumerate(samples):
            # Add the index to train/test indices for each fold
            for fold_idx in range(FOLD_CNT):
                fold = FOLDS[fold_idx]

                if fold[0] == 'All':
                    train_indices[fold_idx] += [s_idx]

                    if len(test_indices[fold_idx]) < 1000:
                        test_indices[fold_idx] += [s_idx]
                    continue

                trial = sample.subject
                label_gt = sample.label

                if trial == 'trial0':
                    train_indices[fold_idx] += [s_idx]
                elif label_gt == 'Natural':
                    train_indices[fold_idx] += [s_idx]
                elif trial in fold:
                    # Add the instance as a TESTING instance to this fold
                    test_indices[fold_idx] += [s_idx]

                    # For all other folds, this guy would be a TRAINING instance
                    for other_idx in range(FOLD_CNT):
                        if fold_idx == other_idx:
                            continue
                        train_indices[other_idx] += [s_idx]


        # k-fold sanity check
        for fold_idx in range(FOLD_CNT):
            if FOLDS[fold_idx][0] == 'All':   # skip for 'All' fold
                continue
            assert len(train_indices[fold_idx]) + len(test_indices[fold_idx]) == len(samples)
            # Ensure there is no intersection between training/test indices
            assert len(set(train_indices[fold_idx]).intersection(test_indices[fold_idx])) == 0

        return LowLevelDataset(samples, train_indices, test_indices)

    def _check_dataset(self):
        if not os.path.isdir(self.root):
            log("Dataset files do not exist, check...")
            raise Exception

    @staticmethod
    def _series_len(pts):
        """
        Computes the path length of a sample
        """
        ret = 0.0

        for idx in range(1, len(pts)):
            ret += np.linalg.norm(pts[idx] - pts[idx - 1])

        return ret


    def _compute_ang_from_joint(self, joint):  # joint : (21, 3)

        joint = joint.reshape(21, 3)
        # Compute angles between joints
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
        v = v2 - v1  # [20, 3]
        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

        angle = np.degrees(angle)  # Convert radian to degree

        return angle


    def _visualize_tips(self, pts, fig="vis"):
        pts = np.asarray(pts)   # (n, 78)

        canvas = np.zeros((360, 640, 3), dtype=np.uint8)

        # 엄지와 검지 관절 인덱스
        thumb_indices = [1, 2, 3, 4]
        index_indices = [5, 6, 7, 8]

        # 궤적 누적
        for frame in range(pts.shape[0]):
            joints = pts[frame, :][:-15]
            joints = joints.reshape(21, 3)

            # 엄지 궤적 그리기 (빨간색)
            for i in range(len(thumb_indices) - 1):
                pt1 = tuple(joints[thumb_indices[i]][:2].astype(int))
                pt2 = tuple(joints[thumb_indices[i + 1]][:2].astype(int))
                cv2.line(canvas, pt1, pt2, (0, 0, 255), thickness=2)

            # 검지 궤적 그리기 (청록색)
            for i in range(len(index_indices) - 1):
                pt1 = tuple(joints[index_indices[i]][:2].astype(int))
                pt2 = tuple(joints[index_indices[i + 1]][:2].astype(int))
                cv2.line(canvas, pt1, pt2, (255, 255, 0), thickness=2)

        # 결과 이미지 보기
        cv2.imshow(fig, canvas)
        cv2.waitKey(1)

def _normalize(pts, norm_ratio_x=180.0, norm_ratio_y=180.0, norm_ratio_z=100.0):
    """
    Normalize a single sample

    :param sample: the sample to normalize
    :return: the normalized sample
    """

    pts = np.asarray(pts)

    pts_norm = np.zeros((pts.shape[0], pts.shape[1]))

    for frame_idx in range(pts.shape[0]):
        target_pose = pts[frame_idx, :63].reshape(21, 3)
        target_angle = pts[frame_idx, 63:]

        # norm 2d pose
        if frame_idx == 0:
            root_pose = target_pose[0, :]
        norm_pose = target_pose - root_pose

        norm_pose[:, 0] = norm_pose[:, 0] / norm_ratio_x
        norm_pose[:, 1] = norm_pose[:, 1] / norm_ratio_y
        norm_pose[:, 2] = norm_pose[:, 2] / norm_ratio_z

        # update pose and angle
        pts_norm[frame_idx, :63] = norm_pose.flatten()
        pts_norm[frame_idx, 63:] = target_angle / 180.0

    # # remove NaN values
    # nan_indice = np.argwhere(np.isnan(pts_norm))
    # for nan_idx in nan_indice:
    #     pts_norm[nan_idx[0], nan_idx[1]] = 0.0

    pts_norm = [np.array(row) for row in pts_norm]

    return pts_norm

def _extract_partialhand(pts_norm):
        # set partial pts
        # 0~4   5~8   9 12   13 16   17 20
        # pts_norm : (seq_len, 63+15) -> (seq_len, 45+15)
        pts_norm = np.asarray(pts_norm)
        temp = []
        for frame_idx in range(pts_norm.shape[0]):
            target_pose = pts_norm[frame_idx, :63].reshape(21, 3)
            target_angle = pts_norm[frame_idx, 63:]

            target_pose = target_pose[partial_idx, :]
            target_pose = target_pose.flatten()

            pts_ = np.concatenate((target_pose, target_angle), axis=0)
            temp.append(pts_)
        pts_norm = [np.array(row) for row in temp]

        return pts_norm