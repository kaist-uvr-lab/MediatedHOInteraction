# from dataset.impl.sbu_kinect import DatasetSBUKinect
from dataset.dataset_uvrhand import DatasetUvrHand

# ----------------------------------------------------------------------------------------------------------------------
class DataFactory:
    """
    A factory class for instantiating different datasets
    """
    dataset_names = [
            'sbu', 'uvrhand'
        ]

    @staticmethod
    def instantiate(dataset_name, num_synth):
        """
        Instantiates a dataset with its name
        """

        if dataset_name not in DataFactory.dataset_names:
            raise Exception('Unknown dataset "{}"'.format(dataset_name))

        # if dataset_name == "sbu":
        #     return DatasetSBUKinect(num_synth=num_synth)
        if dataset_name == "uvrhand":
            return DatasetUvrHand(num_synth=num_synth)

        raise Exception('Unknown dataset "{}"'.format(dataset_name))
