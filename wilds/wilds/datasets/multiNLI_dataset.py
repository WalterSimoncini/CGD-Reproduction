import os
import numpy as np
import torch
import pandas as pd
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class MultiNLIDataset(WILDSDataset):

    _dataset_name = 'multiNLI'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://drive.google.com/uc?export=download&confirm=9iBg&id=1TsCvu-qH2hILKKSu05AcF6K0gXmlo9-C',
            'compressed_size': 41_674_333}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata_preset.csv'),
            index_col=0)

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['gold_label'].values)
        self._y_size = 1
        self._n_classes = len(np.unique(self.y_array))

        # # Extract text
        # self._text_array = pd.read_csv(
        #     os.path.join(self._data_dir, 'metadata_preset.csv'),
        #     index_col=0)

        self.confounder_array = self._metadata_df['sentence2_has_negation'].values
        self.n_confounders = 1

        # Map to groups
        self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        self.group_array = (self.y_array * (self.n_groups / self.n_classes) + self.confounder_array).to(torch.int32)

        # Extract splits
        self._split_scheme = split_scheme
        self._split_array = self._metadata_df['split'].values
        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Load features
        self.features_array = []
        for feature_file in [
            'cached_train_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli-mm'
            ]:

            features = torch.load(
                os.path.join(
                    self.data_dir,
                    feature_file))

            self.features_array += features

        self.all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
        # self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)

        self._x_array = torch.stack((
            self.all_input_ids,
            self.all_input_masks,), dim=2)

        assert np.all(np.array(self.all_label_ids) == np.array(self.y_array))

        self._metadata_array = torch.cat(
            (
                torch.LongTensor(self.confounder_array).reshape((-1, 1)),
                self._y_array.reshape((-1, 1))
            ),
            dim=1
        )

        self._metadata_fields = ['sentence2_has_negation', 'y']

        self._eval_grouper = CombinatorialGrouper(
                dataset=self,
                groupby_fields=['sentence2_has_negation', 'y'])

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._x_array[idx, ...]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
