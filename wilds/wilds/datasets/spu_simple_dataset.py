import os
import torch
import numpy as np

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import Loss

def make_environment(num, frac, train=True):
    # first feature is common and second is spurious
    group_samples = (num * np.array(frac)).astype(int)
    np_groups = np.concatenate([
        np.zeros(group_samples[0]),
        np.ones(group_samples[1]),
        np.ones(group_samples[2]) * 2
    ]).astype(int)

    gs = torch.from_numpy(np_groups)

    X = torch.randn([num, 3])
    Y = (X[:, :2].sum(dim=-1) > 0).type(torch.float32)

    if train:
        rand = torch.tensor(
            np.random.binomial(
                1,
                0.6,
                [len(Y)]
            )
        )

        _Y = rand * Y + (1 - rand) * (1 - Y)
        Y = torch.where(gs == 0, _Y, Y)

    p1, p2, p3 = 1, 0.6, 0

    rand1 = np.random.binomial(1, p1, [len(Y)])
    rand3 = np.random.binomial(1, p3, [len(Y)])
    rand2 = np.random.binomial(1, p2, [len(Y)])

    rand1, rand2, rand3 = torch.tensor(rand1), torch.tensor(rand2), torch.tensor(rand3)

    # Create the spurious feature
    Y1 = Y
    Y2 = rand2 * Y + (1 - rand2) * (1 - Y)
    Y3 = 1 - Y

    X[:, 2] = torch.where(gs == 0, Y1, X[:, 2])
    X[:, 2] = torch.where(gs == 1, Y2, X[:, 2])
    X[:, 2] = torch.where(gs == 2, Y3, X[:, 2])

    # Measure the feature correlation
    correlations = [
        np.corrcoef(X[:, 0].numpy(), Y.numpy())[0, 1],
        np.corrcoef(X[:, 1].numpy(), Y.numpy())[0, 1],
        np.corrcoef(X[:, 2].numpy(), Y.numpy())[0, 1]
    ]

    print(f"feature correlations: {correlations}")

    print(X[:10], Y[:10], gs[:10])
    print("label stats:", np.unique(Y.numpy(), return_counts=True))
    print("g stats:", np.unique(gs.numpy(), return_counts=True))

    return {'X': X, 'y': Y, 'g': gs}


class SpuSimpleDataset(WILDSDataset):
    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']

        self._dataset_name = "spu_2feature"
        self._data_dir = os.path.join(root_dir, self._dataset_name)

        train_data = make_environment(1000, [0.49, 0.49, 0.02], train=True)
        val_data = make_environment(1000, [0.34, 0.33, 0.33], train=False)
        test_data = make_environment(10000, [0.34, 0.33, 0.33], train=False)
        
        _x_array, _y_array, _split_array, _g_array = [], [], [], []
        i = 0
        for di, d in enumerate([train_data, val_data, test_data]):
            x, y = d['X'], d['y']
            g = d['g']
            for j in range(len(y)):
                _x_array.append(x[j])
                _y_array.append(y[j])
                _g_array.append(g[j])
            _split_array += [di]*len(y)
        
        _y_array = np.array(_y_array)
        _g_array = np.array(_g_array)
        self._input_array = _x_array
        self._y_array = torch.LongTensor(_y_array)
        self._split_array = np.array(_split_array)
        # partition the train in to val and test
        self._split_scheme = split_scheme
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(_g_array), self._y_array),
            dim=1
        )
        self._metadata_fields = ['group', 'y']
        self._metadata_map = {
            'group': ['       majority', ' clean majority', '       minority'], 
            'y': [' 0', '1']
        }
                        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['group']))
        self._metric = Loss(loss_fn=torch.nn.CrossEntropyLoss())
        
        super().__init__(root_dir, download, split_scheme)
    
    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        return self._input_array[idx]
        
    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

if __name__ == '__main__':
    dset = SpuSimpleDataset('data')
    train, val, test = dset.get_subset('train'), dset.get_subset('val'), dset.get_subset('test')
    print ("Train, val, test sizes:", len(train), len(val), len(test))
