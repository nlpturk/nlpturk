import pytest

from nlpturk.utils import batch_dataset, split_dataset


def test_batch_dataset():
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    # raises ValueError, if batch_size is not a positive integer
    with pytest.raises(ValueError):
        next(batch_dataset(data, batch_size=None))
    # default batch_size is 1
    batches = list(batch_dataset(data))
    assert len(batches) == 8
    # if batch_size is bigger than data length, return a single batch
    batches = next(batch_dataset(data, batch_size=10))
    assert batches == [1, 2, 3, 4, 5, 6, 7, 8]
    # last batch will not be padded
    batches = list(batch_dataset(data, batch_size=3))
    assert batches[0] == [1, 2, 3]
    assert batches[1] == [4, 5, 6]
    assert batches[2] == [7, 8]
    # data can be any Iterable, as well a string
    batches = list(batch_dataset('12345678', batch_size=3))
    assert batches[0] == '123'
    assert batches[1] == '456'
    assert batches[2] == '78'


def test_split_dataset():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # train and test sets
    train, dev, test = split_dataset(data, split_ratios=[0.8, 0., 0.2])
    assert len(train) == 8
    assert dev == None
    assert len(test) == 2
    # only train set
    train, dev, test = split_dataset(data, split_ratios=[1., 0., 0.])
    assert len(train) == 10
    assert dev == None
    assert test == None
    # train, dev, test sets
    train, dev, test = split_dataset(data, split_ratios=[0.6, 0.2, 0.2])
    assert len(train) == 6
    assert len(dev) == 2
    assert len(test) == 2
    # returns same outputs across multiple function calls
    train2, dev2, test2 = split_dataset(data, split_ratios=[0.6, 0.2, 0.2])
    assert train == train2
    assert dev == dev2
    assert test == test2
    # raises ValueError, if split ratios are not float in the [0, 1] range
    # and the sum is not 1.0.
    with pytest.raises(ValueError):
        split_dataset(data, split_ratios=[0.7, 0.2, 0.2])
    with pytest.raises(ValueError):
        split_dataset(data, split_ratios=[None, 0.2, 0.2])
    # raises ValueError, if train split is not float in the (0, 1] range
    with pytest.raises(ValueError):
        split_dataset(data, split_ratios=[0., 0.5, 0.5])
    # raises ValueError, if split_ratios is not type of list or tuple
    with pytest.raises(ValueError):
        split_dataset(data, split_ratios={0.6, 0.2, 0.2})
