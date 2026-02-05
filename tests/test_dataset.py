from unittest.mock import mock_open, patch
from skvaider_api_client.dataset import Dataset

def test_dataset_init():
    mock_data = "line1\nline2\n\nline3\n"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        dataset = Dataset("dummy_path.txt")
        assert dataset.data == ["line1", "line2", "line3"]
        assert len(dataset.data) == 3

def test_get_n_samples_smaller():
    mock_data = "line1\nline2\nline3\n"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        dataset = Dataset("dummy")
        samples = dataset.get_n_samples(2)
        assert len(samples) == 2
        assert samples == ["line1", "line2"]

def test_get_n_samples_larger():
    # should repeat
    mock_data = "line1\nline2\n"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        dataset = Dataset("dummy")
        samples = dataset.get_n_samples(5)
        # data has 2 items. need 5.
        # repetitions = 5//2 + 1 = 3. 
        # extended = [l1, l2, l1, l2, l1, l2] (len 6)
        # result = extended[:5] -> l1, l2, l1, l2, l1
        assert len(samples) == 5
        assert samples == ["line1", "line2", "line1", "line2", "line1"]

def test_create_batches():
    items = ["a", "b", "c", "d", "e"]
    batches = Dataset.create_batches(items, 2)
    assert len(batches) == 3
    assert batches[0] == ["a", "b"]
    assert batches[1] == ["c", "d"]
    assert batches[2] == ["e"]
