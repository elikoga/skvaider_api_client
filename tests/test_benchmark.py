from unittest.mock import mock_open, patch, MagicMock
from skvaider_api_client import benchmark

def test_check_finish_reasons_success():
    responses = [
        {"choices": [{"finish_reason": "length"}]},
        {"choices": [{"finish_reason": "length"}, {"finish_reason": "length"}]}
    ]
    assert benchmark.check_finish_reasons(responses, 0, "length") is True

def test_check_finish_reasons_fail():
    responses = [
        {"choices": [{"finish_reason": "stop"}]}
    ]
    # Should print warning (we could capture stdout but return value is enough)
    assert benchmark.check_finish_reasons(responses, 0, "length") is False

def test_calculate_tokens_per_second():
    assert benchmark.calculate_tokens_per_second(100, 10.0) == 10.0
    assert benchmark.calculate_tokens_per_second(100, 0.0) == 0.0

def test_create_batch_result():
    res = benchmark.create_batch_result(0, 5, 100, 5.0, extra_field="val")
    assert res["batch_idx"] == 0
    assert res["tokens_per_second"] == 20.0
    assert res["extra_field"] == "val"

def test_tracker():
    tracker = benchmark.BenchmarkTracker()
    tracker.add_batch_result(0, 1, 10, 1.0)
    tracker.add_batch_result(1, 1, 20, 2.0)
    
    assert tracker.total_tokens == 30
    assert tracker.total_time == 3.0
    assert len(tracker.batch_results) == 2

def test_save_benchmark_results():
    with patch("builtins.open", mock_open()) as m:
        with patch("json.dump") as mock_json:
            data = {"a": 1}
            benchmark.save_benchmark_results("out.json", data)
            m.assert_called_with("out.json", "w")
            mock_json.assert_called()

def test_setup_benchmark():
    # Mock Config and Dataset
    with patch("skvaider_api_client.benchmark.Dataset") as MockDataset:
        with patch("skvaider_api_client.benchmark.APIClient") as MockClient:
            mock_conf = MagicMock()
            d, c = benchmark._setup_benchmark("dpath", mock_conf)
            MockDataset.assert_called_with("dpath")
            MockClient.assert_called_with(mock_conf)
