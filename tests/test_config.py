from skvaider_api_client.config import Config

def test_config_init():
    config = Config(skvaider_token="test_token", skvaider_url="http://test.url")
    assert config.skvaider_token == "test_token"
    assert config.skvaider_url == "http://test.url"

def test_config_validation():
    # pydantic validates types at runtime if configured, or just simple instantiation
    # Here we just check it instantiates correctly
    config = Config(skvaider_token="abc", skvaider_url="xyz")
    assert config.skvaider_token == "abc"
