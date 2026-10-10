import hashlib
import pytest
from desi_fm import assets


def test_fixed_checkpoint_resolution_and_rejections(tmp_path, monkeypatch):
    import huggingface_hub
    path = tmp_path/'checkpoint.pt'
    path.write_bytes(b'verified fixture')
    monkeypatch.setitem(assets.ASSETS, 'checkpoint_last.pt', (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
    calls = []
    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(path)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    assert assets.download_asset('checkpoint_last.pt') == str(path)
    assert calls == [(assets.MODEL_REPO, 'checkpoint_last.pt', {'revision': assets.MODEL_REVISION})]
    path.write_bytes(b'corrupt fixture!')
    with pytest.raises(ValueError, match='identity'):
        assets.download_asset('checkpoint_last.pt')


def test_api_uses_verified_default_without_remote_local_fallback(tmp_path, monkeypatch):
    from desi_fm import api
    monkeypatch.setattr(api, '_model', None)
    monkeypatch.delenv('DESI_FM_CKPT', raising=False)
    monkeypatch.setattr(api, 'download_asset', lambda name: 'verified.pt')
    monkeypatch.setattr(api, 'load_model_from_checkpoint', lambda path, device: path)
    assert api.get_model() == 'verified.pt'
    monkeypatch.setattr(api, '_model', None)
    monkeypatch.setenv('DESI_FM_CKPT', str(tmp_path/'missing.pt'))
    monkeypatch.setattr(api, 'download_asset', lambda name: pytest.fail('remote fallback'))
    assert api.get_model() == str(tmp_path/'missing.pt')
