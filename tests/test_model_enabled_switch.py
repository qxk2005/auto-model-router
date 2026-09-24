import pytest
from auto_router.catalog import ModelInfo, Catalog, Prices
from auto_router.config import build_model, load_config, RouterConfig

def test_model_info_enabled_default():
    m = ModelInfo(name="test-model", provider="test", upstream_id="test-model", prices=Prices(0, 0))
    assert m.enabled is True

def test_model_info_enabled_false():
    m = ModelInfo(name="test-model", provider="test", upstream_id="test-model", prices=Prices(0, 0), enabled=False)
    assert m.enabled is False

def test_build_model_enabled():
    entry_enabled = {"name": "m1", "provider": "p1", "enabled": True, "free": True}
    entry_disabled = {"name": "m2", "provider": "p2", "enabled": False, "free": True}
    entry_default = {"name": "m3", "provider": "p3", "free": True}

    m1 = build_model(entry_enabled, providers={}, bench=None)
    m2 = build_model(entry_disabled, providers={}, bench=None)
    m3 = build_model(entry_default, providers={}, bench=None)

    assert m1.enabled is True
    assert m2.enabled is False
    assert m3.enabled is True

def test_catalog_filters_disabled_models():
    m_on = ModelInfo(name="on", provider="p", upstream_id="on", prices=Prices(0, 0), enabled=True)
    m_off = ModelInfo(name="off", provider="p", upstream_id="off", prices=Prices(0, 0), enabled=False)

    catalog = Catalog([m for m in [m_on, m_off] if m.enabled])
    assert "on" in catalog
    assert "off" not in catalog
    assert len(catalog.all()) == 1
    assert catalog.all()[0].name == "on"
