import pytest
import urllib.request
import json
from leaderboard import (
    score_to_elo,
    normalize_elo,
    SUBSET_REGISTRY,
    SUBSET_CATEGORIES,
    LeaderboardManager,
    leaderboard_mgr,
)

def test_score_to_elo_mapping():
    assert score_to_elo(0.0) == 1350.0
    assert score_to_elo(0.1) == 1350.0 + 115.0
    assert score_to_elo(-0.1) == 1350.0 - 115.0
    assert score_to_elo(None) == 1350.0

def test_normalize_elo():
    # Min is 1000 -> 0.50, max is 1850 -> 0.99
    assert normalize_elo(1000.0) == 0.50
    assert normalize_elo(1850.0) == 0.99
    assert 0.40 <= normalize_elo(500.0) <= 0.99

def test_subsets_meta_structure():
    meta = leaderboard_mgr.get_subsets_meta()
    assert "categories" in meta
    assert "registry" in meta
    assert len(meta["registry"]) >= 22
    for key, item in meta["registry"].items():
        assert "label" in item
        assert "cat" in item
        assert "type" in item

def test_ensure_subsets_structure():
    mgr = LeaderboardManager()
    raw_model = {
        "model_name": "test-gpt-model",
        "rating_overall": 1400.0,
        "rating_coding": 1380.0,
        "rating_math": 1390.0,
        "rating_hard": 1350.0,
    }
    mgr._cache = [raw_model]
    mgr._ensure_subsets_structure()
    processed = mgr._cache[0]
    assert "subsets" in processed
    subsets = processed["subsets"]
    assert "text" in subsets
    assert subsets["text"]["elo"] == 1400.0
    assert "webdev" in subsets
    assert subsets["webdev"]["elo"] == 1380.0
    assert "agent" in subsets
    assert "vision" in subsets

@pytest.mark.asyncio
async def test_leaderboard_api_and_sorting():
    from server import get_leaderboard
    res = await get_leaderboard(sort_col="agent", sort_order="desc")
    assert res["status"] == "ok"
    assert "subsets_meta" in res
    assert "registry" in res["subsets_meta"]
    assert len(res["data"]) > 0
    first = res["data"][0]
    assert "subsets" in first
    assert "agent" in first["subsets"]
