from src.paper.run_registry import (
    REGISTRY_PATH,
    format_run_source,
    get_run_entry,
    get_run_path,
    load_run_entries,
    load_run_registry,
    resolve_project_path,
)


def test_run_registry_yaml_exists_and_loads():
    assert REGISTRY_PATH.exists()

    entries = load_run_entries()
    paths = load_run_registry()

    assert "within_eegnet_binary" in entries
    assert entries["within_eegnet_binary"]["run_tag"] == "20260316_1411"
    assert paths["within_eegnet_binary"].endswith("_comparison_cache_imagery_binary.json")


def test_run_registry_resolves_paths_and_formats_sources():
    entry = get_run_entry("extra_sessions_binary")

    assert entry["run_tag"] == "20260324_2131"
    assert get_run_path("extra_sessions_binary") == entry["path"]
    assert resolve_project_path("extra_sessions_binary").name == (
        "20260324_2131_extra_sessions_cache_imagery_binary.json"
    )
    assert "20260324_2131" in format_run_source("extra_sessions_binary")
    assert entry["path"] in format_run_source("extra_sessions_binary")
