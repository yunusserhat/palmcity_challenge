"""Check untrusted Drive listings and restart behavior without any network access."""

from pathlib import Path

from PIL import Image
import pytest

from palmcity.download_data import download_one, inspect_png, parse_folder_listing


def listing(name: str, *, folder: bool = False) -> str:
    href = ("https://drive.google.com/drive/folders/officialfolder" if folder else
            "https://drive.google.com/file/d/officialfile/view?usp=drive_web")
    return (f'<a href="{href}"><div class="flip-entry-visual"><img alt="PNG Image"/></div>'
            f'<div class="flip-entry-title">{name}</div></a>')


def test_complete_embedded_listing_and_html_entities():
    html = "".join(listing(f"GS__{index:04}.png") for index in range(497))
    assert len(parse_folder_listing(html)) == 497
    assert parse_folder_listing(listing("train", folder=True))[0]["kind"] == "folder"
    assert parse_folder_listing(listing("A&amp;B.png"))[0]["name"] == "A&B.png"


@pytest.mark.parametrize("name", ["../evil.png", "..", "", r"sub\evil.png"])
def test_rejects_unsafe_listing_names(name):
    with pytest.raises(ValueError, match="Unsafe"):
        parse_folder_listing(listing(name))


def test_rejects_duplicate_filenames():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_folder_listing(listing("GS.png") + listing("GS.png"))


def test_existing_valid_png_is_verified_without_network(tmp_path: Path, monkeypatch):
    path = tmp_path / "GS.png"
    Image.new("L", (1024, 512), 31).save(path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network accessed"))
    result = download_one({"path": "GS.png", "id": "source_id", "download_url": "unused"}, tmp_path)
    assert result["existing_file_verified"] is True
    assert result["png_mode"] == "L"
    assert len(result["sha256"]) == 64


def test_invalid_existing_image_is_not_overwritten(tmp_path: Path):
    path = tmp_path / "GS.png"
    Image.new("RGB", (8, 4)).save(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="dimensions/format"):
        inspect_png(path)
    assert path.read_bytes() == original


def test_existing_symlink_is_not_read(tmp_path: Path, monkeypatch):
    link = tmp_path / "GS.png"
    link.symlink_to(tmp_path / "unrelated.png")
    monkeypatch.setattr("palmcity.download_data.inspect_png", lambda *args: pytest.fail("symlink read"))
    with pytest.raises(ValueError, match="Symlink"):
        download_one({"path": "GS.png", "download_url": "unused"}, tmp_path)
