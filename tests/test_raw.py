"""
Tests for Thorlabs ThorImageLS RAW input (suite2p.io.raw) using small synthetic
RAW/XML fixtures.

The synthetic RAW stream has shape (nvolumes, nplanes + nflyback, nchannels, Ly, Lx)
(time -> z-slice -> channel, flyback slices last), stored as little-endian uint16.
"""
import contextlib
import importlib
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import suite2p
from suite2p import io
from suite2p.io import raw as raw_mod

run_s2p_mod = importlib.import_module("suite2p.run_s2p")

LY, LX = 3, 5  # non-square so that row/column orientation is checked


def make_xml(Ly=LY, Lx=LX, channel_mask=1, frame_rate=30.0, frames=10, nplanes=1,
             nflyback=0, zfast=None, status="Complete", wavelengths=None,
             average_mode=None, average_num=None):
    if zfast is None:
        zfast = int(nplanes > 1 or nflyback > 0)
    if wavelengths is None:
        wavelengths = [n for b, n in zip((1, 2, 4, 8), "ABCD") if channel_mask & b]
    wl = "".join(f'<Wavelength name="Chan{n}" exposureTimeMS="0" />' for n in wavelengths)
    status_xml = f'<ExperimentStatus value="{status}" />' if status is not None else ""
    avg_xml = "".join(f' {k}="{v}"' for k, v in [("averageMode", average_mode),
                                                  ("averageNum", average_num)] if v is not None)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<ThorImageExperiment>
  <Name name="synthetic" />
  <Wavelengths nyquistExWavelengthNM="488" nyquistEmWavelengthNM="488">{wl}</Wavelengths>
  <ZStage name="ThorZPiezo" steps="{nplanes}" stepSizeUM="5" startPos="0" />
  <LSM pixelX="{Lx}" pixelY="{Ly}" channel="{channel_mask}" frameRate="{frame_rate}"
       widthUM="100" heightUM="60"{avg_xml} />
  <Streaming enable="1" frames="{frames}" rawData="1" flybackFrames="{nflyback}"
             zFastEnable="{zfast}" />
  {status_xml}
</ThorImageExperiment>
"""


def write_recording(folder, nvol=7, nplanes=1, nflyback=0, nchannels=1,
                    channel_mask=None, frame_rate=30.0, seed=0, raw_name="Image_0001_0001.raw",
                    xml_name="Experiment.xml", frames=None, extra_bytes=None, **xml_kw):
    """Write a synthetic recording; returns the uint16 data (nvol, z, c, Ly, Lx)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if channel_mask is None:
        channel_mask = {1: 1, 2: 3}[nchannels]
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 65536, size=(nvol, nplanes + nflyback, nchannels, LY, LX),
                        dtype=np.uint16)
    if frames is None:
        frames = nvol * (nplanes + nflyback)
    payload = data.astype("<u2").tobytes()
    if extra_bytes is not None:
        payload = payload + bytes(extra_bytes) if extra_bytes >= 0 else payload[:extra_bytes]
    (folder / raw_name).write_bytes(payload)
    if xml_name is not None:
        (folder / xml_name).write_text(make_xml(
            channel_mask=channel_mask, frame_rate=frame_rate, frames=frames,
            nplanes=nplanes, nflyback=nflyback, **xml_kw), encoding="utf-8")
    return data


def expected(data, plane, chan):
    """suite2p int16 movie of one plane/channel (same uint16 conversion as tiff)."""
    return (data[:, plane, chan] // 2).astype(np.int16)


def make_db(tmp_path, data_path, **kw):
    db = suite2p.default_db()
    db.update(data_path=[str(p) for p in data_path], save_path0=str(tmp_path / "out"),
              input_format="raw")
    db.update(kw)
    return db


def convert(db, settings=None):
    """Mirror run_s2p's conversion block (find files, init dbs, open files, convert)."""
    settings = suite2p.default_settings() if settings is None else settings
    fs, first_files = io.get_file_list(db)
    db["file_list"], db["first_files"] = fs, first_files
    db, settings = io.raw_init_db(db, settings)
    dbs = io.init_dbs(db)
    with contextlib.ExitStack() as stack:
        raw_str = "raw" if db["keep_movie_raw"] else "reg"
        files = [stack.enter_context(open(d[f"{raw_str}_file"], "wb")) for d in dbs]
        files_chan2 = ([stack.enter_context(open(d[f"{raw_str}_file_chan2"], "wb"))
                        for d in dbs] if db["nchannels"] > 1 else None)
        dbs = io.raw_to_binary(dbs, settings, files, files_chan2)
    return dbs, settings


def read_bin(path, Ly=LY, Lx=LX):
    return np.fromfile(path, dtype=np.int16).reshape(-1, Ly, Lx)


# ---------------------------------------------------------------- conversion


def test_single_plane_single_channel(tmp_path):
    data = write_recording(tmp_path / "rec", nvol=9, frame_rate=31.0)
    db = make_db(tmp_path, [tmp_path / "rec"], batch_size=4)  # several batches
    dbs, settings = convert(db)

    assert len(dbs) == 1
    d = dbs[0]
    assert (d["Ly"], d["Lx"]) == (LY, LX)  # Ly = pixelY rows, Lx = pixelX columns
    assert d["nframes"] == 9 and d["nplanes"] == 1 and d["nchannels"] == 1
    mov = read_bin(d["reg_file"])
    np.testing.assert_array_equal(mov, expected(data, 0, 0))
    np.testing.assert_allclose(d["meanImg"], mov.astype(np.float64).mean(axis=0), rtol=1e-6)
    assert d["meanImg"].dtype == np.float32
    np.testing.assert_array_equal(d["frames_per_file"], [9])
    np.testing.assert_array_equal(d["frames_per_folder"], [9])
    assert settings["fs"] == pytest.approx(31.0)

    # persisted db / settings
    saved = np.load(d["db_path"], allow_pickle=True).item()
    assert saved["nframes"] == 9 and (saved["Ly"], saved["Lx"]) == (LY, LX)
    np.testing.assert_array_equal(saved["meanImg"], d["meanImg"])
    assert saved["raw_metadata"][0]["nvolumes"] == 9
    saved_settings = np.load(d["settings_path"], allow_pickle=True).item()
    assert saved_settings["fs"] == pytest.approx(31.0)


@pytest.mark.parametrize("functional_chan", [1, 2])
def test_multiplane_two_channels_with_flyback(tmp_path, functional_chan):
    data = write_recording(tmp_path / "rec", nvol=6, nplanes=3, nflyback=1,
                           nchannels=2, frame_rate=40.0)
    db = make_db(tmp_path, [tmp_path / "rec"], functional_chan=functional_chan,
                 batch_size=10)
    dbs, settings = convert(db)

    assert len(dbs) == 3  # flyback slice discarded
    assert settings["fs"] == pytest.approx(40.0 / 4)
    f, o = functional_chan - 1, 2 - functional_chan
    for j, d in enumerate(dbs):
        assert d["nplanes"] == 3 and d["nchannels"] == 2 and d["nframes"] == 6
        np.testing.assert_array_equal(read_bin(d["reg_file"]), expected(data, j, f))
        np.testing.assert_array_equal(read_bin(d["reg_file_chan2"]), expected(data, j, o))
        np.testing.assert_allclose(d["meanImg"], expected(data, j, f).mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(d["meanImg_chan2"], expected(data, j, o).mean(axis=0),
                                   rtol=1e-6)


def test_single_channel_b_bitmask_is_one_channel(tmp_path):
    # LSM channel is a bitmask: 2 = ChanB only (the legacy converter treated it as 2 channels)
    data = write_recording(tmp_path / "rec", nvol=5, nchannels=1, channel_mask=2)
    dbs, _ = convert(make_db(tmp_path, [tmp_path / "rec"]))
    assert dbs[0]["nchannels"] == 1
    np.testing.assert_array_equal(read_bin(dbs[0]["reg_file"]), expected(data, 0, 0))


def test_multiplane_single_channel(tmp_path):
    # legacy converter reshaped single-channel data as plane-blocked; it is interleaved
    data = write_recording(tmp_path / "rec", nvol=5, nplanes=2, nflyback=2)
    dbs, _ = convert(make_db(tmp_path, [tmp_path / "rec"], batch_size=1))
    for j, d in enumerate(dbs):
        np.testing.assert_array_equal(read_bin(d["reg_file"]), expected(data, j, 0))


@pytest.mark.parametrize("layout", ["data_path", "look_one_level_down", "subfolders",
                                    "file_list"])
def test_multiple_recordings_are_concatenated(tmp_path, layout):
    root = tmp_path / "data"
    d1 = write_recording(root / "run1", nvol=4, nplanes=2, nchannels=2, seed=1)
    d2 = write_recording(root / "run2", nvol=3, nplanes=2, nchannels=2, seed=2)
    write_recording(root / "unused", nvol=3, nplanes=2, nchannels=2, seed=3)
    kw = {}
    if layout == "data_path":
        paths = [root / "run1", root / "run2"]
    elif layout == "look_one_level_down":
        (root / "unused" / "Image_0001_0001.raw").unlink()
        (root / "unused" / "Experiment.xml").unlink()
        paths, kw = [root], dict(look_one_level_down=True)
    elif layout == "subfolders":
        paths, kw = [root], dict(subfolders=["run1", "run2"])
    else:
        paths = [root]
        kw = dict(file_list=[os.path.join("run1", "Image_0001_0001.raw"),
                             os.path.join("run2", "Image_0001_0001.raw")])
    dbs, _ = convert(make_db(tmp_path, paths, **kw))

    for j, d in enumerate(dbs):
        assert d["nframes"] == 7
        np.testing.assert_array_equal(d["frames_per_file"], [4, 3])
        if layout != "file_list":  # file_list counts as a single folder
            np.testing.assert_array_equal(d["frames_per_folder"], [4, 3])
        for chan, key in [(0, "reg_file"), (1, "reg_file_chan2")]:
            np.testing.assert_array_equal(
                read_bin(d[key]),
                np.concatenate([expected(d1, j, chan), expected(d2, j, chan)]))
        assert [Path(m["raw_path"]).parent.name for m in d["raw_metadata"]] == ["run1", "run2"]


def test_stopped_recording_drops_incomplete_volume(tmp_path, caplog):
    # 5 complete volumes of 3 slices + 2 frames of a 6th; XML still declares 30 frames
    data = write_recording(tmp_path / "rec", nvol=6, nplanes=2, nflyback=1, frames=30,
                           status="Stopped", extra_bytes=-(LY * LX * 2))
    with caplog.at_level(logging.WARNING, logger="suite2p.io.raw"):
        dbs, _ = convert(make_db(tmp_path, [tmp_path / "rec"]))
    assert "incomplete volume" in caplog.text
    for j, d in enumerate(dbs):
        assert d["nframes"] == 5
        np.testing.assert_array_equal(read_bin(d["reg_file"]), expected(data[:5], j, 0))


# ---------------------------------------------------------------- metadata checks


def test_recorded_values_replace_defaults_and_conflicts_raise(tmp_path):
    write_recording(tmp_path / "rec", nplanes=3, nchannels=2, frame_rate=60.0)
    paths = [tmp_path / "rec"]

    def init(db_kw=None, fs=None):
        db = make_db(tmp_path, paths, **(db_kw or {}))
        db["file_list"], db["first_files"] = io.get_file_list(db)
        settings = suite2p.default_settings()
        if fs is not None:
            settings["fs"] = fs
        return io.raw_init_db(db, settings)

    db, settings = init()
    assert (db["nplanes"], db["nchannels"], settings["fs"]) == (3, 2, pytest.approx(20.0))
    db, settings = init(dict(nplanes=3, nchannels=2), fs=20.05)  # agrees within tolerance
    assert settings["fs"] == 20.05
    with pytest.raises(ValueError, match="nplanes"):
        init(dict(nplanes=2))
    with pytest.raises(ValueError, match="fs"):
        init(fs=60.0)

    write_recording(tmp_path / "rec1", nchannels=1)
    paths = [tmp_path / "rec1"]
    with pytest.raises(ValueError, match="nchannels"):
        init(dict(nchannels=2))
    with pytest.raises(ValueError, match="functional_chan"):
        init(dict(functional_chan=2))


@pytest.mark.parametrize("avg, nplanes, nflyback, expected_fs", [
    (dict(), 1, 0, 30.0),                                     # no averaging attributes
    (dict(average_mode=0, average_num=4), 1, 0, 30.0),        # averaging disabled
    (dict(average_mode=1, average_num=2), 1, 0, 15.0),        # saved at frameRate / 2
    (dict(average_mode=1, average_num=2), 2, 1, 5.0),         # / 2 averaged / 3 z-slices
])
def test_frame_averaging_sets_fs(tmp_path, avg, nplanes, nflyback, expected_fs):
    data = write_recording(tmp_path / "rec", nvol=4, nplanes=nplanes, nflyback=nflyback,
                           frame_rate=30.0, **avg)
    dbs, settings = convert(make_db(tmp_path, [tmp_path / "rec"]))
    assert settings["fs"] == pytest.approx(expected_fs)
    assert dbs[0]["nframes"] == 4  # averaging does not change the frame layout
    np.testing.assert_array_equal(read_bin(dbs[0]["reg_file"]), expected(data, 0, 0))


def test_unsupported_average_mode(tmp_path):
    write_recording(tmp_path / "rec", average_mode=2, average_num=2)
    with pytest.raises(ValueError, match="averageMode"):
        convert(make_db(tmp_path, [tmp_path / "rec"]))


def test_xml_overrides_for_unfinalized_xml(tmp_path):
    # XML written at start and never finalized: declares 15 averaged frames, but the
    # file holds 30 frames saved without averaging
    data = write_recording(tmp_path / "rec", nvol=30, frames=15, status="started",
                           frame_rate=30.0, average_mode=1, average_num=2)
    with pytest.raises(ValueError, match="extra data"):
        convert(make_db(tmp_path, [tmp_path / "rec"]))

    overrides = {"Streaming@frames": 30, "LSM@averageNum": 1}
    dbs, settings = convert(make_db(tmp_path, [tmp_path / "rec"], raw_xml_overrides=overrides))
    assert dbs[0]["nframes"] == 30 and settings["fs"] == pytest.approx(30.0)
    np.testing.assert_array_equal(read_bin(dbs[0]["reg_file"]), expected(data, 0, 0))
    meta = np.load(dbs[0]["db_path"], allow_pickle=True).item()["raw_metadata"][0]
    assert meta["xml_overrides"] == {"Streaming@frames": {"xml": "15", "used": "30"},
                                     "LSM@averageNum": {"xml": "2", "used": "1"}}
    assert meta["declared_frames"] == 30 and meta["average_num"] == 1

    with pytest.raises(ValueError, match="unknown RAW XML override"):
        convert(make_db(tmp_path, [tmp_path / "rec"],
                        raw_xml_overrides={"Streaming@frame": 30}))
    with pytest.raises(ValueError, match="must be a dict"):
        convert(make_db(tmp_path, [tmp_path / "rec"], raw_xml_overrides=[30]))


@pytest.mark.parametrize("db_kw", [dict(lines=[[0, 1, 2]], dy=[0], dx=[0]),
                                   dict(swap_order=True), dict(nrois=2)])
def test_unsupported_options_raise(tmp_path, db_kw):
    write_recording(tmp_path / "rec")
    db = make_db(tmp_path, [tmp_path / "rec"], **db_kw)
    db["file_list"], db["first_files"] = io.get_file_list(db)
    with pytest.raises(ValueError, match="not supported"):
        io.raw_init_db(db, suite2p.default_settings())


@pytest.mark.parametrize("field, value", [("Ly", 4), ("nchannels", 2), ("nplanes", 2),
                                          ("frame_rate", 15.0), ("channel_mask", 2),
                                          ("average_num", 2)])
def test_incompatible_recordings_raise(tmp_path, field, value):
    write_recording(tmp_path / "a")
    kw = dict(nchannels=1, nplanes=1, frame_rate=30.0, channel_mask=1)
    if field == "Ly":
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "Image_0001_0001.raw").write_bytes(bytes(10 * 4 * LX * 2))
        (tmp_path / "b" / "Experiment.xml").write_text(make_xml(Ly=4, frames=10))
    else:
        kw[field] = value
        if field == "nchannels":
            kw["channel_mask"] = 3
        if field == "average_num":
            kw["average_mode"] = 1
        write_recording(tmp_path / "b", **kw)
    db = make_db(tmp_path, [tmp_path / "a", tmp_path / "b"])
    with pytest.raises(ValueError, match="incompatible"):
        convert(db)


def test_missing_xml(tmp_path):
    write_recording(tmp_path / "rec", xml_name=None)
    with pytest.raises(FileNotFoundError, match="Experiment.xml"):
        convert(make_db(tmp_path, [tmp_path / "rec"]))


def test_ambiguous_xml_and_raw_layouts(tmp_path):
    write_recording(tmp_path / "rec", xml_name="a.xml")
    (tmp_path / "rec" / "b.xml").write_text(make_xml())
    with pytest.raises(ValueError, match="ambiguous metadata"):
        convert(make_db(tmp_path, [tmp_path / "rec"]))

    write_recording(tmp_path / "rec2")
    write_recording(tmp_path / "rec2", raw_name="Image_0002_0001.raw")
    with pytest.raises(ValueError, match="ambiguous RAW layout"):
        convert(make_db(tmp_path, [tmp_path / "rec2"]))
    # selecting one file disambiguates
    dbs, _ = convert(make_db(tmp_path, [tmp_path / "rec2"],
                             file_list=["Image_0002_0001.raw"]))
    assert dbs[0]["nframes"] == 7


def test_one_xml_without_standard_name_is_used(tmp_path):
    write_recording(tmp_path / "rec", xml_name="MyExperiment.xml")
    dbs, _ = convert(make_db(tmp_path, [tmp_path / "rec"]))
    assert dbs[0]["nframes"] == 7


@pytest.mark.parametrize("xml_text, match", [
    ("<ThorImageExperiment><LSM pixelX='5'", "could not parse"),
    ("<Other />", "ThorImageExperiment"),
    (make_xml().replace('pixelY="3"', ""), "pixelY"),
    (make_xml().replace('frameRate="30.0"', 'frameRate="fast"'), "frameRate"),
    (make_xml().replace("<Streaming", "<NotStreaming"), "Streaming"),
    (make_xml(nplanes=2).replace("<ZStage", "<NoZStage"), "ZStage"),
    (make_xml(channel_mask=0), "channel mask"),
    (make_xml(channel_mask=7), "at most 2 channels"),
    (make_xml(channel_mask=2, wavelengths=["A", "B"]), "Wavelength"),
])
def test_malformed_or_unsupported_xml(tmp_path, xml_text, match):
    write_recording(tmp_path / "rec")
    (tmp_path / "rec" / "Experiment.xml").write_text(xml_text)
    with pytest.raises(ValueError, match=match):
        convert(make_db(tmp_path, [tmp_path / "rec"]))


@pytest.mark.parametrize("kw, match", [
    (dict(extra_bytes=-3), "not a whole number"),          # partial frame
    (dict(extra_bytes=-(LY * LX * 2)), "truncated"),       # missing frame(s)
    (dict(extra_bytes=LY * LX * 2), "extra data"),         # more frames than declared
    (dict(nplanes=2, nflyback=1, frames=8, nvol=3, extra_bytes=-(LY * LX * 2)),
     "whole number of volumes"),                           # frames agree, volume incomplete
    (dict(status="Stopped", nvol=1, nplanes=2, extra_bytes=-(LY * LX * 2)),
     "no complete volume"),
])
def test_truncated_or_malformed_raw(tmp_path, kw, match):
    write_recording(tmp_path / "rec", **kw)
    with pytest.raises(ValueError, match=match):
        convert(make_db(tmp_path, [tmp_path / "rec"]))


def test_missing_xmltodict(tmp_path, monkeypatch):
    write_recording(tmp_path / "rec")
    monkeypatch.setattr(raw_mod, "HAS_XMLTODICT", False)
    db = make_db(tmp_path, [tmp_path / "rec"])
    with pytest.raises(ImportError, match="xmltodict"):
        convert(db)
    # through run_s2p the error happens before any plane folder is created
    with pytest.raises(ImportError, match="xmltodict"):
        suite2p.run_s2p(db=make_db(tmp_path, [tmp_path / "rec"]),
                        settings=suite2p.default_settings())
    assert not list((tmp_path / "out" / "suite2p").glob("plane*"))


def test_converter_rejects_dbs_not_matching_metadata(tmp_path):
    write_recording(tmp_path / "rec", nplanes=2)
    db = make_db(tmp_path, [tmp_path / "rec"])
    db["file_list"], db["first_files"] = io.get_file_list(db)
    dbs = io.init_dbs(db)  # skipped raw_init_db -> 1 plane
    with open(dbs[0]["reg_file"], "wb") as f:
        with pytest.raises(ValueError, match="raw_init_db"):
            io.raw_to_binary(dbs, suite2p.default_settings(), [f], None)


# ---------------------------------------------------------------- run_s2p integration


def test_dispatch_registered():
    assert run_s2p_mod.files_to_binary["raw"] is io.raw_to_binary
    assert run_s2p_mod.init_db_from_metadata["raw"] is io.raw_init_db
    assert "'raw'" in suite2p.parameters.DB["input_format"]["description"]
    assert io.utils.EXTS["raw"]


def test_unknown_input_format(tmp_path):
    db = make_db(tmp_path, [tmp_path], input_format="thorraw")
    with pytest.raises(ValueError, match="unknown input_format"):
        suite2p.run_s2p(db=db, settings=suite2p.default_settings())


@pytest.fixture
def run_plane_calls(monkeypatch):
    calls = []

    def fake_run_plane(db, settings, db_path=None, stat=None):
        calls.append(dict(db=db, settings=settings, db_path=db_path))

    monkeypatch.setattr(run_s2p_mod, "run_plane", fake_run_plane)
    return calls


def test_run_s2p_conversion_path(tmp_path, run_plane_calls):
    root = tmp_path / "data"
    d1 = write_recording(root / "run1", nvol=4, nplanes=2, nflyback=1, nchannels=2,
                         frame_rate=45.0, seed=1)
    d2 = write_recording(root / "run2", nvol=5, nplanes=2, nflyback=1, nchannels=2,
                         frame_rate=45.0, seed=2)
    fast_disk = tmp_path / "fast"
    db = suite2p.default_db()
    db.update(data_path=[str(root)], look_one_level_down=True, input_format="raw",
              save_path0=str(tmp_path / "out"), save_folder="s2p_raw",
              fast_disk=str(fast_disk), keep_movie_raw=True, functional_chan=2)
    settings = suite2p.default_settings()
    settings["io"]["combined"] = False

    db_paths = suite2p.run_s2p(db=db, settings=settings)

    save_folder = tmp_path / "out" / "s2p_raw"
    assert [Path(p) for p in db_paths] == [save_folder / f"plane{j}" / "db.npy"
                                           for j in range(2)]
    assert len(run_plane_calls) == 2
    top_settings = np.load(save_folder / "settings.npy", allow_pickle=True).item()
    top_db = np.load(save_folder / "db.npy", allow_pickle=True).item()
    assert top_settings["fs"] == pytest.approx(15.0)
    assert (top_db["nplanes"], top_db["nchannels"]) == (2, 2)

    for j, call in enumerate(run_plane_calls):
        d = call["db"]
        assert call["settings"]["fs"] == pytest.approx(15.0)  # what the pipeline gets
        assert Path(call["db_path"]) == save_folder / f"plane{j}" / "db.npy"
        assert d["iplane"] == j and d["nframes"] == 9 and (d["Ly"], d["Lx"]) == (LY, LX)
        # keep_movie_raw: unregistered movie written to data_raw.bin on fast_disk
        plane_fast = fast_disk / "suite2p" / f"plane{j}"
        assert Path(d["raw_file"]) == plane_fast / "data_raw.bin"
        assert Path(d["raw_file_chan2"]) == plane_fast / "data_chan2_raw.bin"
        assert not Path(d["reg_file"]).exists()
        np.testing.assert_array_equal(
            read_bin(d["raw_file"]), np.concatenate([expected(d1, j, 1), expected(d2, j, 1)]))
        np.testing.assert_array_equal(
            read_bin(d["raw_file_chan2"]),
            np.concatenate([expected(d1, j, 0), expected(d2, j, 0)]))
        np.testing.assert_array_equal(d["frames_per_file"], [4, 5])
        np.testing.assert_allclose(d["meanImg"], read_bin(d["raw_file"]).mean(axis=0),
                                   rtol=1e-6)
        plane_settings = np.load(save_folder / f"plane{j}" / "settings.npy",
                                 allow_pickle=True).item()
        assert plane_settings["fs"] == pytest.approx(15.0)


def test_run_s2p_rerun_on_existing_binaries_keeps_recorded_fs(tmp_path, run_plane_calls):
    write_recording(tmp_path / "rec", nvol=4, nplanes=2, frame_rate=50.0)
    db = make_db(tmp_path, [tmp_path / "rec"])
    settings = suite2p.default_settings()
    settings["io"]["combined"] = False
    suite2p.run_s2p(db=dict(db), settings=settings)
    assert [c["settings"]["fs"] for c in run_plane_calls] == [pytest.approx(25.0)] * 2

    # without the RAW file, a re-run can only succeed from the existing binaries
    (tmp_path / "rec" / "Image_0001_0001.raw").unlink()
    run_plane_calls.clear()
    suite2p.run_s2p(db=dict(db), settings=settings)
    assert [c["settings"]["fs"] for c in run_plane_calls] == [pytest.approx(25.0)] * 2
    plane_settings = np.load(tmp_path / "out" / "suite2p" / "plane1" / "settings.npy",
                             allow_pickle=True).item()
    assert plane_settings["fs"] == pytest.approx(25.0)

    settings["fs"] = 30.0
    with pytest.raises(ValueError, match="fs"):
        suite2p.run_s2p(db=dict(db), settings=settings)
