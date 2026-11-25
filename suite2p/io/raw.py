"""
Thorlabs ThorImageLS RAW input.

Originally contributed by Yael Prilutski (Yoav Livneh Lab, 2023); refactored for the
``(dbs, settings, reg_file, reg_file_chan2)`` converter interface used by ``run_s2p``.

Layout assumptions
------------------
A ThorImageLS experiment folder holds one streamed ``*.raw`` file and its
``Experiment.xml``. The RAW file is headerless little-endian uint16. Each image is
``pixelY`` rows by ``pixelX`` columns in row-major order. Images are ordered
time -> z-slice (including flyback slices, which come last in each volume) ->
channel, i.e. the RAW stream has shape
``(nvolumes, nplanes + nflyback, nchannels, Ly, Lx)``. This is the same
time/plane/channel interleaving that suite2p expects for TIFFs (see
``docs/inputs.md``, "Frame ordering", and the stride logic in ``tiff_to_binary``).
Flyback slices are discarded.

Pixel values are converted to int16 the same way as uint16 TIFF / sbx inputs
(integer division by 2).
"""
import glob
import logging
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from ..parameters import default_db, default_settings

logger = logging.getLogger(__name__)

try:
    import xmltodict
    HAS_XMLTODICT = True
except ImportError:
    xmltodict = None
    HAS_XMLTODICT = False

RAW_DTYPE = np.dtype("<u2")
XML_NAME = "Experiment.xml"
# relative tolerance when comparing a user-provided sampling rate to the recorded one
FS_RTOL = 1e-2


def _require_xmltodict():
    if not HAS_XMLTODICT:
        raise ImportError("xmltodict is required for Thorlabs RAW files, "
                          "please 'pip install xmltodict' (or 'pip install suite2p[io]')")


@dataclass
class RawRecording:
    """Metadata of a single ThorImageLS RAW recording, validated against its file size."""
    raw_path: str
    xml_path: str
    Ly: int
    Lx: int
    nchannels: int
    channel_mask: int
    nplanes: int
    nflyback: int
    frame_rate: float
    declared_frames: int
    file_frames: int
    nvolumes: int
    stopped: bool
    average_num: int = 1
    width_um: float = None
    height_um: float = None
    xml_overrides: dict = None

    @property
    def recorded_planes(self):
        """Number of z-slices per volume in the RAW stream, including flyback."""
        return self.nplanes + self.nflyback

    @property
    def bytes_per_volume(self):
        return (self.recorded_planes * self.nchannels * self.Ly * self.Lx *
                RAW_DTYPE.itemsize)

    @property
    def fs(self):
        """Sampling rate per plane (volume rate) of the saved frames."""
        return recorded_fs(self.to_dict())

    def config(self):
        """Acquisition parameters that must match to concatenate recordings."""
        return dict(Ly=self.Ly, Lx=self.Lx, nchannels=self.nchannels,
                    channel_mask=self.channel_mask, nplanes=self.nplanes,
                    nflyback=self.nflyback, frame_rate=self.frame_rate,
                    average_num=self.average_num)

    def to_dict(self):
        return asdict(self)


def recorded_fs(meta):
    """
    Per-plane sampling rate of the saved frames from RAW metadata (a
    ``RawRecording.to_dict()``): scanner frame rate / frames averaged / z-slices
    per volume (planes + flyback).
    """
    return (meta["frame_rate"] / meta.get("average_num", 1) /
            (meta["nplanes"] + meta["nflyback"]))


# XML attributes that db["raw_xml_overrides"] may replace, as "Element@attribute"
OVERRIDABLE_XML_ATTRS = (
    "LSM@pixelX", "LSM@pixelY", "LSM@channel", "LSM@frameRate", "LSM@averageMode",
    "LSM@averageNum", "Streaming@frames", "Streaming@zFastEnable",
    "Streaming@flybackFrames", "ZStage@steps", "ExperimentStatus@value",
)


def _apply_xml_overrides(root, overrides, xml_path):
    """Replace attributes of the parsed XML; returns {key: {"xml": old, "used": new}}."""
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("db['raw_xml_overrides'] must be a dict like "
                         "{'Streaming@frames': 30000}")
    applied = {}
    for key, value in (overrides or {}).items():
        if key not in OVERRIDABLE_XML_ATTRS:
            raise ValueError(f"unknown RAW XML override {key!r}; must be one of "
                             f"{list(OVERRIDABLE_XML_ATTRS)}")
        elem_name, attr = key.split("@")
        elem = root.get(elem_name)
        if elem is None:
            elem = root[elem_name] = {}
        if not isinstance(elem, dict):
            raise ValueError(f"{xml_path}: cannot override {key}: <{elem_name}> is not a "
                             "single element with attributes")
        old = elem.get(f"@{attr}")
        elem[f"@{attr}"] = str(value)
        applied[key] = {"xml": old, "used": str(value)}
        logger.warning(f"{xml_path}: overriding <{elem_name} {attr}=\"{old}\"> with "
                       f"\"{value}\" (db['raw_xml_overrides'])")
    return applied


def _get_element(root, name, xml_path):
    elem = root.get(name)
    if elem is None:
        raise ValueError(f"{xml_path}: missing <{name}> element")
    if isinstance(elem, list):
        raise ValueError(f"{xml_path}: expected one <{name}> element, found {len(elem)}")
    if not isinstance(elem, dict):
        raise ValueError(f"{xml_path}: <{name}> element has no attributes")
    return elem


def _get_attr(elem, elem_name, attr, cast, xml_path):
    value = elem.get(f"@{attr}")
    if value is None:
        raise ValueError(f"{xml_path}: missing attribute <{elem_name} {attr}=...>")
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{xml_path}: invalid value {value!r} for <{elem_name} {attr}=...>")


def _to_int(value):
    fvalue = float(value)
    if not fvalue.is_integer():
        raise ValueError(value)
    return int(fvalue)


def parse_experiment_xml(xml_path, overrides=None):
    """
    Parse the acquisition parameters needed for conversion from a ThorImageLS
    ``Experiment.xml``.

    Parameters
    ----------
    xml_path : str
        Path to the XML file.
    overrides : dict, optional
        Attribute values that replace the XML's, keyed "Element@attribute"
        (see ``OVERRIDABLE_XML_ATTRS``), e.g. ``{"Streaming@frames": 30000}``
        for an XML that was never finalized. Each override is logged and
        recorded in the returned "xml_overrides".

    Returns
    -------
    meta : dict
        Keys "Ly", "Lx", "nchannels", "channel_mask", "nplanes", "nflyback",
        "frame_rate", "average_num", "declared_frames", "stopped", "width_um",
        "height_um", "xml_overrides".

    Raises
    ------
    ImportError
        If xmltodict is not installed.
    ValueError
        If the XML is malformed, required fields are missing or invalid, or the
        recording configuration is not supported.
    """
    _require_xmltodict()
    try:
        with open(xml_path, "rb") as f:
            xml = xmltodict.parse(f.read())
    except Exception as e:  # expat errors are not a stable public type
        raise ValueError(f"{xml_path}: could not parse XML ({e})") from e

    root = xml.get("ThorImageExperiment") if isinstance(xml, dict) else None
    if not isinstance(root, dict):
        raise ValueError(f"{xml_path}: not a ThorImageLS experiment file "
                         "(missing <ThorImageExperiment> root)")
    applied = _apply_xml_overrides(root, overrides, xml_path)

    lsm = _get_element(root, "LSM", xml_path)
    Lx = _get_attr(lsm, "LSM", "pixelX", _to_int, xml_path)
    Ly = _get_attr(lsm, "LSM", "pixelY", _to_int, xml_path)
    channel_mask = _get_attr(lsm, "LSM", "channel", _to_int, xml_path)
    frame_rate = _get_attr(lsm, "LSM", "frameRate", float, xml_path)
    width_um = float(lsm["@widthUM"]) if "@widthUM" in lsm else None
    height_um = float(lsm["@heightUM"]) if "@heightUM" in lsm else None
    if Lx < 1 or Ly < 1:
        raise ValueError(f"{xml_path}: invalid image size pixelX={Lx}, pixelY={Ly}")
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"{xml_path}: invalid frameRate={frame_rate}")

    # frame averaging: averageMode 0 = none, 1 = averageNum scanner frames are
    # averaged into each saved frame, so frames are saved at frameRate / averageNum
    average_mode = (_get_attr(lsm, "LSM", "averageMode", _to_int, xml_path)
                    if "@averageMode" in lsm else 0)
    if average_mode == 0:
        average_num = 1
    elif average_mode == 1:
        average_num = _get_attr(lsm, "LSM", "averageNum", _to_int, xml_path)
        if average_num < 1:
            raise ValueError(f"{xml_path}: invalid averageNum={average_num}")
    else:
        raise ValueError(f"{xml_path}: unsupported LSM averageMode={average_mode} "
                         "(only 0 = none and 1 = frame averaging are supported)")

    # LSM "channel" is a bitmask of enabled channels (A=1, B=2, C=4, D=8)
    if channel_mask < 1 or channel_mask > 15:
        raise ValueError(f"{xml_path}: invalid LSM channel mask {channel_mask}")
    nchannels = bin(channel_mask).count("1")
    if nchannels > 2:
        raise ValueError(f"{xml_path}: {nchannels} channels enabled (LSM channel="
                         f"{channel_mask}); suite2p supports at most 2 channels")
    # cross-check against the listed wavelengths when present
    wavelengths = root.get("Wavelengths")
    if isinstance(wavelengths, dict) and "Wavelength" in wavelengths:
        wl = wavelengths["Wavelength"]
        n_wl = len(wl) if isinstance(wl, list) else 1
        if n_wl != nchannels:
            raise ValueError(f"{xml_path}: LSM channel mask {channel_mask} implies "
                             f"{nchannels} channel(s) but {n_wl} <Wavelength> entries "
                             "are listed; cannot determine channel layout")

    streaming = _get_element(root, "Streaming", xml_path)
    declared_frames = _get_attr(streaming, "Streaming", "frames", _to_int, xml_path)
    zfast = _get_attr(streaming, "Streaming", "zFastEnable", _to_int, xml_path)
    if zfast:
        nflyback = _get_attr(streaming, "Streaming", "flybackFrames", _to_int, xml_path)
        zstage = _get_element(root, "ZStage", xml_path)
        nplanes = _get_attr(zstage, "ZStage", "steps", _to_int, xml_path)
        if nplanes < 1 or nflyback < 0:
            raise ValueError(f"{xml_path}: invalid fast-z configuration "
                             f"(ZStage steps={nplanes}, flybackFrames={nflyback})")
    else:
        nplanes, nflyback = 1, 0
    if declared_frames < 1:
        raise ValueError(f"{xml_path}: invalid Streaming frames={declared_frames}")

    status = root.get("ExperimentStatus")
    stopped = isinstance(status, dict) and str(status.get("@value", "")).lower() == "stopped"

    return dict(Ly=Ly, Lx=Lx, nchannels=nchannels, channel_mask=channel_mask,
                nplanes=nplanes, nflyback=nflyback, frame_rate=frame_rate,
                average_num=average_num, declared_frames=declared_frames,
                stopped=stopped, width_um=width_um, height_um=height_um,
                xml_overrides=applied or None)


def find_experiment_xml(raw_path):
    """
    Find the ThorImageLS XML file describing a RAW file (in the same folder).

    ``Experiment.xml`` is used if present, otherwise the folder must contain
    exactly one ``*.xml`` file.
    """
    folder = os.path.dirname(os.path.abspath(raw_path))
    xmls = [f for f in os.listdir(folder) if f.lower().endswith(".xml")]
    exp = [f for f in xmls if f.lower() == XML_NAME.lower()]
    if len(exp) == 1:
        return os.path.join(folder, exp[0])
    if len(xmls) == 1:
        return os.path.join(folder, xmls[0])
    if len(xmls) == 0:
        raise FileNotFoundError(f"no {XML_NAME} found next to RAW file {raw_path}")
    raise ValueError(f"ambiguous metadata for RAW file {raw_path}: no {XML_NAME} and "
                     f"multiple XML files in folder {sorted(xmls)}")


def read_raw_recording(raw_path, xml_overrides=None):
    """
    Read the XML metadata of a RAW file and validate it against the file size.

    Parameters
    ----------
    raw_path : str
        Path to the ThorImageLS ``*.raw`` file.
    xml_overrides : dict, optional
        XML attribute overrides, see ``parse_experiment_xml``.

    Returns
    -------
    rec : RawRecording

    Raises
    ------
    ValueError
        If the RAW file size is inconsistent with the metadata (truncated,
        partial images, extra data, or no complete volume).
    """
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"RAW file not found: {raw_path}")
    xml_path = find_experiment_xml(raw_path)
    meta = parse_experiment_xml(xml_path, xml_overrides)
    nbytes = os.path.getsize(raw_path)
    image_bytes = meta["Ly"] * meta["Lx"] * meta["nchannels"] * RAW_DTYPE.itemsize
    if nbytes % image_bytes != 0:
        raise ValueError(
            f"{raw_path}: size {nbytes} bytes is not a whole number of "
            f"{meta['Ly']}x{meta['Lx']}x{meta['nchannels']}-channel uint16 frames "
            "(truncated or malformed RAW file, or XML does not match)")
    file_frames = nbytes // image_bytes
    recorded_planes = meta["nplanes"] + meta["nflyback"]
    nvolumes = file_frames // recorded_planes

    if not meta["stopped"]:
        if file_frames != meta["declared_frames"]:
            kind = "truncated" if file_frames < meta["declared_frames"] else "has extra data"
            raise ValueError(
                f"{raw_path}: RAW file {kind}: {xml_path} declares "
                f"{meta['declared_frames']} frames but the file contains {file_frames}")
        if file_frames % recorded_planes != 0:
            raise ValueError(
                f"{raw_path}: {file_frames} frames is not a whole number of volumes of "
                f"{recorded_planes} z-slices ({meta['nplanes']} planes + "
                f"{meta['nflyback']} flyback)")
    elif file_frames % recorded_planes != 0:
        logger.warning(f"{raw_path}: recording was stopped early; dropping "
                       f"{file_frames % recorded_planes} frames of an incomplete volume")
    if nvolumes < 1:
        raise ValueError(f"{raw_path}: no complete volume in RAW file "
                         f"({file_frames} frames, {recorded_planes} z-slices per volume)")

    return RawRecording(raw_path=os.path.abspath(raw_path), xml_path=xml_path,
                        file_frames=file_frames, nvolumes=nvolumes, **meta)


def find_raw_recordings(file_list, xml_overrides=None):
    """
    Load and cross-validate the metadata of all RAW files to convert.

    Parameters
    ----------
    file_list : list of str
        RAW files (as found by ``io.get_file_list``), one per ThorImageLS
        experiment folder.
    xml_overrides : dict, optional
        XML attribute overrides applied to every file, see
        ``parse_experiment_xml``.

    Returns
    -------
    recs : list of RawRecording
        One per file, in the given order.

    Raises
    ------
    ValueError
        If no files are given, a file is not a RAW file, two files share a
        folder (and hence an XML), or recordings have incompatible
        configurations.
    """
    _require_xmltodict()
    if file_list is None or len(file_list) == 0:
        raise ValueError("no RAW files to convert")
    folders = {}
    for f in file_list:
        if not str(f).lower().endswith(".raw"):
            raise ValueError(f"not a RAW file: {f}")
        folder = os.path.normcase(os.path.dirname(os.path.abspath(f)))
        folders.setdefault(folder, []).append(os.path.basename(f))
    shared = {k: v for k, v in folders.items() if len(v) > 1}
    if shared:
        raise ValueError(
            "ambiguous RAW layout: each ThorImageLS experiment folder must contain one "
            f"RAW recording, found several in {shared}. Select one per folder with "
            "db['file_list'] or move recordings to separate folders.")

    recs = [read_raw_recording(str(f), xml_overrides) for f in file_list]
    cfg0 = recs[0].config()
    for rec in recs[1:]:
        cfg = rec.config()
        diff = {k: (cfg0[k], cfg[k]) for k in cfg0
                if not (np.isclose(cfg0[k], cfg[k], rtol=1e-6, atol=0)
                        if k == "frame_rate" else cfg0[k] == cfg[k])}
        if diff:
            raise ValueError(
                f"RAW recordings have incompatible configurations and cannot be "
                f"concatenated: {recs[0].raw_path} vs {rec.raw_path} differ in "
                + ", ".join(f"{k} ({a} != {b})" for k, (a, b) in diff.items()))
    return recs


def _reconcile(name, current, recorded, default, where="db"):
    if current == recorded:
        return recorded
    if current == default:
        logger.info(f"{where}['{name}'] set to {recorded} from RAW metadata")
        return recorded
    raise ValueError(f"{where}['{name}']={current} disagrees with the RAW metadata "
                     f"({recorded}); remove it or set it to {recorded}")


def reconcile_fs(settings, recorded_fs):
    """
    Use the recorded per-plane sampling rate unless the user set a different
    (non-default) one, in which case raise.
    """
    current = settings.get("fs", None)
    if current is not None and np.isclose(current, recorded_fs, rtol=FS_RTOL, atol=0):
        return settings
    if current is None or current == default_settings()["fs"]:
        logger.info(f"settings['fs'] set to {recorded_fs:.4g} Hz (per plane) "
                    "from RAW metadata")
        settings["fs"] = float(recorded_fs)
        return settings
    raise ValueError(f"settings['fs']={current} disagrees with the RAW metadata "
                     f"({recorded_fs:.4g} Hz per plane); remove it or set it to the "
                     "recorded rate")


def raw_reconcile_saved_settings(db, settings):
    """
    Reconcile ``settings["fs"]`` with the RAW metadata stored in an existing
    plane db (used by ``run_s2p`` when re-running on existing binaries).
    """
    return reconcile_fs(settings, recorded_fs(db["raw_metadata"][0]))


def raw_init_db(db, settings):
    """
    Read RAW/XML metadata and set up ``db`` and ``settings`` before per-plane dbs
    and binary files are created.

    Sets ``db["nplanes"]``, ``db["nchannels"]`` and ``settings["fs"]`` from the
    recordings. A value left at its default is replaced by the recorded value; a
    non-default value that disagrees with the recording raises ``ValueError``.
    Stores the per-file metadata in ``db["raw_metadata"]``.

    Parameters
    ----------
    db : dict
        Must contain "file_list" (from ``io.get_file_list``). Optional
        "raw_xml_overrides" ({"Element@attribute": value}) replaces XML values
        for every file, e.g. for an XML that was never finalized.
    settings : dict
        Pipeline settings; "fs" may be updated.

    Returns
    -------
    db, settings : dict
        The updated dictionaries.
    """
    _require_xmltodict()
    if db.get("lines") is not None and len(db["lines"]) > 0:
        raise ValueError("db['lines'] (multi-ROI) is not supported for RAW input")
    if db.get("nrois", 1) not in (None, 1):
        raise ValueError("db['nrois'] > 1 is not supported for RAW input")
    if db.get("swap_order", False):
        raise ValueError("db['swap_order'] is not supported for RAW input "
                         "(channel/plane order is read from the RAW metadata)")

    recs = find_raw_recordings(db["file_list"], db.get("raw_xml_overrides"))
    rec = recs[0]
    defaults = default_db()
    db["nplanes"] = _reconcile("nplanes", db.get("nplanes"), rec.nplanes,
                               defaults["nplanes"])
    db["nchannels"] = _reconcile("nchannels", db.get("nchannels"), rec.nchannels,
                                 defaults["nchannels"])
    functional_chan = db.get("functional_chan", defaults["functional_chan"])
    if functional_chan not in range(1, rec.nchannels + 1):
        raise ValueError(f"db['functional_chan']={functional_chan} is invalid for a "
                         f"recording with {rec.nchannels} channel(s)")
    reconcile_fs(settings, rec.fs)
    db["raw_metadata"] = [r.to_dict() for r in recs]
    logger.info(f"RAW: {len(recs)} recording(s), {rec.nplanes} plane(s) "
                f"(+{rec.nflyback} flyback discarded), {rec.nchannels} channel(s), "
                f"{rec.Ly}x{rec.Lx} px, {rec.fs:.4g} Hz per plane")
    return db, settings


def raw_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """
    Convert ThorImageLS RAW files to per-plane suite2p binaries.

    The per-plane ``dbs`` must have been created for the recorded number of
    planes and channels, i.e. after ``raw_init_db`` (``run_s2p`` does this).

    Parameters
    ----------
    dbs : list of dict
        Per-plane database dictionaries. Must contain "file_list",
        "first_files", "batch_size", "nplanes", "nchannels" and
        "functional_chan". Updated in-place with "Ly", "Lx", "nframes",
        "frames_per_file", "frames_per_folder", "meanImg" and (two channels)
        "meanImg_chan2", then saved to "db_path" with ``settings`` to
        "settings_path".
    settings : dict
        Suite2p settings dictionary, saved alongside each plane's database.
    reg_file : list of file objects
        Opened binary files for each plane's functional channel.
    reg_file_chan2 : list of file objects or None
        Opened binary files for each plane's second channel (two channels only).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries.

    Raises
    ------
    ValueError
        If the RAW metadata does not match ``dbs`` or the data is invalid.
    """
    t0 = time.time()
    fs = dbs[0]["file_list"]
    first_files = np.asarray(dbs[0]["first_files"], dtype=bool)
    recs = find_raw_recordings(fs, dbs[0].get("raw_xml_overrides"))
    rec0 = recs[0]
    nplanes, nchannels = dbs[0]["nplanes"], dbs[0]["nchannels"]

    if (nplanes, nchannels) != (rec0.nplanes, rec0.nchannels) or len(dbs) != nplanes:
        raise ValueError(
            f"RAW metadata ({rec0.nplanes} planes, {rec0.nchannels} channels) does not "
            f"match the per-plane dbs ({len(dbs)} dbs, nplanes={nplanes}, "
            f"nchannels={nchannels}); call io.raw_init_db before io.init_dbs")
    if len(reg_file) != nplanes:
        raise ValueError(f"expected {nplanes} output files, got {len(reg_file)}")
    if nchannels > 1 and (reg_file_chan2 is None or len(reg_file_chan2) != nplanes):
        raise ValueError(f"two-channel recording needs {nplanes} channel 2 output files")
    if len(first_files) != len(fs) or not first_files[0]:
        raise ValueError("db['first_files'] does not match db['file_list']")
    nfunc = dbs[0]["functional_chan"] - 1 if nchannels > 1 else 0
    if nfunc not in range(nchannels):
        raise ValueError(f"db['functional_chan']={dbs[0]['functional_chan']} is invalid "
                         f"for a recording with {nchannels} channel(s)")

    Ly, Lx = rec0.Ly, rec0.Lx
    folder_ids = np.cumsum(first_files) - 1
    for db in dbs:
        db["Ly"], db["Lx"] = Ly, Lx
        db["nframes"] = 0
        db["frames_per_file"] = np.zeros(len(fs), "int")
        db["frames_per_folder"] = np.zeros(int(first_files.sum()), "int")
        db["meanImg"] = np.zeros((Ly, Lx), "float64")
        if nchannels > 1:
            db["meanImg_chan2"] = np.zeros((Ly, Lx), "float64")

    nzc = rec0.recorded_planes * nchannels
    vols_per_batch = max(1, math.ceil(dbs[0]["batch_size"] / nzc))
    ntotal = 0
    for ifile, rec in enumerate(recs):
        logger.info(f"converting RAW file {rec.raw_path} ({rec.nvolumes} volumes)")
        with open(rec.raw_path, "rb") as f:
            iv = 0
            while iv < rec.nvolumes:
                nv = min(vols_per_batch, rec.nvolumes - iv)
                buf = f.read(nv * rec.bytes_per_volume)
                if len(buf) != nv * rec.bytes_per_volume:
                    raise ValueError(f"{rec.raw_path}: unexpected end of file at volume "
                                     f"{iv} (file changed during conversion?)")
                data = np.frombuffer(buf, dtype=RAW_DTYPE).reshape(
                    nv, rec.recorded_planes, nchannels, Ly, Lx)
                # same uint16 -> int16 conversion as tiff/sbx inputs
                data = (data[:, :nplanes] // 2).astype(np.int16)
                for j in range(nplanes):
                    db = dbs[j]
                    im = np.ascontiguousarray(data[:, j, nfunc])
                    reg_file[j].write(bytearray(im))
                    db["meanImg"] += im.sum(axis=0, dtype="float64")
                    if nchannels > 1:
                        im2 = np.ascontiguousarray(data[:, j, 1 - nfunc])
                        reg_file_chan2[j].write(bytearray(im2))
                        db["meanImg_chan2"] += im2.sum(axis=0, dtype="float64")
                    db["nframes"] += nv
                    db["frames_per_file"][ifile] += nv
                    db["frames_per_folder"][folder_ids[ifile]] += nv
                iv += nv
                ntotal += nv
        logger.info(f"{ntotal} frames per plane written, time {time.time() - t0:0.2f} sec.")

    for db in dbs:
        db["meanImg"] = (db["meanImg"] / db["nframes"]).astype("float32")
        if nchannels > 1:
            db["meanImg_chan2"] = (db["meanImg_chan2"] / db["nframes"]).astype("float32")
        db["raw_metadata"] = [r.to_dict() for r in recs]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)
    return dbs
