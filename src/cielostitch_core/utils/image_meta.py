#!/usr/bin/env python3
"""
Image Metadata Handler

Provides functionality to display, read, copy, overwrite, replace, and merge
metadata across different image formats.

Supported formats:
  - TIFF (.tif, .tiff): Full metadata support via tifffile
  - FITS (.fit, .fits): Full metadata support via astropy.io.fits
  - XISF (.xisf): Full metadata support via xisf
  - PNG, JPG/JPEG, BMP, WebP: Basic metadata via OpenCV

Dependencies:
  - tifffile: TIFF reading/writing
  - numpy: Array operations
  - cv2 (opencv-python-headless): Standard image format support
  - astropy: FITS support
  - xisf: XISF support
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging
import re

import numpy as np
import cv2
import tifffile

try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False

try:
    from xisf import XISF
    HAS_XISF = True
except ImportError:
    HAS_XISF = False


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff",
    ".png",
    ".jpg", ".jpeg",
    ".bmp",
    ".webp",
    ".fit", ".fits",
    ".xisf",
}

TIFF_PROTECTED_TAG_IDS = {
    256,  # ImageWidth
    257,  # ImageLength
    258,  # BitsPerSample
    262,  # PhotometricInterpretation
    277,  # SamplesPerPixel
    284,  # PlanarConfiguration
    338,  # ExtraSamples
    339,  # SampleFormat
}

FITS_PROTECTED_KEYS = {
    "SIMPLE", "BITPIX", "NAXIS", "EXTEND", "XTENSION", "PCOUNT", "GCOUNT", "END",
    # Preserve target data interpretation/scaling to avoid visual changes.
    "BSCALE", "BZERO", "BLANK",
    # Protect display-impacting color/model semantics.
    "COLOR", "COLORSPC", "COLORTYP", "BAYERPAT", "BAYERCOL",
    # Generic structural keys used by some metadata writers.
    "WIDTH", "HEIGHT", "SHAPE", "CHANNELS", "BITDEPTH", "DATATYPE", "DTYPE",
}

# ============================================================
# Utility Functions
# ============================================================

def normalize_ext(path: Union[str, Path]) -> str:
    """Get normalized file extension (lowercase with dot)."""
    return Path(path).suffix.lower()


def ensure_supported(path: Union[str, Path]) -> str:
    """Validate file has supported extension, return normalized extension."""
    ext = normalize_ext(path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    return ext


def is_fits(path: Union[str, Path]) -> bool:
    """Check if file is FITS format."""
    return normalize_ext(path) in {".fit", ".fits"}


def is_xisf(path: Union[str, Path]) -> bool:
    """Check if file is XISF format."""
    return normalize_ext(path) == ".xisf"


def is_tiff(path: Union[str, Path]) -> bool:
    """Check if file is TIFF format."""
    return normalize_ext(path) in {".tif", ".tiff"}


def is_standard_format(path: Union[str, Path]) -> bool:
    """Check if file is standard raster format (PNG, JPG, BMP, WebP)."""
    return normalize_ext(path) in {
        ".png", ".jpg", ".jpeg", ".bmp", ".webp"
    }


def metadata_transfer_family(path: Union[str, Path]) -> str | None:
    """Return transfer family for supported metadata transfer formats."""
    if is_tiff(path):
        return "tiff"
    if is_fits(path):
        return "fits"
    if is_xisf(path):
        return "xisf"
    return None


def can_transfer_metadata(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
) -> tuple[bool, str]:
    """Validate whether metadata can be transferred from source to target."""
    source_ext = ensure_supported(source_path)
    target_ext = ensure_supported(target_path)

    source_family = metadata_transfer_family(source_path)
    target_family = metadata_transfer_family(target_path)

    if source_family is None or target_family is None:
        return (
            False,
            (
                "Metadata transfer is supported only for TIFF, FITS, and XISF. "
                f"Received source '{source_ext}' and target '{target_ext}'."
            ),
        )

    return True, ""


# ============================================================
# TIFF Metadata
# ============================================================

def read_tiff_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read TIFF metadata using tifffile.
    
    Returns:
        Dictionary with 'tags' (TIFF tags) and 'descriptions'
    """
    metadata = {
        "format": "TIFF",
        "tags": {},
        "descriptions": {},
    }

    try:
        with tifffile.TiffFile(filepath) as tif:
            if tif.pages:
                page = tif.pages[0]
                
                # Extract TIFF tags
                if hasattr(page, 'tags'):
                    for tag_name, tag in page.tags.items():
                        try:
                            tag_id = tag.code
                            tag_value = tag.value
                            metadata["tags"][str(tag_name)] = {
                                "id": tag_id,
                                "value": _serialize_metadata_value(tag_value),
                            }
                            if hasattr(tag, 'name'):
                                metadata["descriptions"][str(tag_name)] = tag.name
                        except Exception as e:
                            logger.warning(f"Error reading TIFF tag {tag_name}: {e}")
    except Exception as e:
        logger.error(f"Error reading TIFF metadata from {filepath}: {e}")

    return metadata


def write_tiff_metadata(
    target_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata: Dict[str, Any],
    mode: str = "overwrite"
) -> None:
    """
    Write metadata to TIFF file.
    
    Args:
        target_path: Source TIFF file to read
        output_path: Output TIFF file path
        metadata: Metadata dictionary with 'tags' key
        mode: "overwrite" (replace all tags) or "merge" (merge with existing)
    """
    if mode not in ("overwrite", "merge"):
        raise ValueError(f"Invalid mode: {mode}. Use 'overwrite' or 'merge'")

    target_path = Path(target_path)
    output_path = Path(output_path)

    same_path = False
    try:
        same_path = target_path.resolve() == output_path.resolve()
    except Exception:
        same_path = str(target_path) == str(output_path)

    temp_output_path: Path | None = None
    write_path: Path = output_path
    if same_path:
        temp_output_path = target_path.with_name(
            f"{target_path.stem}.meta_tmp_{os.getpid()}{target_path.suffix}"
        )
        write_path = temp_output_path

    try:
        # Read image data from target
        with tifffile.TiffFile(target_path) as tif:
            image_data = tif.asarray()
            existing_tags: Dict[int, Any] = {}
            
            if mode == "merge" and tif.pages:
                # Preserve existing tags
                page = tif.pages[0]
                if hasattr(page, 'tags'):
                    for _, tag in page.tags.items():
                        try:
                            existing_tags[int(tag.code)] = tag.value
                        except Exception:
                            pass

        source_tags = {
            tag_id: value
            for tag_id, value in _extract_tiff_tag_values(metadata).items()
            if not _is_tiff_protected_tag_id(tag_id)
        }
        merged_tags: Dict[int, Any] = {}

        if mode == "merge":
            merged_tags.update(existing_tags)
        merged_tags.update(source_tags)

        write_kwargs = _tiff_imwrite_kwargs_from_tags(merged_tags)

        # Write TIFF with metadata
        tifffile.imwrite(write_path, image_data, **write_kwargs)
        if same_path and temp_output_path is not None:
            os.replace(str(temp_output_path), str(output_path))
        logger.info(f"TIFF metadata written to {output_path}")

    except Exception as e:
        if temp_output_path is not None:
            try:
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception:
                pass
        logger.error(f"Error writing TIFF metadata to {output_path}: {e}")
        raise


# ============================================================
# FITS Metadata
# ============================================================

def read_fits_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read FITS metadata using astropy.
    
    Returns:
        Dictionary with HDU information and headers
    """
    if not HAS_FITS:
        raise RuntimeError(
            "astropy is required for FITS support.\n"
            "Install with: pip install astropy"
        )

    metadata = {
        "format": "FITS",
        "hdu_count": 0,
        "hdus": {},
    }

    try:
        with fits.open(filepath) as hdul:
            metadata["hdu_count"] = len(hdul)

            for i, hdu in enumerate(hdul):
                hdu_meta = {
                    "name": hdu.name if hasattr(hdu, 'name') else f"HDU_{i}",
                    "type": type(hdu).__name__,
                    "header": {},
                }

                # Extract header cards
                if hasattr(hdu, 'header'):
                    for card in hdu.header.cards:
                        try:
                            hdu_meta["header"][card.keyword] = {
                                "value": _serialize_metadata_value(card.value),
                                "comment": card.comment or "",
                            }
                        except Exception:
                            pass

                metadata["hdus"][f"HDU_{i}"] = hdu_meta

    except Exception as e:
        logger.error(f"Error reading FITS metadata from {filepath}: {e}")

    return metadata


def write_fits_metadata(
    target_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata: Dict[str, Any],
    mode: str = "overwrite"
) -> None:
    """
    Write metadata to FITS file.
    
    Args:
        target_path: Source FITS file to read
        output_path: Output FITS file path
        metadata: Metadata dictionary with 'hdus' key
        mode: "overwrite" or "merge"
    """
    if not HAS_FITS:
        raise RuntimeError("astropy is required for FITS support")
    
    if mode not in ("overwrite", "merge"):
        raise ValueError(f"Invalid mode: {mode}")

    target_path = Path(target_path)
    output_path = Path(output_path)

    same_path = False
    try:
        same_path = target_path.resolve() == output_path.resolve()
    except Exception:
        same_path = str(target_path) == str(output_path)

    temp_output_path: Path | None = None
    write_path: Path = output_path
    if same_path:
        temp_output_path = target_path.with_name(
            f"{target_path.stem}.meta_tmp_{os.getpid()}{target_path.suffix}"
        )
        write_path = temp_output_path

    try:
        # Read target FITS data; keep image payload untouched.
        with fits.open(target_path) as hdul:
            new_hdul = hdul.copy()

            # Apply metadata
            if "hdus" in metadata:
                for hdu_key, hdu_meta in metadata["hdus"].items():
                    hdu_idx = int(hdu_key.split("_")[1]) if "_" in hdu_key else 0
                    
                    if hdu_idx < len(new_hdul):
                        hdu = new_hdul[hdu_idx]
                        if "header" in hdu_meta:
                            if mode == "overwrite":
                                _clear_fits_user_header_cards(hdu.header)
                            for key, val in hdu_meta["header"].items():
                                if _is_fits_protected_keyword(str(key)):
                                    continue
                                if isinstance(val, dict):
                                    hdu.header[key] = val.get("value", val)
                                else:
                                    hdu.header[key] = val

            # Write while HDUs are in-scope to avoid closed-file lazy data access.
            new_hdul.writeto(write_path, overwrite=True)

        if same_path and temp_output_path is not None:
            os.replace(str(temp_output_path), str(output_path))
        logger.info(f"FITS metadata written to {output_path}")

    except Exception as e:
        if temp_output_path is not None:
            try:
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception:
                pass
        logger.error(f"Error writing FITS metadata to {output_path}: {e}")
        raise


# ============================================================
# XISF Metadata
# ============================================================

def read_xisf_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read XISF metadata using xisf module.
    
    Returns:
        Dictionary with XISF metadata
    """
    if not HAS_XISF:
        raise RuntimeError(
            "xisf is required for XISF support.\n"
            "Install with: pip install xisf"
        )

    metadata = {
        "format": "XISF",
        "properties": {},
        "metadata": {},
    }

    try:
        xisf = XISF(filepath)

        # Preferred API from xisf package.
        if hasattr(xisf, "get_images_metadata"):
            images_meta = xisf.get_images_metadata() or []
            if images_meta and isinstance(images_meta[0], dict):
                first = images_meta[0]
                fits_keywords = first.get("FITSKeywords")
                if isinstance(fits_keywords, dict):
                    for key, entries in fits_keywords.items():
                        value = None
                        if isinstance(entries, list) and entries:
                            first_entry = entries[0]
                            if isinstance(first_entry, dict):
                                value = first_entry.get("value")
                        elif isinstance(entries, dict):
                            value = entries.get("value")
                        else:
                            value = entries
                        if value is not None:
                            metadata["properties"][key] = _serialize_metadata_value(value)

                xisf_props = first.get("XISFProperties")
                if isinstance(xisf_props, dict):
                    for key, prop in xisf_props.items():
                        if isinstance(prop, dict) and "value" in prop:
                            metadata["properties"][key] = _serialize_metadata_value(prop.get("value"))
                        else:
                            metadata["properties"][key] = _serialize_metadata_value(prop)

        if hasattr(xisf, "get_file_metadata"):
            metadata["metadata"] = _serialize_metadata_value(xisf.get_file_metadata())

        # Backward-compatible fallback for alternate wrappers/mocks.
        if not metadata["properties"] and hasattr(xisf, "fits_header"):
            for key, val in xisf.fits_header.items():
                metadata["properties"][key] = _serialize_metadata_value(val)
        if not metadata["metadata"] and hasattr(xisf, "metadata"):
            metadata["metadata"] = _serialize_metadata_value(xisf.metadata)

    except Exception as e:
        logger.error(f"Error reading XISF metadata from {filepath}: {e}")

    return metadata


def write_xisf_metadata(
    target_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata: Dict[str, Any],
    mode: str = "overwrite"
) -> None:
    """
    Write metadata to XISF file.
    
    Args:
        target_path: Source XISF file to read
        output_path: Output XISF file path
        metadata: Metadata dictionary
        mode: "overwrite" or "merge"
    """
    if not HAS_XISF:
        raise RuntimeError("xisf is required for XISF support")

    target_path = Path(target_path)
    output_path = Path(output_path)

    same_path = False
    try:
        same_path = target_path.resolve() == output_path.resolve()
    except Exception:
        same_path = str(target_path) == str(output_path)

    temp_output_path: Path | None = None
    write_path: Path = output_path
    if same_path:
        temp_output_path = target_path.with_name(
            f"{target_path.stem}.meta_tmp_{os.getpid()}{target_path.suffix}"
        )
        write_path = temp_output_path

    try:
        # Read source pixels via xisf reader API.
        xisf_src = XISF(target_path)
        image_data = None
        if hasattr(xisf_src, "read_image"):
            try:
                image_data = xisf_src.read_image(0, data_format="channels_last")
            except TypeError:
                image_data = xisf_src.read_image(0)
        elif hasattr(xisf_src, "data"):
            image_data = xisf_src.data

        if image_data is None:
            raise RuntimeError(f"Could not read XISF image payload from: {target_path}")

        image_array = np.asarray(image_data)
        if image_array.ndim == 2:
            image_array = image_array[..., np.newaxis]
        image_array = np.ascontiguousarray(image_array)

        image_metadata: Dict[str, Any] = {}
        xisf_metadata: Dict[str, Any] = {}

        if mode == "merge":
            if hasattr(xisf_src, "get_images_metadata"):
                existing_images = xisf_src.get_images_metadata() or []
                if existing_images and isinstance(existing_images[0], dict):
                    existing_image_meta = existing_images[0]
                    existing_keywords = existing_image_meta.get("FITSKeywords")
                    if isinstance(existing_keywords, dict):
                        image_metadata["FITSKeywords"] = dict(existing_keywords)
                    existing_props = existing_image_meta.get("XISFProperties")
                    if isinstance(existing_props, dict):
                        image_metadata["XISFProperties"] = dict(existing_props)

            if hasattr(xisf_src, "get_file_metadata"):
                existing_file_meta = xisf_src.get_file_metadata()
                if isinstance(existing_file_meta, dict):
                    xisf_metadata = dict(existing_file_meta)

        if "properties" in metadata and isinstance(metadata["properties"], dict):
            fits_keywords = image_metadata.setdefault("FITSKeywords", {})
            for key, val in metadata["properties"].items():
                if _is_fits_protected_keyword(str(key)):
                    continue
                if isinstance(val, dict):
                    keyword_value = val.get("value", val)
                    keyword_comment = str(val.get("comment", "Transferred metadata"))
                else:
                    keyword_value = val
                    keyword_comment = "Transferred metadata"
                fits_keywords[str(key)] = [{
                    "value": _serialize_metadata_value(keyword_value),
                    "comment": keyword_comment,
                }]

        if "metadata" in metadata and isinstance(metadata["metadata"], dict):
            if mode == "overwrite":
                xisf_metadata = dict(metadata["metadata"])
            else:
                xisf_metadata.update(metadata["metadata"])

        # Release potential source-file handles before replacing the original file.
        try:
            if hasattr(xisf_src, "close"):
                xisf_src.close()
        except Exception:
            pass
        xisf_src = None

        # Use class writer from xisf package (supports metadata dictionaries).
        XISF.write(
            write_path,
            image_array,
            creator_app="CieloStitch",
            image_metadata=image_metadata or None,
            xisf_metadata=xisf_metadata or None,
        )
        if same_path and temp_output_path is not None:
            os.replace(str(temp_output_path), str(output_path))
        logger.info(f"XISF metadata written to {output_path}")

    except Exception as e:
        if temp_output_path is not None:
            try:
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception:
                pass
        logger.error(f"Error writing XISF metadata to {output_path}: {e}")
        raise


# ============================================================
# Standard Format Metadata (PNG, JPG, BMP, WebP)
# ============================================================

def read_standard_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read metadata from standard formats (PNG, JPG, BMP, WebP) using OpenCV.
    
    Limited metadata support due to OpenCV constraints.
    Returns basic image information.
    
    Returns:
        Dictionary with image properties
    """
    metadata = {
        "format": normalize_ext(filepath).upper().lstrip("."),
        "width": 0,
        "height": 0,
        "channels": 0,
        "dtype": "",
    }

    try:
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        
        if img is not None:
            metadata["height"] = img.shape[0]
            metadata["width"] = img.shape[1]
            metadata["channels"] = img.shape[2] if len(img.shape) > 2 else 1
            metadata["dtype"] = str(img.dtype)
        else:
            logger.warning(f"Could not read image: {filepath}")

    except Exception as e:
        logger.error(f"Error reading metadata from {filepath}: {e}")

    return metadata


def write_standard_metadata(
    target_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata: Dict[str, Any],
    mode: str = "overwrite"
) -> None:
    """
    Write metadata to standard format.
    
    Note: Standard formats have limited metadata support without PIL.
    This primarily copies image data. For full metadata preservation,
    use specialized formats (TIFF, FITS, XISF).
    
    Args:
        target_path: Source image file
        output_path: Output image file path
        metadata: Metadata dictionary (limited use)
        mode: "overwrite" or "merge"
    """
    try:
        img = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        
        if img is not None:
            cv2.imwrite(str(output_path), img)
            logger.info(f"Image written to {output_path}")
        else:
            raise RuntimeError(f"Could not read image: {target_path}")

    except Exception as e:
        logger.error(f"Error writing image to {output_path}: {e}")
        raise


# ============================================================
# High-Level Interface
# ============================================================

def read_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read metadata from any supported format.
    
    Args:
        filepath: Path to image file
        
    Returns:
        Dictionary containing format-specific metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = ensure_supported(filepath)

    if is_tiff(filepath):
        return read_tiff_metadata(filepath)
    elif is_fits(filepath):
        return read_fits_metadata(filepath)
    elif is_xisf(filepath):
        return read_xisf_metadata(filepath)
    elif is_standard_format(filepath):
        return read_standard_metadata(filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def write_metadata(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    mode: str = "overwrite"
) -> None:
    """
    Write metadata to target image and save as output.
    
    Args:
        source_path: Source image file (to extract metadata from)
        target_path: Target image file (to apply metadata to)
        output_path: Output image file path
        metadata: Optional metadata dict (if None, copied from source)
        mode: "overwrite" (replace metadata) or "merge" (merge with existing)
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    source_ext = ensure_supported(source_path)
    target_ext = ensure_supported(target_path)
    output_ext = ensure_supported(output_path)

    can_transfer, reason = can_transfer_metadata(source_path, target_path)
    if not can_transfer:
        raise ValueError(reason)

    source_family = metadata_transfer_family(source_path)
    target_family = metadata_transfer_family(target_path)
    output_family = metadata_transfer_family(output_path)
    if target_family != output_family:
        raise ValueError(
            "Output format must match target format family for metadata transfer. "
            f"Target '{target_ext}', output '{output_ext}'."
        )

    # If metadata not provided, read from source
    if metadata is None:
        metadata = read_metadata(source_path)

    if source_family is not None and target_family is not None and source_family != target_family:
        metadata = _convert_metadata_for_target_family(metadata, target_family)

    # Write to target format
    if is_tiff(target_path):
        write_tiff_metadata(target_path, output_path, metadata, mode)
    elif is_fits(target_path):
        write_fits_metadata(target_path, output_path, metadata, mode)
    elif is_xisf(target_path):
        write_xisf_metadata(target_path, output_path, metadata, mode)
    elif is_standard_format(target_path):
        write_standard_metadata(target_path, output_path, metadata, mode)
    else:
        raise ValueError(f"Unsupported target format: {target_ext}")


def copy_metadata(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Copy metadata from source image to target image.
    
    Args:
        source_path: Source image file
        target_path: Target image file (data + metadata destination)
        output_path: Output file path
    """
    metadata = read_metadata(source_path)
    write_metadata(source_path, target_path, output_path, metadata, mode="overwrite")


def merge_metadata(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Merge metadata from source into target, preserving existing target metadata.
    
    Args:
        source_path: Source image file
        target_path: Target image file
        output_path: Output file path
    """
    metadata = read_metadata(source_path)
    write_metadata(source_path, target_path, output_path, metadata, mode="merge")


def replace_metadata(
    target_path: Union[str, Path],
    output_path: Union[str, Path],
    new_metadata: Dict[str, Any]
) -> None:
    """
    Replace metadata in target image with new metadata.
    
    Args:
        target_path: Target image file
        output_path: Output file path
        new_metadata: New metadata dictionary
    """
    write_metadata(target_path, target_path, output_path, new_metadata, mode="overwrite")


def replace_metadata_from_source(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:
    """Replace target metadata with source metadata while preserving target pixels."""
    metadata = read_metadata(source_path)
    write_metadata(source_path, target_path, output_path, metadata, mode="overwrite")


def display_metadata(filepath: Union[str, Path], to_file: Optional[str] = None) -> str:
    """
    Display metadata in human-readable format.
    
    Args:
        filepath: Image file path
        to_file: Optional file path to save output
        
    Returns:
        Formatted metadata string
    """
    metadata = read_metadata(filepath)
    output = _format_metadata_display(metadata)
    
    if to_file:
        Path(to_file).write_text(output)
        logger.info(f"Metadata saved to {to_file}")
    
    return output


# ============================================================
# Utility Helpers
# ============================================================

def _serialize_metadata_value(value: Any) -> Any:
    """Convert metadata value to serializable format."""
    if isinstance(value, (np.integer, np.floating)):
        return float(value) if isinstance(value, np.floating) else int(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (list, tuple)):
        return [_serialize_metadata_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _serialize_metadata_value(v) for k, v in value.items()}
    elif isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='ignore')
        except Exception:
            return repr(value)
    else:
        return value


def _extract_tiff_tag_values(metadata: Dict[str, Any]) -> Dict[int, Any]:
    """Extract TIFF tag values from metadata payload into {tag_id: value} mapping."""
    out: Dict[int, Any] = {}
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    if not isinstance(tags, dict):
        return out

    for key, tag_info in tags.items():
        tag_id = None
        value = tag_info

        if isinstance(tag_info, dict):
            value = tag_info.get("value")
            if "id" in tag_info:
                try:
                    tag_id = int(tag_info.get("id"))
                except Exception:
                    tag_id = None

        if tag_id is None:
            try:
                tag_id = int(key)
            except Exception:
                continue

        out[int(tag_id)] = value

    return out


def _convert_metadata_for_target_family(metadata: Dict[str, Any], target_family: str) -> Dict[str, Any]:
    """Convert metadata payload into the destination-family schema."""
    canonical = _canonical_metadata_items(metadata)

    if target_family == "fits":
        header: Dict[str, Any] = {}
        for key, value in canonical.items():
            if _is_fits_protected_keyword(key):
                continue
            header[key] = {"value": value, "comment": "Transferred metadata"}
        return {
            "format": "FITS",
            "hdu_count": 1,
            "hdus": {"HDU_0": {"name": "PRIMARY", "type": "PrimaryHDU", "header": header}},
        }

    if target_family == "xisf":
        properties: Dict[str, Any] = {}
        for key, value in canonical.items():
            if _is_fits_protected_keyword(key):
                continue
            properties[key] = value
        return {"format": "XISF", "properties": properties, "metadata": {}}

    if target_family == "tiff":
        tags: Dict[str, Dict[str, Any]] = {}
        description_lines: list[str] = []
        for key, value in canonical.items():
            if _is_fits_protected_keyword(key):
                continue
            description_lines.append(f"{key}={value}")

        if description_lines:
            tags["ImageDescription"] = {"id": 270, "value": "\n".join(description_lines)}

        software = canonical.get("SOFTWARE")
        if software is not None:
            tags["Software"] = {"id": 305, "value": str(software)}

        dt_value = canonical.get("DATE_OBS") or canonical.get("DATE") or canonical.get("DATETIME")
        dt_text = _normalize_tiff_datetime(dt_value)
        if dt_text is not None:
            tags["DateTime"] = {"id": 306, "value": dt_text}

        return {"format": "TIFF", "tags": tags, "descriptions": {}}

    return metadata


def _canonical_metadata_items(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract scalar key-value metadata independent of source family."""
    out: Dict[str, Any] = {}
    if not isinstance(metadata, dict):
        return out

    fmt = str(metadata.get("format") or "").strip().upper()

    if fmt == "FITS":
        hdus = metadata.get("hdus")
        if isinstance(hdus, dict):
            primary = hdus.get("HDU_0") if isinstance(hdus.get("HDU_0"), dict) else None
            if primary is None and hdus:
                first_key = next(iter(hdus.keys()))
                first_hdu = hdus.get(first_key)
                if isinstance(first_hdu, dict):
                    primary = first_hdu
            if isinstance(primary, dict):
                header = primary.get("header")
                if isinstance(header, dict):
                    for key, raw in header.items():
                        norm = _normalize_metadata_key(key)
                        if not norm:
                            continue
                        value = raw.get("value") if isinstance(raw, dict) else raw
                        scalar = _coerce_metadata_scalar(value)
                        if scalar is not None:
                            out[norm] = scalar
        return out

    if fmt == "TIFF":
        tag_map = _extract_tiff_tag_values(metadata)
        tiff_name_map = {
            270: "DESCRIPTION",
            305: "SOFTWARE",
            306: "DATETIME",
            315: "ARTIST",
            33432: "COPYRIGHT",
        }
        for tag_id, raw in tag_map.items():
            if _is_tiff_protected_tag_id(tag_id):
                continue
            key = tiff_name_map.get(int(tag_id), f"TIFFTAG_{int(tag_id)}")
            scalar = _coerce_metadata_scalar(raw)
            if scalar is not None:
                out[key] = scalar
        return out

    if fmt == "XISF":
        props = metadata.get("properties")
        if isinstance(props, dict):
            for key, raw in props.items():
                norm = _normalize_metadata_key(key)
                if not norm:
                    continue
                scalar = _coerce_metadata_scalar(raw)
                if scalar is not None:
                    out[norm] = scalar
        return out

    for key, raw in metadata.items():
        norm = _normalize_metadata_key(key)
        if not norm:
            continue
        scalar = _coerce_metadata_scalar(raw)
        if scalar is not None:
            out[norm] = scalar
    return out


def _normalize_metadata_key(key: Any) -> str:
    text = str(key or "").strip().upper()
    if not text:
        return ""
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^A-Z0-9_]", "", text)
    if not text:
        return ""
    return text[:64]


def _coerce_metadata_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        value = value.get("value")
    value = _serialize_metadata_value(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) <= 8:
            return ",".join(str(v) for v in value)
        return None
    return str(value)


def _tiff_imwrite_kwargs_from_tags(tags: Dict[int, Any]) -> Dict[str, Any]:
    """Map TIFF tags to tifffile.imwrite kwargs supported by this codebase."""
    kwargs: Dict[str, Any] = {"metadata": None}

    # 270 ImageDescription
    description = _coerce_text(tags.get(270))
    if description:
        kwargs["description"] = description

    # 305 Software
    software = _coerce_text(tags.get(305))
    if software:
        kwargs["software"] = software

    # 306 DateTime
    datetime_text = _normalize_tiff_datetime(tags.get(306))
    if datetime_text:
        kwargs["datetime"] = datetime_text

    # 282 XResolution + 283 YResolution + 296 ResolutionUnit
    x_res = _coerce_resolution_scalar(tags.get(282))
    y_res = _coerce_resolution_scalar(tags.get(283))
    if x_res is not None and y_res is not None:
        kwargs["resolution"] = (x_res, y_res)

    res_unit = tags.get(296)
    if res_unit is not None:
        try:
            kwargs["resolutionunit"] = int(getattr(res_unit, "value", res_unit))
        except Exception:
            pass

    return kwargs


def _is_tiff_protected_tag_id(tag_id: int) -> bool:
    try:
        return int(tag_id) in TIFF_PROTECTED_TAG_IDS
    except Exception:
        return False


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    text = str(value).strip()
    return text or None


def _normalize_tiff_datetime(value: Any) -> str | None:
    """Normalize datetime values to TIFF DateTime format YYYY:MM:DD HH:MM:SS."""
    text = _coerce_text(value)
    if not text:
        return None

    # TIFF-native DateTime format.
    if re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", text):
        return text

    cleaned = text
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(cleaned)
        return parsed.strftime("%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    # Fallback formats for simple date/date-time values.
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y:%m:%d %H:%M:%S", "%Y:%m:%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y:%m:%d %H:%M:%S")
        except Exception:
            continue

    return None


def _coerce_resolution_scalar(value: Any) -> float | None:
    """Accept float/int or TIFF rational tuples like (num, den)."""
    if value is None:
        return None
    raw = getattr(value, "value", value)

    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)

    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            num = float(raw[0])
            den = float(raw[1])
            if den != 0:
                return num / den
        except Exception:
            return None

    return None


def _is_fits_protected_keyword(keyword: str) -> bool:
    key = str(keyword or "").upper()
    if key in FITS_PROTECTED_KEYS:
        return True
    if key.startswith("NAXIS"):
        return True
    return False


def _clear_fits_user_header_cards(header: Any) -> None:
    """Remove non-structural cards so overwrite can replace user metadata only."""
    keys = list(header.keys())
    for key in keys:
        if _is_fits_protected_keyword(str(key)):
            continue
        try:
            del header[key]
        except Exception:
            pass


def _format_metadata_display(metadata: Dict[str, Any], indent: int = 0) -> str:
    """Format metadata dictionary for display."""
    lines = []
    prefix = "  " * indent

    for key, value in metadata.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_metadata_display(value, indent + 1))
        elif isinstance(value, (list, tuple)):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(_format_metadata_display(item, indent + 1))
                else:
                    lines.append(f"{prefix}  - {_truncate_value(item)}")
        else:
            lines.append(f"{prefix}{key}: {_truncate_value(value)}")

    return "\n".join(lines)


def _truncate_value(value: Any, max_len: int = 120) -> str:
    """Truncate long values for display."""
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text
