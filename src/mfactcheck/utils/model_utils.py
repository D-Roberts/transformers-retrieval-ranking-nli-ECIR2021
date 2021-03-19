"""
Adapted from 
https://github.com/thunlp/KernelGAT/blob/master/retrieval_model/bert_model.py
"""

import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from hashlib import sha256
from io import open

import requests
from tqdm import tqdm


try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

try:
    from pathlib import Path

    PYTORCH_PRETRAINED_BERT_CACHE = Path(
        os.getenv(
            "PYTORCH_PRETRAINED_BERT_CACHE", Path.home() / ".pytorch_pretrained_bert"
        )
    )
except AttributeError:
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv(
        "PYTORCH_PRETRAINED_BERT_CACHE",
        os.path.join(os.path.expanduser("~"), ".pytorch_pretrained_bert"),
    )


TRAINED_MODEL_ARCHIVE_MAP = {
    "enmbert-sent": "https://claimtraindata.s3.amazonaws.com/models/enmbert/enmbert-sent.tar.gz",
    "enmbert-sent-onnx": "https://claimtraindata.s3.amazonaws.com/models/enmbert/enmbert-sent.tar.gz",
    "enmbert-nli": "https://claimtraindata.s3.amazonaws.com/models/enmbert/enmbert-nli.tar.gz",
    "enmbert-nli-onnx": "https://claimtraindata.s3.amazonaws.com/models/enmbert/enmbert-nli.tar.gz",
    "enrombert-sent": "https://claimtraindata.s3.amazonaws.com/models/enrombert/enrombert-sent.tar.gz",
    "enrombert-sent-onnx": "https://claimtraindata.s3.amazonaws.com/models/enrombert/enrombert-sent.tar.gz",
    "enrombert-nli": "https://claimtraindata.s3.amazonaws.com/models/enrombert/enrombert-nli.tar.gz",
    "enrombert-nli-onnx": "https://claimtraindata.s3.amazonaws.com/models/enrombert/enrombert-nli.tar.gz",
}

logger = logging.getLogger(__name__)


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
        print('cache_dir', cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError(f"file {cache_path} not found")

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError(f"file {meta_path} not found")

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(
            f"unable to parse {url_or_filename} as a URL or as a local path"
        )


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    response = requests.head(url, allow_redirects=True)
    if response.status_code != 200:
        raise IOError(
            f"HEAD request failed for url {url} with status code {response.status_code}"
        )
    etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info(f"copying {temp_file.name} to cache at {cache_path}")
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info(f"creating metadata file for {cache_path}")
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info(f"removing temp file {temp_file.name}")

    return cache_path


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


def get_model_dir(output_dir, add_ro, module, onnx, cache_dir=None):
    # load by default the model from output_dir if there is one
    os.makedirs(output_dir)
    pretrained_model_name_or_path = (
        "en" + "ro" * int(add_ro) + "mbert" + "-" + module + "-onnx" * int(onnx)
    )

    if pretrained_model_name_or_path in TRAINED_MODEL_ARCHIVE_MAP:
        archive_file = TRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
    else:
        archive_file = pretrained_model_name_or_path
    # redirect to the cache, if necessary
    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
    except EnvironmentError:
        logger.error(
            f"Model name '{pretrained_model_name_or_path}' was not found in model name list ({','.join(TRAINED_MODEL_ARCHIVE_MAP.keys())}). \
            We assumed '{archive_file}' was a path or url but couldn't find any file \
            associated to this path or url."
        )
        return None
    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(
            f"loading archive file {archive_file} from cache at {resolved_archive_file}"
        )

    logger.info(f"extracting archive file {resolved_archive_file} to dir {output_dir}")
    with tarfile.open(resolved_archive_file, "r:gz") as archive:
        archive.extractall(output_dir)
