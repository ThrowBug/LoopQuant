"""Compatibility fixes for Hugging Face mirror endpoints."""

from functools import wraps
from urllib.parse import urlsplit, urlunsplit


def patch_hub_pagination() -> bool:
    """Keep Hub API pagination on the configured mirror's origin.

    Some mirrors forward the official Hub's absolute ``Link: rel=next`` URL.
    Only rewrite those links when the response itself came from the configured
    mirror; leave unrelated URLs and official-Hub requests untouched.
    """
    from huggingface_hub import constants
    from huggingface_hub.utils import _pagination

    endpoint = urlsplit(constants.ENDPOINT)
    if endpoint.netloc == "huggingface.co":
        return False
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
        return False

    original = getattr(_pagination, "_get_next_page", None)
    if original is None:
        raise RuntimeError(
            "Unsupported huggingface_hub pagination API: _get_next_page is missing"
        )
    if getattr(original, "_loopquant_mirror_patch", False):
        return True

    @wraps(original)
    def get_next_page(response):
        next_page = original(response)
        if not next_page:
            return next_page
        source = urlsplit(str(response.url))
        target = urlsplit(next_page)
        if (
            (source.scheme, source.netloc) == (endpoint.scheme, endpoint.netloc)
            and target.scheme in ("http", "https")
            and target.netloc == "huggingface.co"
            and target.path.startswith("/api/")
        ):
            return urlunsplit(target._replace(
                scheme=endpoint.scheme, netloc=endpoint.netloc
            ))
        return next_page

    get_next_page._loopquant_mirror_patch = True
    _pagination._get_next_page = get_next_page
    return True
