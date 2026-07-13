class RagxError(Exception):
    """Base error for ragx. CLI maps this to exit code 2."""


class NotInitializedError(RagxError):
    """No ragx.toml or .ragx/ directory found at or above the working directory."""


class ManifestMismatchError(RagxError):
    """Index was built with a different embedding model/dimension than configured."""
