from importlib.metadata import version, PackageNotFoundError
from .annotation import Annotator, JSONFormatException, AnnotationJob, AnnotationJobWithResults, Annotation, AnnotationCache
from .cache import BaseCache, RAMCache
from .data import VariantSpecification

try:
    __version__ = version("cispliceai")
except PackageNotFoundError:
    __version__ = "unknown"
