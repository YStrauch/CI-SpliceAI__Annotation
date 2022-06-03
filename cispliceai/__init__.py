from pkg_resources import get_distribution
from .annotation import Annotator, JSONFormatException, AnnotationJob, AnnotationJobWithResults, Annotation, AnnotationCache
from .cache import BaseCache, RAMCache
from .data import VariantSpecification

__version__ = get_distribution('cispliceai').version