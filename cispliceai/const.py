from importlib.metadata import version, PackageNotFoundError

NAME = 'cispliceai'
try:
    VERSION = version(NAME)
except PackageNotFoundError:
    VERSION = "LOCAL"


PREDICTION_LEN = 5000
CONTEXT_LEN = 10000
INPUT_LEN = 15000
