from pkg_resources import DistributionNotFound, get_distribution

NAME = 'cispliceai'
try:
    VERSION = get_distribution(NAME).version
except DistributionNotFound:
    VERSION = 'LOCAL'

PREDICTION_LEN = 5000
CONTEXT_LEN = 10000
INPUT_LEN = 15000