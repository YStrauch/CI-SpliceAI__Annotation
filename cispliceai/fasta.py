import abc
import pyfaidx
import numpy as np

from cispliceai import cache

class BaseFasta(abc.ABC):
    @abc.abstractmethod
    def extract(self, reference_path: str, contig: str, pos: int, len: int)->str:
        '''Extracts fasta sequence at the given location.Must support optional(!) "chr" prefix in `contig`.'''
        pass


class PyFaidXFasta(BaseFasta):
    def __init__(self):
        self._cache = cache.RAMCache(lambda path: pyfaidx.Fasta(path, sequence_always_upper=True))

    @staticmethod
    def _normalise_chrom(chromosome, has_chr_prefix: bool):
        chromosome_str = str(chromosome)

        if has_chr_prefix:
            if not chromosome_str.startswith('chr'):
                chromosome_str = 'chr' + chromosome_str
        else:
            if chromosome_str.startswith('chr'):
                chromosome_str = chromosome_str[3:]

        return chromosome_str

    def extract(self, reference_path: str, contig: str, pos: int, len: int)->str:
        pos -= 1
        fasta: pyfaidx.Fasta = self._cache.fetch(reference_path)
        chrom = self._normalise_chrom(contig, np.any([key.startswith('chr') for key in fasta.keys()]))
        return fasta[chrom][pos:pos+len].seq

if __name__ == '__main__':
    pf = PyFaidXFasta()

    print(pf.extract('../Train/data/raw/hg.fa', 'chr1', 45915463, 10))