import numpy as np
import pandas as pd
from pkg_resources import resource_filename
import os
from typing import List

from cispliceai import const
from cispliceai import cache
from cispliceai.fasta import BaseFasta



class VariantSpecification():
    '''
        Represents a variant and all parameters required to extract REF and ALT DNA.
    '''
    def __init__(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        reference_path: str,
        annotation_table='grch38',
        max_dist_from_var=1000,
        keep_nucs_outside_gene=False
    ):
        '''
            Chromosome `chrom` can be with or without 'chr' prefix.
            `chrom` and `pos` are relative to the fasta file specified in the VCFAnnotator instance.
            `ref` specifies the reference sequence, can be "." for unknown but not advised.
            `alt` is the list of all alts, can contain "." for deletions but not advised.
            `reference_path` is a path to a fasta file to be used.
            annotates variants within `max_dist_from_var` nucleotides from the variant.
            Gene boundaries and splice sites are defined in `annotation_table` (either 'grch38', 'grch37', or a path to a custom file).
            `max_dist_from_var` specifies the area around a variant that is being annotated.
            Set `mask` to disregard losses of non-splice sites and gains of splice sites.
            Set `rebuild_fasta_index` to False if your fasta file is already indexed.
        '''
        self.chrom = str(chrom)
        self.pos = pos
        self.ref = self.remove_dot(ref)
        self.alt = self.remove_dot(alt)
        self.reference_path = reference_path
        self.annotation_table = annotation_table
        self.max_dist_from_var = max_dist_from_var
        self.keep_nucs_outside_gene = keep_nucs_outside_gene

    def copy(self):
        return VariantSpecification(**self.__dict__)

    @staticmethod
    def remove_dot(field):
        return '' if field == '.' or field is None else field


    def __str__(self):
        chrom = 'chr' + self.chrom if not self.chrom.startswith('chr') else self.chrom
        reference_name = os.path.basename(self.reference_path)

        key = f'{self.annotation_table},{reference_name};{"keep" if self.keep_nucs_outside_gene else "remove"}_nucs; ' +\
            f'{chrom} {self.pos}+-{self.max_dist_from_var} {self.ref}>{self.alt}'

        return key

class AnnotatedArea():
    '''
        Represents an annotated area (i.e. a gene)
    '''
    def __init__(
        self,
        name: str,
        strand: str,
        start: int,
        end: int,
        acceptors: List[int],
        donors: List[int],
    ):
        self.name = name
        self.strand = strand
        self.start = start
        self.end = end
        self.acceptors = acceptors
        self.donors = donors

class AreaWithML(AnnotatedArea):
    def __init__(self, area:AnnotatedArea, var_spec: VariantSpecification, extract_range: int, x_ref: np.ndarray, x_var: np.ndarray):
        super().__init__(**area.__dict__)

        self.var_spec = var_spec
        self.extract_range = extract_range
        self.x_ref = x_ref
        self.x_var = x_var

class DNAPreprocessor():
    ONE_HOT = np.asarray([[0, 0, 0, 0], # N
                          [1, 0, 0, 0], # A
                          [0, 1, 0, 0], # C
                          [0, 0, 1, 0], # G
                          [0, 0, 0, 1]],# T
                          dtype=bool)

    def __init__(self, fasta: BaseFasta):
        self._fasta = fasta
        self._annotation_table_cache = cache.RAMCache(lambda path: pd.read_csv(path).set_index('gene_id'))

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

    def get_ml_data(self, var_spec: VariantSpecification):
        '''
            Extracts GeneSlices from the fasta file around the coordinates given.
            Determines which genes overlay with this position.
            Prepares the DNA for ML for all genes by
                - removing nucs outside of genes (if keep_nucs_outside_gene set in `VariantSpecification`)
                - reverse-complementing reverse-stranded genes
        '''
        var_spec = var_spec.copy()

        window_size = var_spec.max_dist_from_var + const.CONTEXT_LEN//2

        overlapping_areas = self.get_overlapping_areas(var_spec.chrom, var_spec.pos, var_spec.annotation_table)
        if len(overlapping_areas) == 0:
            # position is either not within a gene or we don't remove nucleotides outside of the gene, run on both strands without gene mask and no known splice sites

            overlapping_areas = [
                AnnotatedArea(strand, strand, var_spec.pos-window_size-1, var_spec.pos+window_size+1, [], [])
                for strand in ['+', '-']
            ]
        extract_start, extract_len = var_spec.pos-window_size, 2*window_size+len(var_spec.ref)
        # ensure that the reference length is dividable by 2 or delta position will be half
        if extract_len % 2 != 0:
            extract_len += 1

        seq = self._fasta.extract(var_spec.reference_path, var_spec.chrom, extract_start, extract_len)
        offset = const.CONTEXT_LEN//2 + var_spec.max_dist_from_var
        assert seq[offset:offset+len(var_spec.ref)] == var_spec.ref, 'REF annotation mismatch - Expected "%s", found "%s"' % (var_spec.ref, seq[offset:offset+len(var_spec.ref)])

        x = self._onehot_seq(seq)

        areas_with_ml: List[AreaWithML] = []

        for area in overlapping_areas:
            # mask everything outside of area annotations with N
            if not var_spec.keep_nucs_outside_gene:
                mask = np.max([[0,0], [area.start - extract_start, extract_start+extract_len - area.end]], axis=0)
                x_ref = np.concatenate([np.zeros((mask[0], 4), dtype=x.dtype), x[mask[0]:len(x)-mask[1]], np.zeros((mask[1], 4,), dtype=x.dtype)])
            else:
                x_ref = x
            assert len(x_ref) == len(x)

            x_var = self.apply_var(x_ref, var_spec.ref, var_spec.alt, var_spec.max_dist_from_var)

            if area.strand == '-':
                x_ref = self._reverse_complement(x_ref)
                x_var = self._reverse_complement(x_var)

            areas_with_ml.append(AreaWithML(area, var_spec, (extract_start, extract_start+extract_len), x_ref, x_var))

        return areas_with_ml


    def apply_var(self, x: np.ndarray, ref:str, alt:str, max_dist_from_var):
        ref, alt = self._onehot_seq(ref), self._onehot_seq(alt)
        offset = const.CONTEXT_LEN//2 + max_dist_from_var
        assert (x[offset:offset+len(ref)] == ref).all(), 'REF annotation mismatch'
        return np.concatenate([x[:offset], alt, x[offset+len(ref):]])

    @staticmethod
    def _reverse_complement(x: np.ndarray):
        return np.flip(x, axis=1)[::-1]

    @classmethod
    def onehot_and_pad(cls, seq):
        return np.pad(cls._onehot_seq(seq), ((const.CONTEXT_LEN // 2, const.CONTEXT_LEN // 2), (0, 0)), 'constant')

    @classmethod
    def _onehot_seq(cls, seq):
        if seq == '':
            return np.zeros((0, 4))
        num = seq.upper().replace('N', '0').replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')
        num = np.asarray(list(map(int, list(num))))

        return cls.ONE_HOT[num]

    def get_overlapping_areas(self, chromosome, position, annotation_table_name: str):
        if annotation_table_name == 'grch37' or annotation_table_name == 'grch38':
            annotation_table_name = resource_filename(__name__, os.path.join('data', annotation_table_name + '.csv'))
        annotation_table: pd.DataFrame = self._annotation_table_cache.fetch(annotation_table_name)

        chromosome = self._normalise_chrom(chromosome, annotation_table.iloc[0].chr.startswith('chr'))

        genes_on_chrom = annotation_table[annotation_table.chr == chromosome]
        overlapping_areas = genes_on_chrom[(genes_on_chrom.start <= position) & (genes_on_chrom.end > position)]

        return [
            AnnotatedArea(gene_id, gene.strand, gene.start, gene.end, *(self._get_splice_sites(gene)))
            for gene_id, gene in overlapping_areas.iterrows()
        ]

    @staticmethod
    def _get_splice_sites(gene: pd.Series):
        acceptors, donors = (gene.jn_end, gene.jn_start) if gene.strand == '+' else (gene.jn_start, gene.jn_end)

        acceptors, donors = [np.array(sites.split(','), dtype=int) for sites in (acceptors, donors)]
        # acceptors, donors = [sites[(sites >=-variant.max_dist_from_var) & (sites <= variant.max_dist_from_var)] + variant.max_dist_from_var for sites in (acceptors, donors)]

        return acceptors, donors

    @staticmethod
    def _is_splice_site(pos: int, gene: pd.Series):
        return min(abs(pos - np.array(list(map(int, gene.jn_start.split(',') + gene.jn_end.split(',')))))) == 0
