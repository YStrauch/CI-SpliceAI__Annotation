# from typing import List, NamedTuple
# import numpy as np
# from typing import Union

# class ExtractionData():
#     def __init__(
#         self,
#         chrom: str,
#         pos: int,
#         ref: str,
#         alt: str,
#         reference_path: str,
#         annotation_table='grch38',
#         max_dist_from_var=1000,
#         keep_nucs_outside_gene=False
#     ):
#         '''
#             Represents all parameters required to extract REF and ALT DNA.
#             Chromosome `chrom` can be with or without 'chr' prefix.
#             `chrom` and `pos` are relative to the fasta file specified in the VCFAnnotator instance.
#             `ref` specifies the reference sequence, can be "." for unknown but not advised.
#             `alt` is the list of all alts, can contain "." for deletions but not advised.
#             `reference_path` is a path to a fasta file to be used.
#             annotates variants within `max_dist_from_var` nucleotides from the variant.
#             Gene boundaries and splice sites are defined in `annotation_table` (either 'grch38', 'grch37', or a path to a custom file).
#             `max_dist_from_var` specifies the area around a variant that is being annotated.
#             Set `mask` to disregard losses of non-splice sites and gains of splice sites.
#             Set `rebuild_fasta_index` to False if your fasta file is already indexed.
#         '''
#         self.chrom = chrom
#         self.pos = pos
#         self.ref = self.remove_dot(ref)
#         self.alt = self.remove_dot(alt)
#         self.reference_path = reference_path
#         self.annotation_table = annotation_table
#         self.max_dist_from_var = max_dist_from_var
#         self.keep_nucs_outside_gene = keep_nucs_outside_gene

#         super().__init__(chrom, pos, ref, alt)

#     def copy(self):
#         return VariantInput(**self.__dict__)
    
#     @staticmethod
#     def remove_dot(field):
#         return '' if field == '.' or field is None else field


# # Structure to represent an annotated area (i.e. gene)
# class AnnotatedArea(NamedTuple):
#     name: str
#     strand: str
#     start: int
#     end: int

# # Structure to represent a candidate (i.e. gene) and its ML data
# class VariantCandidateData(NamedTuple):
#     input: DetailedVariantInput
#     area: AnnotatedArea
#     x_ref: np.ndarray
#     x_var: np.ndarray
