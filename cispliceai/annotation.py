from __future__ import division
import json
import os
from typing import BinaryIO, Generator, List, TextIO, Union, Tuple
import numpy as np
import pysam

from cispliceai import const, progress, data
from cispliceai.fasta import BaseFasta, PyFaidXFasta
from cispliceai.model import CISpliceAI

class Annotation(dict): # inheriting from dict to make it JSON serialisable
    '''An annotation of a variant effect'''
    def __init__(self, alt:str, area: str,
        DS_AG:float, DS_AL:float, DS_DG:float, DS_DL:float, DP_AG:int, DP_AL:int, DP_DG:int, DP_DL:int,
        error:str=None
    ):
        super().__init__(
            alt = alt,
            area = area,
            DS_AG = float(DS_AG),
            DS_AL = float(DS_AL),
            DS_DG = float(DS_DG),
            DS_DL = float(DS_DL),
            DP_AG = int(DP_AG),
            DP_AL = int(DP_AL),
            DP_DG = int(DP_DG),
            DP_DL = int(DP_DL),
            error = error,
        )

    def __str__(self):
        if self['error']:
            return f'{self["error"]}|||||||||'

        return f'{self["alt"]}|{self["area"]}|{self["DS_AG"]:.2f}|{self["DS_AL"]:.2f}|{self["DS_DG"]:.2f}|{self["DS_DL"]:.2f}|{self["DP_AG"]:.0f}|{self["DP_AL"]:.0f}|{self["DP_DG"]:.0f}|{self["DP_DL"]:.0f}'

    def get_highest_score(self):
        '''Returns the highest delta score across AG, AL, DG, and DL'''
        return max(self['DS_AG'], self['DS_AL'], self['DS_DG'], self['DS_DL'])

    @classmethod
    def most_significant(cls, instances: List):
        '''Returns the most significant annotation (the one with highest delta score) across all `Annotation` instances'''
        not_erroneous = [instance for instance in instances if instance['error'] is None]
        if len(not_erroneous) == 0:
            return instances[0]

        return not_erroneous[np.argmax([i.get_highest_score() for i in not_erroneous])]

class AnnotationCache():
    '''
        Caches variant annotations and variant annotation states in RAM to prevent double-evaluations of the same variant in one run.
        Does not persist, we recommend extending this class to persist to disk or database.
    '''
    def __init__(self):
        self._dict = {}

    @staticmethod
    def _key(variant: data.VariantSpecification, mask: bool = None):
        if mask is None:
            return str(variant)

        return f'{str(variant)} {"mask" if mask else "unmask"}'

    def has_annotations(self, variant: data.VariantSpecification, mask: bool) -> bool:
        '''Returns if an (un-)masked variant annotations is available'''
        return self._key(variant, mask) in self._dict

    def is_running(self, variant: data.VariantSpecification):
        '''Returns if a variant is currently being evaluated'''

        key = self._key(variant)
        return key in self._dict and self._dict[key] == 'running'

    def mark_running(self, variant: data.VariantSpecification):
        '''Signals that a variant is currently being evaluated'''

        self._dict[self._key(variant)] = 'running'

    def fetch_annotations(self, variant: data.VariantSpecification, mask: bool) -> List[Annotation]:
        '''Returns annotations for an (un-)masked variant from cache'''

        variant_key = self._key(variant)
        assert variant_key in self._dict, 'This variant was not cached'
        assert self._dict[variant_key] != 'queued', 'This variant was not processed'

        if self._dict[variant_key] != 'success':
            return [Annotation(None, None, 0, 0, 0, 0, 0, 0, 0, 0, self._dict[variant_key])]

        return self._dict[self._key(variant, mask)]

    def set_annotations(self, variant: data.VariantSpecification, mask: bool, annotations: List[Annotation]):
        '''Caches variant annotations for an (un-)masked variant definition'''
        self._dict[self._key(variant, mask)] = annotations
        self._dict[self._key(variant)] = 'success'

    def set_error(self, variant: data.VariantSpecification, error:str):
        '''Marks an error while processing a variant (i.e. REF mismatch).'''
        self._dict[self._key(variant)] = error


class VariantEffect():
    '''Internal class to calculate the variant effect from ML predictions. Internal use only.'''
    def __init__(self,
        area:data.AreaWithML,
        preds_ref:np.ndarray,
        preds_var:np.ndarray,
        max_dist_from_var:int,
        error: str = None
    ):
        self.area = area
        self.max_dist_from_var = max_dist_from_var
        self.error = error

        if error is not None:
            return


        diff = abs(len(preds_ref) - len(preds_var))
        if diff:
            # compensate if the variant has another length than the reference

            def shorten_predictions(preds, index, length):
                assert length != 1, 'Length is 1'
                assert index > 0, 'Index must be positive'
                assert index+length < len(preds), 'Cannot shorten more than available'
                return  np.concatenate([
                    preds[:index],
                    [np.max(preds[index:index+length], axis=0)],
                    preds[index+length:]
                ])

            def pad_predictions(preds, index, length):
                assert length > 0, 'Length must be > 0'
                assert index > 0, 'Index must be positive'
                return  np.concatenate([
                    preds[:index],
                    np.zeros((length, 3)),
                    preds[index:]
                ])


            if len(preds_ref) > len(preds_var):
                # DELETION, i.e. GAAG -> G. diff would be 3
                # Pad variant
                preds_var = pad_predictions(preds_var, max_dist_from_var, diff)
            else:
                # INSERTION, i.e. G -> GT. diff would be 1
                # Shorten variant
                preds_var = shorten_predictions(preds_var, max_dist_from_var, diff+1)

        # Make sure we correctly sliced this madness
        assert len(preds_ref) == len(preds_var), 'Indel compensation failed'
        # assert len(preds_ref) % 2 == 0, 'Sequence len is not dividable by 2'

        self.delta = preds_var - preds_ref

    def get_annotation(self, mask: bool):
        if self.error:
            return Annotation(None, None, 0, 0, 0, 0, 0, 0, 0, 0, self.error)
        if mask:
            delta = self.delta.copy()
            mask_acceptors, mask_donors = np.zeros((2, len(delta)), dtype=bool)

            # get acceptor/donor positions relative to extraction start within our prediction window
            acceptors, donors = [np.array([
                site for site in sites if site >= self.area.extract_range[0] + const.CONTEXT_LEN//2 and site < self.area.extract_range[1] - const.CONTEXT_LEN//2]
            ) - self.area.extract_range[0] - const.CONTEXT_LEN//2 for sites in (self.area.acceptors, self.area.donors)]

            if acceptors.size > 0:
                mask_acceptors[acceptors] = True

            if donors.size > 0:
                mask_donors[donors] = True

            AL = delta[:, 1] < 0
            DL = delta[:, 2] < 0
            delta[(~AL & mask_acceptors), 1] = 0
            delta[(~DL & mask_donors), 2] = 0
            delta[(AL & ~mask_acceptors), 1] = 0
            delta[(DL & ~mask_donors), 2] = 0
        else:
            delta = self.delta

        return Annotation(
            self.area.var_spec.alt,
            self.area.name,
            np.max(delta[:, 1]),
            -np.min(delta[:, 1]),
            np.max(delta[:, 2]),
            -np.min(delta[:, 2]),
            np.argmax(delta[:, 1]) - self.max_dist_from_var,
            np.argmin(delta[:, 1]) - self.max_dist_from_var,
            np.argmax(delta[:, 2]) - self.max_dist_from_var,
            np.argmin(delta[:, 2]) - self.max_dist_from_var,
        )

class AnnotationJob():
    '''
        A job specification for annotating variant effect(s).
    '''
    def __init__(self, variant: data.VariantSpecification, mask: bool):
        '''
            Takes a `variant` of type `VariantSpecification`.
            `mask` specifies if annotations should only look at gains of non-splice sites and losses of splice sites (`True`), or all effects.
        '''
        self._variant = variant
        self._mask = mask

    @property
    def variant(self):
        return self._variant

    @property
    def mask(self):
        return self._mask

    def make_reference_path_absolute(self, cwd: str):
        '''Prepend a directory to relative reference paths of a variant.'''
        if not os.path.isabs(self.variant.reference_path):
            self._variant.reference_path = os.path.join(cwd, self.variant.reference_path)


class AnnotationJobWithMLData(AnnotationJob):
    '''
        An intermediary job specification that includes all ML data. Internal use only.
    '''
    def __init__(self, ml_data: List[data.AreaWithML], job: AnnotationJob):
        super().__init__(job.variant, job.mask)
        self._ml_data:List[data.AreaWithML] = ml_data

    @property
    def ml_data(self):
        return self._ml_data


class AnnotationJobWithResults(AnnotationJob):
    '''
        A processed annotation job.
        The fields `annotations` or `most_significant_annotation` return the annotation results.
    '''
    def __init__(self, annotations: List[Annotation], job: AnnotationJob):
        super().__init__(job.variant, job.mask)
        self._annotations = annotations

    @property
    def annotations(self):
        return self._annotations

    @property
    def most_significant_annotation(self):
        return Annotation.most_significant(self._annotations)

    @property
    def annotation(self):
        if self._most_significant_only:
            return self.most_significant_annotation
        return self.annotations

class JSONFormatException(Exception):
    pass


class Annotator():
    '''
        The main interface to annotate variants with. Supports a list of jobs, VCF and JSON input.
    '''
    def __init__(self, batch_size_mb: float = 1.0, annotation_cache: AnnotationCache = None, fasta: BaseFasta=None, model=None):
        '''
            `batch_size_mb` specifies the maximum input size of ML data per batch. Keep in mind that the model and inference process needs RAM too.
            `annotation_cache` caches previously seen variant annotations. Defaults to a memory cache (`AnnotationCache`).
            `fasta` provides extraction method for sequences. Defaults to a local extractor using pfaidx (`PyFaidXFasta`).
        '''
        if fasta is None:
            fasta = PyFaidXFasta()

        self._model = CISpliceAI() if model is None else model
        self.batch_size_mb = batch_size_mb
        self.preprocessor = data.DNAPreprocessor(fasta)
        self.annotation_cache:AnnotationCache = annotation_cache if annotation_cache is not None else AnnotationCache()

    def run_jobs(self, jobs: List[AnnotationJob]):
        '''
            Runs the `jobs` (list of `AnnotationJob`s).
            Returns a list of `AnnotationJobWithResults`.

            Will return annotations from variant cache if possible.
            Batches all data according to `batch_size_mb` (specified in constructor)
        '''
        jobs_to_run = []

        for job in jobs:
            if not self.annotation_cache.has_annotations(job.variant, job.mask) and not self.annotation_cache.is_running(job.variant):
                jobs_to_run.append(job)
                self.annotation_cache.mark_running(job.variant)


        self._run_batches(jobs_to_run)
        results: List[AnnotationJobWithResults] = []

        for job in jobs:
            if self.annotation_cache.is_running(job.variant):
                print(f'Potential race condition - someone else tried to compute {job.variant} but has not done in time')
                continue

            results.append(AnnotationJobWithResults(self.annotation_cache.fetch_annotations(job.variant, job.mask), job))

        return results



    def _run_batches(self, jobs: List[AnnotationJob]):
        for jobs_batch, error in self._create_ml_batches(jobs):
            if error:
                # pre-processing error (i.e. mismatching REF annotation)
                for job in jobs_batch:
                    self.annotation_cache.set_error(job.variant, error)

                continue

            x_ref = [area.x_ref for job in jobs_batch for area in job.ml_data]
            x_var = [area.x_var for job in jobs_batch for area in job.ml_data]
            preds = self._model.predict(x_ref + x_var)

            preds_ref, preds_var = preds[:len(preds)//2], preds[len(preds)//2:]
            assert len(preds_ref) == len(preds_var) == len(x_ref)
            i = 0
            for job in jobs_batch:
                annotations_masked = []
                annotations_unmasked = []
                for area in job.ml_data:
                    pred_ref, pred_var = preds_ref[i], preds_var[i]

                    if area.strand == '-':
                        pred_ref, pred_var = pred_ref[::-1], pred_var[::-1]

                    annotations_masked.append(VariantEffect(area, pred_ref, pred_var, job.variant.max_dist_from_var).get_annotation(True))
                    annotations_unmasked.append(VariantEffect(area, pred_ref, pred_var, job.variant.max_dist_from_var).get_annotation(False))
                    i += 1

                self.annotation_cache.set_annotations(job.variant, False, annotations_unmasked)
                self.annotation_cache.set_annotations(job.variant, True, annotations_masked)


    def _create_ml_batches(self, jobs: List[AnnotationJob]) -> Generator[Tuple[List[AnnotationJobWithMLData], str], None, None]:
        batch: List[AnnotationJob] = []
        prog = progress.ProgressOutput(len(jobs))

        for i, job in enumerate(jobs):
            try:
                ml_data = self.preprocessor.get_ml_data(job.variant)
                batch.append(AnnotationJobWithMLData(ml_data, job))
            except AssertionError as e:
                yield [AnnotationJobWithMLData(None, job)], str(e)
                continue

            remainder = []
            while self._batch_size_in_mb(batch) > self.batch_size_mb:
                remainder.append(batch[-1])
                batch = batch[:-1]

                if not batch:
                    raise RuntimeError('%.f MB per batch is not enough to hold even one variant. Increase batch size or limit max dist from var.' % self.batch_size_mb)

                if self._batch_size_in_mb(batch) < self.batch_size_mb:
                    yield batch, None
                    batch = remainder
                    remainder = []

                prog.update(i) # not entirely accurate as there's still a remainder but oh well


        if batch:
            assert self._batch_size_in_mb(batch) < self.batch_size_mb, 'Error deriving batches from batch size. This is a programming error.'

            yield batch, None
            prog.update(len(jobs))

    @staticmethod
    def _batch_size_in_mb(batch: List[AnnotationJob], bytes_per_num = 4):
        ml = [area for job in batch for area in job.ml_data]

        # model will pad the batch to the same size, so we only need to measure the longest input from the batch
        longest_input = np.max([np.cumprod(x.shape)[-1] for area in ml for x in [area.x_ref, area.x_var]])

        return len(batch) * longest_input / 1000000 * bytes_per_num

    def annotate_vcf(self,
        vcf_in: Union[BinaryIO,str],
        vcf_out: Union[BinaryIO,str],
        reference_path: str,
        annotation_table: str = 'grch38',
        max_dist_from_var=1000,
        most_significant_only=True,
        mask=False,
        keep_nucs_outside_gene=False
    ):
        '''
            Annotates `vcf_in` and writes annotations into `vcf_out` (both can either be file paths or python file BinaryIO such as stdin.cache or stdout.cache).
            `reference_path` is a file path to a fasta file containing the reference genome. Can be `.fa.gz` if you install `biopython>=1.73`.
            `annotation_table` can either be 'grch38' or 'grch37' to use the tables provided in this module, or a path to a table of similar format.
            `max_dist_from_var` specifies the area around a variant that is being annotated.
            Set `most_significant_only` to annotate all effects per variant line, not just the most significant.
            Set `mask` to disregard losses of non-splice sites and gains of splice sites.
            Set `keep_nucs_outside_gene` to not mask nucleotides outside of overlapping areas from the annotation table.
            (Will never mask nucleotides if no areas overlap with the variant.)

            Will return annotations from variant cache if possible.
            Batches all data according to `batch_size_mb` (specified in constructor)

            Returns nothing.
        '''

        with pysam.VariantFile(vcf_in, 'r') as file_in:
            header = file_in.header

            vcf_in_lines = [v for v in file_in]

        # VCF files can have a list of ALTs while BaseAnnotator expects one ALT per line
        jobs = [AnnotationJob(
            data.VariantSpecification(
                chrom=line.chrom,
                pos=line.pos,
                ref=line.ref,
                alt=alt,
                reference_path=reference_path,
                annotation_table=annotation_table,
                max_dist_from_var=max_dist_from_var,
                keep_nucs_outside_gene=keep_nucs_outside_gene
            ),
            mask=mask
        ) for line in vcf_in_lines for alt in line.alts]

        header.add_line(
            '##INFO=<ID=CI-SpliceAI,Number=.,Type=String,Description="CI-SpliceAI V%s variant annotation for a maximum distance of %d nucleotides from the variant; masking %s; %s nucs outside tx %s. Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL (DS=Delta Score, DP=Delta Position, AG/AL=Acceptor Gain/Acceptor Loss, DG/DL=Donor Gain/Donor Loss). ">\n#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO'
            % (const.VERSION, max_dist_from_var, 'on' if mask else 'off', 'keep' if keep_nucs_outside_gene else 'remove', 'most significant only' if most_significant_only else 'all effects')
        )

        jobs = self.run_jobs(jobs)

        # re-arrange into VCF format where more than one effect can exist per line
        vcf_annotations = []
        i = 0
        for vcf_in_line in vcf_in_lines:
            annotations = []
            for alt in vcf_in_line.alts:
                job = jobs[i]
                assert job.variant.pos == vcf_in_line.pos
                assert job.variant.alt == alt

                if most_significant_only:
                    annotations.append(str(job.most_significant_annotation))
                else:
                    annotations.append(','.join(map(str, job.annotations)))

                i += 1

            vcf_annotations.append(','.join(annotations))

        with pysam.VariantFile(vcf_out, 'w', header=header) as file_out:
            for vcf_line, annotation in zip(vcf_in_lines, vcf_annotations):
                vcf_line.info['CI-SpliceAI'] = annotation
                file_out.write(vcf_line)

    def annotate_json(self, json_in: Union[TextIO,str], json_out: Union[TextIO,str]):
        '''
            Annotates `json_in` and writes annotations into `json_out` (both can either be file paths or python TextIO streams such as stdin or stdout).
            Annotates all affected genes, not only the most significantly affected.
            Returns nothing.
            Raises `JSONFormatException` if JSON has not the right format (should be JSON equivalent to `List[AnnotationJob]`).

            Will return annotations from variant cache if possible.
            Batches all data according to `batch_size_mb` (specified in constructor)
        '''
        if type(json_in) == str:
            with open(json_in, 'r') as f:
                jobs_json = json.load(f)
        else:
            jobs_json = json.load(json_in)


        if type(jobs_json) != list:
            raise JSONFormatException('Input JSON must be array of AnnotationJobs')
        try:
            jobs = [AnnotationJob(
                mask=job['mask'],
                variant=data.VariantSpecification(**job['variant'])
            ) for job in jobs_json]
        except Exception as e:
            raise JSONFormatException('Could not parse input to AnnotationJob') from e

        # Relative paths for fasta files are relative to the JSON file, not to CWD
        if type(json_in) == str:
            for job in jobs:
                job.make_reference_path_absolute(os.path.dirname(json_in))

        jobs = self.run_jobs(jobs)

        if type(json_out) == str:
            with open(json_out, 'w') as f:
                json.dump([job.annotation for job in jobs], f)
        else:
            json.dump([job.annotation for job in jobs], json_out)

if __name__ == '__main__':
    import sys
    annotation_table = 'grch38'
    ref_genome = '/Users/ich/Desktop/Programmieren/CI-SpliceAI/Train/data/raw/hg.fa'
    md = 50
    keep_nucs_outside = False

    annotator = Annotator(1)

    # # Programmatic use
    # jobs = annotator.run_jobs([
    #     AnnotationJob(
    #         data.VariantSpecification(1, 2303402, 'T', 'G', ref_genome, annotation_table, md, keep_nucs_outside),
    #         mask=True,
    #     ),
    #     AnnotationJob(
    #         data.VariantSpecification(1, 2303402, 'T', 'G', ref_genome, annotation_table, md, keep_nucs_outside),
    #         mask=False,
    #     )
    # ])

    # for job in jobs:
    #     print(job.most_significant_annotation)

    # VCF use
    # annotator.annotate_vcf('../input.vcf', sys.stdout, ref_genome, annotation_table, md, most_significant_only=True)

    # JSON use
    annotator.annotate_json('../input.json', sys.stdout)
