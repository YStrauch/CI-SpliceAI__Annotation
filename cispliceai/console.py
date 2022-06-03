import sys
import argparse
import os

from cispliceai import const
from cispliceai import annotation

if 'DEBUG_PYTHON' in os.environ and os.environ['DEBUG_PYTHON']:
    import ptvsd
    ptvsd.enable_attach(address = ('0.0.0.0', 8082))
    ptvsd.wait_for_attach()

def num_val(min_val, max_val=None, dtype=int):
    def ensure(arg):
        try:
            arg = dtype(arg)
        except ValueError as error:
            raise argparse.ArgumentTypeError('Must be %s' % str(dtype)) from error
        if arg < min_val:
            raise argparse.ArgumentTypeError('Must be at least %s' % str(min_val))
        if max_val is not None and arg > max_val:
            raise argparse.ArgumentTypeError('Must be at most %s' % str(max_val))
        return arg
    return ensure

def vcf():
    parser = argparse.ArgumentParser(description=const.VERSION)

    parser.add_argument('reference', help='path to the reference fasta file (*.fa, *.fa.gz)')
    parser.add_argument('--annotation', '-a', default='grch38', help='annotation table with gene names, defaults to "grch38" (table included). You can specify "grch37" or a path to your own table of the same format.')

    parser.add_argument('--input', '-i', default=sys.stdin.buffer, help='input VCF; defaults to stdin')
    parser.add_argument('--output', '-o', default=sys.stdout.buffer, help='output VCF; defaults to stdout')
    parser.add_argument('--distance', '-d', default=1000, type=num_val(2), help='maximum distance from the variant; defaults to 1000')
    parser.add_argument('--batch', '-b', default=10.0, type=num_val(1, dtype=float), help='maximum input batch size in MB. Be careful to leave enough space for the model and inference process. We recommend to increase this only for GPU processing. Defaults to 10')
    parser.add_argument('--all', action='store_true', help='annotate all affected genes/regions, not only the most significant')
    parser.add_argument('--outside', action='store_true', help='keep nucleotides outside of annotated transcript areas (defined in annotation table); by default outside nucleotides are encoded as N')
    parser.add_argument('--mask', '-m', action='store_true', help='mask events to disregard gains of splice sites and losses of non-splice sites')

    args = parser.parse_args()

    annotator = annotation.Annotator(
        batch_size_mb=args.batch,
    )

    annotator.annotate_vcf(
        vcf_in=args.input,
        vcf_out=args.output,
        reference_path=args.reference,
        annotation_table=args.annotation,
        max_dist_from_var=args.distance,
        most_significant_only=not args.all,
        mask=args.mask,
        keep_nucs_outside_gene=args.outside,
    )

def json():
    parser = argparse.ArgumentParser(description=const.VERSION)

    parser.add_argument('--input', '-i', default=sys.stdin, help='input JSON; defaults to stdin')
    parser.add_argument('--output', '-o', default=sys.stdout, help='output JSON; defaults to stdout')
    parser.add_argument('--batch', '-b', default=10.0, type=num_val(1, dtype=float), help='maximum input batch size in MB. Be careful to leave enough space for the model and inference process. We recommend to increase this only for GPU processing.')

    args = parser.parse_args()

    annotator = annotation.Annotator(
        batch_size_mb=args.batch,
    )

    annotator.annotate_json(
        json_in=args.input,
        json_out=args.output,
    )


if __name__ == '__main__':
    _argv = sys.argv

    sys.argv = _argv + [
        '../Train/data/raw/hg.fa',
        '-i', '../input.vcf',
        '-o', '../output.vcf',
        '-a', 'grch38',
        '-b', '1',
        '-d', '50',
        '--outside',
    ]
    vcf()

    # sys.argv = _argv + [
    #     '-i', '../input.json',
    #     '-o', '../output.json',
    # ]
    # json()
