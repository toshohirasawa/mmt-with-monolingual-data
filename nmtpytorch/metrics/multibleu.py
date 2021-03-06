# -*- coding: utf-8 -*
import logging
import subprocess
import pkg_resources

from ..utils.misc import listify
from .metric import Metric
from time import sleep

logger = logging.getLogger('nmtpytorch')

BLEU_SCRIPT = pkg_resources.resource_filename('nmtpytorch',
                                              'lib/multi-bleu.perl')


class BLEUScorer:
    """BLEUScorer class."""
    def __init__(self):
        # For multi-bleu.perl we give the reference(s) files as argv,
        # while the candidate translations are read from stdin.
        self.__cmdline = [BLEU_SCRIPT]

    def compute(self, refs, hyps, language=None, lowercase=False,
                n_retry=5):
        cmdline = self.__cmdline[:]

        if lowercase:
            cmdline.append("-lc")

        # Make reference files a list
        cmdline.extend(listify(refs))

        if isinstance(hyps, str):
            hypstring = open(hyps).read().strip()
        elif isinstance(hyps, list):
            hypstring = "\n".join(hyps)

        #
        for _ in range(n_retry):
            try:
                score = subprocess.run(cmdline, 
                    stdout=subprocess.PIPE,
                    input=hypstring,
                    universal_newlines=True).stdout.splitlines()
                err = None
            except Exception as _e:
                # OSError: [Errno 12] Cannot allocate memory
                err = _e
        else:
            if err is not None:
                raise err

        if len(score) == 0:
            return Metric('BLEU', 0, "0.0")
        else:
            score = score[0].strip()
            float_score = float(score.split()[2][:-1])
            verbose_score = score.replace('BLEU = ', '')
            return Metric('BLEU', float_score, verbose_score)
