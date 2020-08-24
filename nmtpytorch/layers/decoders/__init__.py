from .conditional import ConditionalDecoder
from .simplegru import SimpleGRUDecoder
from .conditionalmm import ConditionalMMDecoder
from .multisourceconditional import MultiSourceConditionalDecoder
from .xu import XuDecoder
from .switchinggru import SwitchingGRUDecoder
from .vector import VectorDecoder
from .imagination import ImaginationDecoder
from .seq_imagination import SequentialImaginationDecoder
from .seq_imagination_v2 import SequentialImaginationDecoderV2
from .vag_conditional import VAGConditionalDecoder
from .vag_sharedspace import VAGSharedSpaceDecoder
from .transformer import TransformerDecoder
from .multisource_transformer import MultiSourceTransformerDecoder

def get_decoder(type_):
    """Only expose ones with compatible __init__() arguments for now."""
    return {
        'cond': ConditionalDecoder,
        'simplegru': SimpleGRUDecoder,
        'condmm': ConditionalMMDecoder,
        'vector': VectorDecoder,
    }[type_]
