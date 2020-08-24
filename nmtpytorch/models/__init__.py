from .sat import ShowAttendAndTell
from .nli import NLI
from .nmt import NMT
from .mnmt import MultimodalNMT
from .acapt import AttentiveCaptioning

# Raw images
from .amnmtraw import AttentiveRawMNMT
# Spatial features + NMT
from .amnmtfeats import AttentiveMNMTFeatures
# Filtered attention (LIUMCVC-MMT2018)
from .amnmtfeats_fa import AttentiveMNMTFeaturesFA

# Speech models
from .asr import ASR
from .multimodal_asr import MultimodalASR

# Experimental: requires work/adaptation
from .multitask import Multitask
from .multitask_att import MultitaskAtt

from .imagination import Imagination
from .seq_imagination import SequentialImagination
from .vag import VisualAttentionGrounding
from .transformer import Transformer
from .imagination_transformer import ImaginationTransformer
from .seq_imagination_transformer import SequentialImaginationTransformer
from .am_transformer import AttentiveMultimodalTransformer
from .multitask_transformer import MultitaskTransformer
from .multitask_am_transformer import MultitaskAttentiveMultimodalTransformer

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
AttentiveMNMT = AttentiveRawMNMT
AttentiveEncAttMNMTFeatures = AttentiveMNMTFeaturesFA
