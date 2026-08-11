from .TCN_resnet import TCNResNNWithMetaAndEmbeddings as TCN_FM_multi_head
from .TCN_resnet_with_lstm import TCNResNNWithMetaAndEmbeddings as TCN_FM_multi_head_with_LSTM
from .TCN_resnet_ablate_embeddings import TCNResNNWithMetaAndEmbeddings as TCN_FM_ablate_embeddings
from .TCN_resnet_ablate_TCN import NNWithMetaAndEmbeddings as TCN_FM_ablate_TCN
from .TCN_resnet_ablate_meta_MLP import TCNResNNWithMetaAndEmbeddings as TCN_FM_ablate_meta_MLP
from .TCN_resnet_ablate_siRNA_input import TCNResNNWithMetaAndEmbeddings as TCN_FM_ablate_siRNA
from .TCN_resnet_ablate_embeddings_no_lstm import TCNResNNWithMetaAndEmbeddings as TCN_FM_ablate_embeddings_no_lstm

from .TCN_rnafm_lora import TCNResNNWithRNAFM_LoRA as TCN_RNAFM_LoRA
from .TCN_orthrus_lora import TCNResNNWithOrthrus_LoRA as TCN_Orthrus_LoRA

MODELS = {
    'TCN_FM_multi_head': TCN_FM_multi_head,
    'TCN_FM_multi_head_with_LSTM': TCN_FM_multi_head_with_LSTM,
    'TCN_FM_ablate_TCN': TCN_FM_ablate_TCN,
    'TCN_FM_ablate_embeddings': TCN_FM_ablate_embeddings,
    'TCN_FM_ablate_embeddings_no_lstm': TCN_FM_ablate_embeddings_no_lstm,
    'TCN_FM_ablate_siRNA': TCN_FM_ablate_siRNA,
    'TCN_FM_ablate_meta_MLP': TCN_FM_ablate_meta_MLP,
    'TCN_RNAFM_LoRA': TCN_RNAFM_LoRA,
    'TCN_Orthrus_LoRA': TCN_Orthrus_LoRA,
}
