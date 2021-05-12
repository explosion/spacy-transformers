from thinc.api import Model, chain
from typing import List
from ..data_classes import FullTransformerBatch, TransformerData


def split_trf_batch() -> Model[FullTransformerBatch, List[TransformerData]]:
    return Model("split-trf-batch", forward)


def forward(model, trf_full, is_train):
    def backprop(d_trf_datas):
        return trf_full.unsplit_by_doc([x.tensors for x in d_trf_datas])

    return trf_full.doc_data, backprop


def replace_listener(model):
    return chain(model, split_trf_batch())


def replace_listener_cfg(tok2vec_model_cfg, listener_model_cfg):
    result = tok2vec_model_cfg
    if "TransformerModel" in tok2vec_model_cfg["@architectures"] and "TransformerListener" in listener_model_cfg["@architectures"]:
        result["@architectures"] = "spacy-transformers.Tok2VecTransformer.v1"
        if "pooling" in listener_model_cfg and "pooling" not in result:
            result["pooling"] = listener_model_cfg["pooling"]
        if "grad_factor" in listener_model_cfg and "grad_factor" not in result:
            result["grad_factor"] = listener_model_cfg["pooling"]

    return result
