from thinc.api import chain
from .split_trf import split_trf_batch


def replace_listener(model):
    return chain(model, split_trf_batch())


def replace_listener_cfg(tok2vec_model_cfg, listener_model_cfg):
    result = tok2vec_model_cfg.copy()
    if (
        "TransformerModel" in tok2vec_model_cfg["@architectures"]
        and "TransformerListener" in listener_model_cfg["@architectures"]
    ):
        result["@architectures"] = "spacy-transformers.Tok2VecTransformer.v3"
        for key in ["pooling", "grad_factor"]:
            if key in listener_model_cfg and key not in result:
                result[key] = listener_model_cfg[key]
    return result
