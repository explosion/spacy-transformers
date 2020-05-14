from typing import List

from thinc.api import chain, Model, Ragged
from thinc.types import Floats2d, FloatsXd

from spacy.util import registry

from .pipeline import TransformerListener
from .wrapper import TransformerModelByName
from .util import find_last_hidden, TransformerData, FullTransformerBatch
from ._align import apply_alignment


@registry.architectures.register("spacy.Tok2VecTransformerListener.v1")
def transformer_listener_tok2vec_v1(
    pooling, width: int, grad_factor: float = 1.0
) -> Model[List[TransformerData], List[Floats2d]]:
    return chain(
        TransformerListener("transformer", width=width),
        trf_data_to_tensor(pooling, width, grad_factor),
    )


@registry.architectures.register("spacy.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(
    pooling,
    get_spans,
    name: str,
    width: int,
    fast_tokenizer: bool,
    grad_factor: float = 1.0,
) -> Model[List[TransformerData], List[Floats2d]]:
    return debug_print(
        chain(
            TransformerModelByName(
                name, fast_tokenizer=fast_tokenizer, get_spans=get_spans
            ),
            get_trf_data(),
            trf_data_to_tensor(pooling, width, grad_factor),
        )
    )


def debug_print(layer):
    def debug_forward(model, docs, is_train):
        transformer, get_trf, get_tokvecs = model.layers
        tensors, bp_tensors = transformer(docs, is_train)
        trf, bp_trf = get_trf(tensors, is_train)
        assert len(trf) == len(docs), (len(trf), len(docs))
        tokvecs, bp_tokvecs = get_tokvecs(trf, is_train)
        assert len(tokvecs) == len(docs), (len(tokvecs), len(docs))

        def backprop_debug(d_tokvecs):
            return bp_tensors(bp_trf(bp_tokvecs(d_tokvecs)))

        return tokvecs, backprop_debug

    return Model(
        "debug", debug_forward, layers=layer.layers, init=layer.init, dims=layer._dims
    )


def get_trf_data() -> Model[FullTransformerBatch, List[TransformerData]]:
    def _forward(model, trf_full, is_train):
        def backprop(d_trf_datas):
            return trf_full.unsplit_by_doc([x.tensors for x in d_trf_datas])

        return trf_full.doc_data, backprop

    return Model("get-trf-data", _forward)


def trf_data_to_tensor(
    pooling: Model[Ragged, Floats2d], width: int, grad_factor: float
) -> Model[List[TransformerData], List[Floats2d]]:
    return Model(
        "trf-data-to-tensor",
        forward,
        layers=[pooling],
        dims={"nO": width},
        attrs={"grad_factor": grad_factor},
    )


def forward(model: Model, trf_datas: List[TransformerData], is_train: bool):
    pooling: Model[Ragged, Floats2d] = model.layers[0]
    grad_factor = model.attrs["grad_factor"]
    outputs = []
    backprops = []
    for trf_data in trf_datas:
        t_i = find_last_hidden(trf_data.tensors)
        src = model.ops.reshape2f(trf_data.tensors[t_i], -1, trf_data.width)
        dst, get_d_src = apply_alignment(model.ops, trf_data.align, src)
        output, get_d_dst = pooling(dst, is_train)
        outputs.append(output)
        backprops.append((get_d_dst, get_d_src))

    def backprop_trf_to_tensor(d_outputs: List[Floats2d]) -> List[TransformerData]:
        d_trf_datas = []
        zipped = zip(trf_datas, d_outputs, backprops)
        for trf_data, d_output, (get_d_dst, get_d_src) in zipped:
            d_tensors: List[FloatsXd] = [
                model.ops.alloc(x.shape, dtype=x.dtype) for x in trf_data.tensors
            ]
            d_dst = get_d_dst(d_output)
            d_src = get_d_src(d_dst)
            d_src *= grad_factor
            t_i = find_last_hidden(trf_data.tensors)
            d_tensors[t_i] = d_src.reshape(trf_data.tensors[t_i].shape)
            d_trf_datas.append(
                TransformerData(
                    tensors=d_tensors,
                    spans=trf_data.spans,
                    tokens=trf_data.tokens,
                    align=trf_data.align,
                )
            )
        return d_trf_datas

    return outputs, backprop_trf_to_tensor
