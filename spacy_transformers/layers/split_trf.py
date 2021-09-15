from thinc.api import Model
from typing import List
from ..data_classes import FullTransformerBatch, TransformerData


def split_trf_batch() -> Model[FullTransformerBatch, List[TransformerData]]:
    return Model("split-trf-batch", forward)


def forward(model, trf_full, is_train):
    def backprop(d_trf_datas):
        return trf_full.unsplit_by_doc([x.tensors for x in d_trf_datas])

    return trf_full.doc_data, backprop
