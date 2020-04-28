from collections import UserDict, defaultdict
from typing import Dict, Union, Any


class BatchEncoding(UserDict):
    """ BatchEncoding hold the output of the encode and batch_encode methods (tokens, attention_masks, etc).
        This class is derived from a python Dictionary and can be used as a dictionnary.
        In addition, this class expose utility methods to map from word/char space to token space.

        Args:
            data (:obj:`dict`): Dictionary of lists/arrays returned by the encode/batch_encode methods ('input_ids', 'attention_mask'...)
            encoding (:obj:`EncodingFast`, :obj:`list(EncodingFast)`, `optional`, defaults to :obj:`None`):
                If the tokenizer is a fast tokenizer which outputs additional informations like mapping from word/char space to token space
                the `EncodingFast` instance or list of instance (for batches) hold these informations.

    """

    def __init__(self, data: Dict[str, Any], encoding = None):
        super().__init__(data)

        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

    def __getitem__(self, item: Union[int, str]):
        """ If the key is a string, get the value of the dict associated to `key` ('input_ids', 'attention_mask'...)
            If the key is an integer, get the EncodingFast for batch item with index `key`
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
            )

    def __getattr__(self, item: str):
        return self.data[item]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()
