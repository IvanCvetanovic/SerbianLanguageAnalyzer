import classla
import torch

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        use_gpu = torch.cuda.is_available()
        _nlp = classla.Pipeline(
            lang="sr",
            processors="tokenize,pos,lemma,ner,depparse",
            use_gpu=use_gpu,
        )
    return _nlp
