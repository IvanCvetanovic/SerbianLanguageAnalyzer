import os
import classla

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        use_gpu = os.getenv("CLASSLA_GPU", "0").strip() == "1"
        _nlp = classla.Pipeline(
            lang="sr",
            processors="tokenize,pos,lemma,ner,depparse",
            use_gpu=use_gpu,
        )
    return _nlp
