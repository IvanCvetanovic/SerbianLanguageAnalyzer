import classla
import torch

# classla calls torch.load without weights_only=False, which breaks in PyTorch >= 2.6.
# Replace the torch reference inside classla's pretrain module with a compat wrapper.
import classla.models.common.pretrain as _pretrain_mod

class _TorchCompat:
    def __getattr__(self, name):
        return getattr(torch, name)
    def load(self, *args, weights_only=False, **kwargs):
        return torch.load(*args, weights_only=weights_only, **kwargs)

_pretrain_mod.torch = _TorchCompat()

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
