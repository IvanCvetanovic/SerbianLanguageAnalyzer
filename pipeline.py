import classla

nlp = classla.Pipeline(
    lang='sr',
    processors='tokenize,pos,lemma,ner,depparse'
)