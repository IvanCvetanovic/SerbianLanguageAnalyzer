from app_modules.pipeline import nlp
from pyvis.network import Network

class TextAnalyzer:
    def __init__(self):
        self.pipeline = nlp

    def analyze_named_entities(self, text):
        doc = self.pipeline(text)
        ner_results = []

        conll_output = doc.to_conll()
        
        for line in conll_output.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.split('\t')
            
            word = fields[1]
            ner_tag = fields[-1]

            ner_tag = ner_tag.split('|')[0]

            if 'NER=B-' in ner_tag or 'NER=I-' in ner_tag:
                entity_type = ner_tag.split('=')[-1]
                ner_results.append({
                    'text': word,
                    'entity': entity_type
                })
        
        return ner_results

    def analyze_dependency_parsing(self, text):
        doc = self.pipeline(text)
        
        dp_results = []
        for sentence in doc.sentences:
            for word in sentence.words:
                dp_results.append({
                    'word': word.text, 
                    'lemma': word.lemma,      
                    'pos': word.upos,  
                    'head': word.head,   
                    'deprel': word.deprel
                })

        return dp_results

    def visualize_dependency_tree(self, input_string):
        doc = self.pipeline(input_string)

        graph = Network()

        for sentence in doc.sentences:
            for word in sentence.words:
                graph.add_node(word.id, label=word.text, shape="circle", font={'size': 16, 'color': 'black'})

        for sentence in doc.sentences:
            for word in sentence.words:
                if word.head != word.id: 
                    if word.head > 0:
                        if word.head in graph.get_nodes():
                            graph.add_edge(word.head, word.id, 
                                            title=word.deprel,
                                            label=word.deprel,
                                            arrows="from")

        return graph.generate_html()


