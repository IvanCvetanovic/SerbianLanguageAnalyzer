from pyvis.network import Network
from app_modules.pipeline import get_nlp

class TextAnalyzer:
    def __init__(self):
        self.pipeline = get_nlp()

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

        graph = Network(height="500px", width="100%")

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

        html = graph.generate_html()
        # Fix relative lib path so it resolves correctly when served from /dependency_tree/
        html = html.replace('src="lib/bindings/utils.js"',
                            'src="/lib/bindings/utils.js"')
        # Remove float:left from #mynetwork — float is ignored inside Bootstrap's flex
        # card but can confuse width resolution in some browsers
        html = html.replace('float: left;', '')
        # Defer drawGraph() so the browser finishes layout before vis.js measures
        # container.offsetWidth — without this the canvas can be stamped at the wrong size
        html = html.replace('drawGraph();', 'setTimeout(drawGraph, 50);')
        return html


