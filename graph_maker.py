import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')

class Visualizer:
    def generate_word_cloud(self, frequencies: dict) -> str:
        if not frequencies:
            return None

        wc = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate_from_frequencies(frequencies)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_str

    def generate_ner_heatmap(self, classla_doc) -> str:
        entity_types = ['PER', 'LOC', 'ORG']
        num_types = len(entity_types)
        num_sents = len(classla_doc.sentences)

        if num_sents == 0:
            return None
        
        counts = np.zeros((num_types, num_sents), dtype=int)

        for si, sentence in enumerate(classla_doc.sentences):
            for ent in sentence.ents:
                if ent.type in entity_types:
                    ti = entity_types.index(ent.type)
                    counts[ti, si] += 1

        fig, ax = plt.subplots(figsize=(max(6, num_sents * 1.2), 3))
        im = ax.imshow(counts, cmap='Reds', aspect='auto')

        ax.set_xticks(np.arange(num_sents))
        ax.set_yticks(np.arange(num_types))
        ax.set_xticklabels([f"Sent {i+1}" for i in range(num_sents)])
        ax.set_yticklabels(entity_types)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Entity Count')
        ax.set_title("NER Heatmap by Sentence")
        ax.set_xlabel("Sentence")
        ax.set_ylabel("Entity Type")

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_str

    def generate_pos_sunburst(self, classla_doc) -> str:
        upos_counts = OrderedDict()
        xpos_counts = OrderedDict()

        for sent in classla_doc.sentences:
            for w in sent.words:
                up, xp = w.upos, w.xpos
                upos_counts[up] = upos_counts.get(up, 0) + 1
                xpos_counts[(up, xp)] = xpos_counts.get((up, xp), 0) + 1

        if not upos_counts:
            return None

        inner_labels = list(upos_counts.keys())
        inner_sizes = list(upos_counts.values())
        outer_labels, outer_sizes, outer_colors = [], [], []

        cmap = plt.get_cmap("tab20c")
        inner_colors = cmap(np.linspace(0, 1, len(inner_labels)))

        for i, up in enumerate(inner_labels):
            sub = [(xp, c) for (u, xp), c in xpos_counts.items() if u == up]
            color_slice = cmap(np.linspace(i / len(inner_labels), (i + 1) / len(inner_labels), len(sub)))
            for (xp, c), color in zip(sub, color_slice):
                outer_labels.append(xp)
                outer_sizes.append(c)
                outer_colors.append(color)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('equal')

        ax.pie(
            inner_sizes, radius=1.0, labels=inner_labels, labeldistance=0.8,
            colors=inner_colors, wedgeprops=dict(width=0.3, edgecolor='w')
        )
        ax.pie(
            outer_sizes, radius=1.3, labels=outer_labels, labeldistance=1.05,
            colors=outer_colors, wedgeprops=dict(width=0.3, edgecolor='w')
        )

        plt.title("Part-of-Speech (POS) Distribution", y=1.08)
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_str