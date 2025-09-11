import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Iterable


class Visualizer:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def generate_word_cloud(
        self,
        frequencies: Dict[str, int],
        width: int = 800,
        height: int = 400,
        background_color: str = "white",
        max_words: int = 200,
        collocations: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        if not frequencies:
            return None

        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            collocations=collocations,
            random_state=self.random_state,
            mask=mask,
        ).generate_from_frequencies(frequencies)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=200)
        plt.close()

        img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return img_str

    def _collect_entity_types(self, classla_doc, include_types: Optional[Iterable[str]]) -> List[str]:
        if include_types is not None:
            types = list(include_types)
        else:
            found = set()
            for s in getattr(classla_doc, "sentences", []):
                for e in getattr(s, "ents", []):
                    if getattr(e, "type", None):
                        found.add(e.type)
            types = sorted(found) if found else ["PER", "LOC", "ORG"]
        return types

    def generate_ner_heatmap(
        self,
        classla_doc,
        normalize: bool = False,
        include_types: Optional[Iterable[str]] = None,
        cmap: str = "magma",
    ) -> Optional[str]:
        sentences = getattr(classla_doc, "sentences", [])
        num_sents = len(sentences)
        if num_sents == 0:
            return None

        entity_types = self._collect_entity_types(classla_doc, include_types)
        num_types = len(entity_types)

        counts = np.zeros((num_types, num_sents), dtype=float)
        sent_lengths = np.zeros((num_sents,), dtype=int)

        for si, sentence in enumerate(sentences):
            sent_lengths[si] = max(1, len(getattr(sentence, "words", [])))
            for ent in getattr(sentence, "ents", []):
                et = getattr(ent, "type", None)
                if et in entity_types:
                    ti = entity_types.index(et)
                    counts[ti, si] += 1.0

        if normalize:
            counts = counts / sent_lengths[None, :]

        fig_w = max(6, num_sents * 1.2)
        fig_h = max(3, num_types * 0.7)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(counts, cmap=cmap, aspect="auto")

        ax.set_xticks(np.arange(num_sents))
        ax.set_xticklabels([f"Sent {i+1}" for i in range(num_sents)], rotation=45, ha="right")
        ax.set_yticks(np.arange(num_types))
        ax.set_yticklabels(entity_types)

        ax.set_xticks(np.arange(-0.5, num_sents, 1), minor=True)
        ax.grid(which="minor", axis="x", linestyle=":", linewidth=0.5)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Entity Density" if normalize else "Entity Count")
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()

        ax.set_title("NER Heatmap by Sentence")
        ax.set_xlabel("Sentence")
        ax.set_ylabel("Entity Type")

        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=200)
        plt.close()
        img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return img_str

    def generate_pos_sunburst(self, classla_doc) -> Optional[str]:
        upos_counts: Dict[str, int] = {}
        xpos_counts: Dict[Tuple[str, str], int] = {}

        for sent in getattr(classla_doc, "sentences", []):
            for w in getattr(sent, "words", []):
                up = getattr(w, "upos", None) or "X"
                xp = getattr(w, "xpos", None) or "X"
                upos_counts[up] = upos_counts.get(up, 0) + 1
                xpos_counts[(up, xp)] = xpos_counts.get((up, xp), 0) + 1

        if not upos_counts:
            return None

        inner_items = sorted(upos_counts.items(), key=lambda kv: kv[1], reverse=True)
        inner_labels = [k for k, _ in inner_items]
        inner_sizes = [v for _, v in inner_items]

        outer_labels: List[str] = []
        outer_sizes: List[int] = []

        for up in inner_labels:
            sub = [(xp, c) for (u, xp), c in xpos_counts.items() if u == up]
            sub_sorted = sorted(sub, key=lambda kv: kv[1], reverse=True)
            for xp, c in sub_sorted:
                outer_labels.append(xp)
                outer_sizes.append(c)

        cmap = plt.get_cmap("tab20c")
        inner_colors = cmap(np.linspace(0, 1, len(inner_labels)))

        outer_colors: List = []
        offset = 0
        for i, up in enumerate(inner_labels):
            n_children = sum(1 for (u, xp) in xpos_counts.keys() if u == up)
            if n_children == 0:
                continue
            color_slice = cmap(np.linspace(i / max(1, len(inner_labels)), (i + 1) / max(1, len(inner_labels)), n_children))
            outer_colors.extend(color_slice)
            offset += n_children

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("equal")

        ax.pie(
            inner_sizes,
            radius=1.0,
            labels=inner_labels,
            labeldistance=0.8,
            colors=inner_colors,
            wedgeprops=dict(width=0.3, edgecolor="w"),
            textprops={"fontsize": 10},
        )

        if outer_sizes:
            ax.pie(
                outer_sizes,
                radius=1.3,
                labels=outer_labels,
                labeldistance=1.05,
                colors=outer_colors[:len(outer_sizes)],
                wedgeprops=dict(width=0.3, edgecolor="w"),
                textprops={"fontsize": 9},
            )

        plt.title("Part-of-Speech (POS) Distribution", y=1.06)
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=200)
        plt.close()
        img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return img_str
