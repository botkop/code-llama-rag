import ast
import os
import re

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


class CodeChunker(ast.NodeVisitor):
    """This class takes a filename and a code string and returns a list of code chunks.
    A code chunk is a dictionary with the following keys:
    - file: the filename
    - code: the code
    - type: the type of the code (e.g. function, class, rubble)
    Note that this is a shallow parser, and we only visit the top level function and class nodes, not their children.
    """

    def __init__(self, filename, code):
        self.filename = filename
        self.code = code
        self.lines = self.code.splitlines()
        self.locations = []

    def visit_FunctionDef(self, node):
        """
        Visit a function definition node and append its location to the locations list.
        Note that this is a shallow parser, and we only visit this node, not its children.
        """
        start_lineno = node.lineno - 1
        end_lineno = node.end_lineno
        self.locations.append((start_lineno, end_lineno, type(node).__name__))

    def visit_ClassDef(self, node):
        """
        Visit a class definition node and append its location to the locations list.
        Note that this is a shallow parser, and we only visit this node, not its children.
        """
        start_lineno = node.lineno - 1
        end_lineno = node.end_lineno
        self.locations.append((start_lineno, end_lineno, type(node).__name__))

    def add_chunk(self, chunks, start_lineno, end_lineno, node_type):
        """Add a code chunk if not empty."""
        code = "\n".join(self.lines[start_lineno:end_lineno])
        if code.strip():
            chunk = {"file": self.filename, "code": code, "type": node_type}
            chunks.append(chunk)

    def get_chunks(self):
        """Get the code chunks."""
        tree = ast.parse(self.code)
        self.visit(tree)
        chunks = []

        # in self.locations we now have the start and end positions of functions and classes
        # now let's find the locations of the rest of the code
        last_end = 0
        for start_lineno, end_lineno, node_type in self.locations:
            self.add_chunk(chunks, last_end, start_lineno, "rubble")
            self.add_chunk(chunks, start_lineno, end_lineno, node_type)
            last_end = end_lineno

        # add the last chunk
        self.add_chunk(chunks, last_end, len(self.lines), "rubble")
        return chunks


class FolderChunker:
    """
    This class takes a folder and a pattern and returns a dataframe with the code chunks.
    """

    def __init__(self, folder, pattern=r".*\.py$"):
        self.folder = folder
        self.pattern = pattern
        self.chunker = CodeChunker

    # recursively find python files in a folder
    @classmethod
    def find_files(cls, directory, pattern):
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if re.search(pattern, basename):
                    filename = os.path.join(root, basename)
                    yield filename

    def chunk_folder(self):
        chunks = []
        for file in self.find_files(self.folder, self.pattern):
            with open(file, "r", encoding="utf-8") as f:
                code = f.read()
                chunker = self.chunker(file, code)
                chunks.extend(chunker.get_chunks())
        df = pd.DataFrame(chunks)
        return df


class ContextBuilder:
    def __init__(
        self,
        folder,
        include_rubble=False,
        model_or_model_name="flax-sentence-embeddings/st-codesearch-distilroberta-base",
    ):
        self.folder = folder
        self.chunk_df = FolderChunker(self.folder).chunk_folder()
        self.chunk_df = self.chunk_df[self.chunk_df["type"] != "rubble"] if not include_rubble else self.chunk_df
        if isinstance(model_or_model_name, str):
            self.model = SentenceTransformer(model_or_model_name)
        else:
            self.model = model_or_model_name
        self.code_embeddings = self.model.encode(self.chunk_df["code"].values.tolist())

    def get_context_from_regex(self, regex):
        matches = self.chunk_df[self.chunk_df["code"].str.contains(regex) | self.chunk_df["file"].str.contains(regex)]
        matches.loc[:, "code"] = "@@@File: " + matches["file"] + "\n" + matches["code"]
        context = matches["code"].values.tolist()
        context = "\n\n".join(context)
        return context

    def get_context_from_embedding(self, query, top_k=10, min_score=0.25):
        query_embedding = self.model.encode(query)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.code_embeddings)[0]
        cos_scores = cos_scores.cpu()

        top_k = min(top_k, len(self.chunk_df))
        top_results = torch.topk(cos_scores, k=top_k)
        top_idxs = [idx.item() for idx in top_results[1] if cos_scores[idx] > min_score]

        # sort top_idxs so that chunks from same file are listed in order, regardless of score ???

        selected_chunks_df = self.chunk_df.iloc[top_idxs]
        selected_chunks_df.loc[:, "code"] = "@@@File: " + selected_chunks_df["file"] + "\n" + selected_chunks_df["code"]
        context = selected_chunks_df["code"].values.tolist()
        context = "\n\n".join(context)
        return context

    def get_context(self, query, top_k=10, min_score=0.25):
        """
        Get the context for a query.
        If the query is a regex, return the context for the regex. Otherwise, return the context for the embedding.
        """
        pattern = r"^'([^']+)'"
        match = re.search(pattern, query)
        if match:
            regex = match.group(1)
            return self.get_context_from_regex(regex)
        else:
            return self.get_context_from_embedding(query, top_k, min_score)
