import ast
import os
import re
from typing import Generator, Union

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


class CodeChunker(ast.NodeVisitor):
    """This class reads a file and returns a list of code chunks.
    A code chunk is a dictionary with the following keys:
    - file: the filename
    - code: the code
    - type: the type of the code (e.g. function, class, rubble)
    Note that this is a shallow parser, and we only visit the top level function and class nodes, not their children.
    """

    def __init__(self, filename: str):
        """
        Initialize the code chunker.

        :param filename: the filename
        """
        self.filename = filename
        with open(filename, "r", encoding="utf-8") as f:
            self.code = f.read()
        self.lines = self.code.splitlines()
        self.locations = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a function definition node and append its location to the locations list.
        Note that this is a shallow parser, and we only visit this node, not its children.

        :param node: the node
        """
        start_lineno = node.lineno - 1
        end_lineno = node.end_lineno
        self.locations.append((start_lineno, end_lineno, type(node).__name__))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit a class definition node and append its location to the locations list.
        Note that this is a shallow parser, and we only visit this node, not its children.

        :param node: the node
        """
        start_lineno = node.lineno - 1
        end_lineno = node.end_lineno
        self.locations.append((start_lineno, end_lineno, type(node).__name__))

    def visit_With(self, node: ast.With) -> None:
        start_lineno = node.lineno - 1
        end_lineno = node.end_lineno
        self.locations.append((start_lineno, end_lineno, type(node).__name__))

    def add_chunk(
        self, chunks: list, start_lineno: int, end_lineno: int, node_type: str
    ) -> None:
        """
        Add a code chunk to the chunks list.

        :param chunks: the list of chunks
        :param start_lineno: the start line number
        :param end_lineno: the end line number
        :param node_type: the type of the code (e.g. function, class, rubble)
        """
        code = "\n".join(self.lines[start_lineno:end_lineno])
        if code.strip():
            chunk = {"file": self.filename, "code": code, "type": node_type}
            chunks.append(chunk)

    def get_chunks(self) -> list:
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
    This class takes a folder and a pattern and returns a dataframe with code chunks.
    """

    def __init__(self, folder: str, pattern: str = r".*\.py$"):
        """
        Initialize the folder chunker.

        :param folder: the folder
        :param pattern: the filename pattern to search for (e.g. r".*\.py$")
        """
        self.folder = folder
        self.pattern = pattern
        self.chunker = CodeChunker

    @classmethod
    def find_files(cls, directory: str, pattern: str) -> Generator:
        """
        Find files in a directory that match a pattern.

        :param directory: the directory
        :param pattern: the filename pattern to search for (e.g. r".*\.py$")
        :return: a generator with the filenames
        """
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if re.search(pattern, basename):
                    filename = os.path.join(root, basename)
                    yield filename

    def chunk_folder(self) -> pd.DataFrame:
        """
        Chunk the folder and return a dataframe with the code chunks.
        The dataframe has the following columns:
        - file: the filename
        - code: the code
        - type: the type of the code (e.g. function, class, rubble)

        :return: the dataframe
        """
        chunks = []
        for file in self.find_files(self.folder, self.pattern):
            chunker = self.chunker(file)
            chunks.extend(chunker.get_chunks())
        if len(chunks) == 0:
            raise ValueError(f"No code chunks found in {self.folder}")
        df = pd.DataFrame(chunks)
        return df


class ContextBuilder:
    """
    This class builds a context for a code search query.
    """

    def __init__(
        self,
        folder: str,
        include_rubble: bool = False,
        model_or_model_name: Union[
            str | SentenceTransformer
        ] = "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    ):
        """
        Initialize the context builder.

        :param folder: the folder with the code
        :param include_rubble: whether to include rubble in the context
        :param model_or_model_name: the sentence transformer model or its name
        """
        self.folder = folder
        self.chunk_df = FolderChunker(self.folder).chunk_folder()
        if not include_rubble:
            self.chunk_df = self.chunk_df[self.chunk_df["type"] != "rubble"]
        if isinstance(model_or_model_name, str):
            self.model = SentenceTransformer(model_or_model_name)
        else:
            self.model = model_or_model_name
        self.code_embeddings = self.model.encode(self.chunk_df["code"].values.tolist())

    def assemble_context_from_df(self, df: pd.DataFrame) -> str:
        """Assemble the context from a dataframe.
        The context is a string that contains the filename and the code of each chunk in the dataframe.

        :param df: the dataframe
        :return: the context
        """
        context = "@@@File: " + df["file"] + "\n" + df["code"]
        context = "\n\n".join(context)
        return context

    def get_context_from_regex(self, regex: str) -> str:
        """Get the context for a regex.
        The context is the code chunks that match the regex.

        :param regex: the regex
        :return: the context
        """
        matches = self.chunk_df[
            self.chunk_df["code"].str.contains(regex)
            | self.chunk_df["file"].str.contains(regex)
        ]
        return self.assemble_context_from_df(matches)

    def get_context_from_embedding(self, query, top_k=10, min_score=0.25):
        """Get the context for a query using the embedding.
        The question is embedded using the same model that was used to embed the code chunks.
        The context is the code chunks that are most similar to the question.

        :param query: the query
        :param top_k: the max number of code chunks to return
        :param min_score: the minimum similarity score to consider a code chunk

        :return: the context
        """
        query_embedding = self.model.encode(query)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.code_embeddings)[0]
        cos_scores = cos_scores.cpu()

        top_k = min(top_k, len(self.chunk_df))
        top_results = torch.topk(cos_scores, k=top_k)
        top_idxs = [idx.item() for idx in top_results[1] if cos_scores[idx] > min_score]

        # sort top_idxs so that chunks from same file are listed in order, regardless of score ???

        selected_chunks_df = self.chunk_df.iloc[top_idxs]
        return self.assemble_context_from_df(selected_chunks_df)

    def get_context(self, query, top_k=10, min_score=0.25):
        """
        Get the context for a query.
        If the query is a regex (string within single quotes), return the context obtained from regex matching.
        Otherwise, return the context from the embedding.

        :param query: the query
        :param top_k: the number of code chunks to return
        :param min_score: the minimum similarity score to consider a code chunk

        :return: the context
        """
        pattern = r"^'([^']+)'"
        match = re.search(pattern, query)
        if match:
            regex = match.group(1)
            return self.get_context_from_regex(regex)
        else:
            return self.get_context_from_embedding(query, top_k, min_score)
