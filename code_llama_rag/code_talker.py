import torch
from transformers.models.code_llama.tokenization_code_llama_fast import (
    CodeLlamaTokenizerFast,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from code_llama_rag.context_builder import ContextBuilder


class CodeTalker:
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: CodeLlamaTokenizerFast,
        context_builder: ContextBuilder,
        max_tokens_in_prompt: int = 8000,
        avg_chars_per_token: int = 3,
    ):
        """
        Initialize the code talker.

        :param model: The model to use for generating answers.
        :param tokenizer: The tokenizer to use for encoding inputs.
        :param context_builder: The context builder to use for obtaining context.
        :param max_tokens_in_prompt: The maximum number of tokens in the prompt.
        :param avg_chars_per_token: The average number of characters per token.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_builder = context_builder
        self.model.eval()

        self.avg_chars_per_token = avg_chars_per_token
        self.max_tokens_in_prompt = max_tokens_in_prompt
        self.n_tokens_in_prompt_template = len(
            self.tokenizer.encode(self.make_prompt("", ""))
        )

    def make_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for the model to generate an answer to the question.

        :param question: The question to ask the model.
        :param context: The context to provide to the model.
        :return: The prompt to provide to the model.
        """
        n_tokens_in_question = len(self.tokenizer.encode(question))
        if context and self.max_tokens_in_prompt:
            max_context_tokens = (
                self.max_tokens_in_prompt
                - n_tokens_in_question
                - self.n_tokens_in_prompt_template
            )
            max_context_length = max_context_tokens * self.avg_chars_per_token
            context = context[:max_context_length]
        prompt = f"""You are given the <CONTEXT/> of python code and a <QUESTION/>. ANSWER the QUESTION.
<CONTEXT>
{context}
</CONTEXT>
<QUESTION>
{question}
</QUESTION>
<ANSWER>
"""
        return prompt

    def answer(
        self, question: str, max_new_tokens: int = 300, truncate_answer: bool = True
    ) -> str:
        """
        Generate an answer to a question.

        The question parameter can contain a hint for the context. If the question contains a colon, the part before the
        colon is used as the context hint. The context is obtained from the context builder.

        :param question: The question to ask the model.
        :param max_new_tokens: The maximum number of tokens to generate in the answer.
        :param truncate_answer: Whether to truncate the answer at the first </ANSWER> tag.
        :return: The answer to the question.
        """
        # if the question contains a :: then split the question and use the first part for obtaining the context
        if "::" in question:
            context_hint, question = question.split("::", 1)
        else:
            context_hint = question

        context = self.context_builder.get_context(context_hint)
        prompt = self.make_prompt(question, context)
        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        n_tokens = model_input.input_ids.shape[1]

        print(f"number of characters in prompt: {len(prompt)}")
        print(f"number of tokens in prompt: {n_tokens}")
        print(f"chars per token: {len(prompt) / n_tokens}")

        with torch.no_grad():
            output = self.model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            output = output[0, n_tokens:]
            output = self.tokenizer.decode(output, skip_special_tokens=True)
            if truncate_answer:
                output = output.split("</ANSWER>")[0]
        return output
