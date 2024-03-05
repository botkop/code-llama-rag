import torch


class CodeTalker:
    def __init__(
        self, model, tokenizer, context_builder, max_tokens_in_prompt=8000, avg_chars_per_token=3
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.context_builder = context_builder
        self.model.eval()

        self.avg_chars_per_token = avg_chars_per_token
        self.max_tokens_in_prompt = max_tokens_in_prompt  # max tokens that fit in 40GB GPU memory
        self.n_tokens_in_prompt_template = len(self.tokenizer.encode(self.make_prompt("", "")))

    def make_prompt(self, question, context):
        n_tokens_in_question = len(self.tokenizer.encode(question))
        if context and self.max_tokens_in_prompt:
            max_context_tokens = self.max_tokens_in_prompt - n_tokens_in_question - self.n_tokens_in_prompt_template
            max_context_length = max_context_tokens * self.avg_chars_per_token
            context = context[:max_context_length]
        return f"""You are given the <CONTEXT/> of python code and a <QUESTION/>. ANSWER the QUESTION.
<CONTEXT>
{context}
</CONTEXT>
<QUESTION>
{question}
</QUESTION>
<ANSWER>
"""

    def answer(self, question, max_new_tokens=300, truncate_answer=True):
        # if the question contains a : then split the question and use the first part for obtaining the context
        if ":" in question:
            context_question, question = question.split(":", 1)
        else:
            context_question = question

        context = self.context_builder.get_context(context_question)
        prompt = self.make_prompt(question, context)
        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        n_tokens = model_input.input_ids.shape[1]

        print(f"number of characters in prompt: {len(prompt)}")
        print(f"number of tokens in prompt: {n_tokens}")
        print(f"chars per token: {len(prompt) / n_tokens}")

        with torch.no_grad():
            output = self.model.generate(
                **model_input, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0, n_tokens:]
            output = self.tokenizer.decode(output, skip_special_tokens=True)
            if truncate_answer:
                output = output.split("</ANSWER>")[0]
        return output
