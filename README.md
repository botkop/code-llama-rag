
# project code-llama-rag

This project is a simple library for doing RAG with code-llama.

It consists of 2 classes:
- ContextBuilder
- CodeTalker

The context builder reads a folder with python code upon instantiation,
and parses every python file in the folder structure.
The python files are chunked into pieces of code (class definitions, function definitions, and the rest of the code).
Every chunk is then embedded with a sentence transfomer model.

The code talker uses the code-llama model and tokenizer, and the context builder to answer questions about the code.

The question, or part of the question, is used to determine the context.
The context can be obtained in 3 different ways:
- plain question: 
    - use nearest neighbor search to find the code pieces most similar to the question
    - use the same question for answering
    - eg. "`What is the purpose of the function sir_this_is_a_wendys?`"
- question with a context hint: 
    - use nearest neighbor search to find the code pieces most similar to the context hint
    - remove the context hint and use the remainder of the question for answering
    - eg. "`Wendy: What is the purpose of the function sir_this_is_a_wendys?`"
    - here `Wendy` is the context hint, and `What is the purpose of the function sir_this_is_a_wendys?` is the question
- question with a regular expression context hint:
    - use the regular expression to find the code pieces that match the regular expression
    - remove the context hint and use the remainder of the question for answering
    - eg. "`'^cl..s Wendys?': What is the purpose of the function sir_this_is_a_wendys?`"
    - here `^cl..s Wendys?` is the regex context hint, and `What is the purpose of the function sir_this_is_a_wendys?` is the question. Note the single quotes around the regex context hint.

Regular expression context hints often yield more focused contexts, and better answers.

Note that the full paths of file names are always part of the context, so you can ask questions about those as well, or use them in context hints.


Typical usage is as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from code_llama_rag.context_builder import ContextBuilder
from code_llama_rag.code_talker import CodeTalker

# instantiate the context builder
folder = "code/my-python-project"
context_builder = ContextBuilder(folder=folder)

# instantiate code-llama in 4 bit mode
bnb_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,)
llm_model_id = "codellama/CodeLlama-7b-Instruct-hf"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id, 
    device_map="auto",       
    quantization_config=bnb_config,         
).eval()

# instantiate the code talker
talker = CodeTalker(llm_model, llm_tokenizer, context_builder)

# define the question
question = "What is the purpose of the function sir_this_is_a_wendys?"

# get the answer
answer = talker.answer(question)

print(answer)
```
