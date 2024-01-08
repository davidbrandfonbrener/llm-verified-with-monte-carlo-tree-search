from lang_config import LANG
from common_interactive import diffprompt

import llm

def generate_full(prompt: str, **kwargs) -> str:
    return llm.generate_full(prompt, max_new_tokens=100, **kwargs)

def reflect(code: str, snippet: str, err: str) -> str:
    # adapted from F.3 Reflection Prompt
    # in https://arxiv.org/pdf/2310.04406v2.pdf
    prompt = f"""<s>[INST] <<SYS>>
You are a {LANG} programming assistant. You will be given some code and an error. Your goal is to write a few sentences to explain why the is wrong as indicated by the error. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation.
<</SYS>>

```{LANG}
{code}
```

The error is:
{err}
in the snippet:
{snippet}

[/INST]
"""
    r = generate_full(prompt)
    return diffprompt(prompt, [r])[0]
