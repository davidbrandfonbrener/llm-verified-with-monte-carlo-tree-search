from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo

from lang import can_be_solution
from lang import score_func as uncached_score_func

from common_cache import create_cached_func
score_func, cache_stats = create_cached_func(uncached_score_func)
from common_interactive import diffprompt

from prompts import prompt, expansion_count, min_lines, check_func
from common import limit_depth, max_completion_depth
from common_stats import stats

import llm

solution = None

def generate_complete(text, current_completion_depth=1):
    if current_completion_depth >= max_completion_depth:
        return None
    prev = text
    texts = llm.generate(text, 1)
    text = texts[0]
    score = score_func(text)
    print(diffprompt(prev, texts))
    if score is not None:
        if score < 0:
            return (text, score)
        else:
            if can_be_solution(text, min_lines, check_func):
                solution = text
            return (text, score)
    else:
        return generate_complete(text, current_completion_depth + 1)


prev_perfect = prompt
text = prompt
while solution is None:
    (next_text, score) = generate_complete(text)
    if score > 0:
        text = next_text
        if text.count('```') % 2 == 0 or score_func(text+'```') > 0:
            prev_perfect = text
    else:
        text = prev_perfect

print('CHOSEN SOLUTION')
print(solution)
print('cache stats', cache_stats)
