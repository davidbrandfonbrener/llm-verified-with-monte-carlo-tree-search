from cmdline import args

from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo

from lang_config import LANG
from lang import can_be_solution, short_verifier_feedback, filter_code

from lang import score_func as uncached_score_func
from common_cache import create_cached_func
score_func, cache_stats, reset_cache = create_cached_func(uncached_score_func)

from prompts import prompt, expansion_count, min_lines, check_func
from common import limit_depth, max_completion_depth
from common_stats import stats

import llm

import reflection

def generate_complete(text, montecarlo, current_completion_depth=1):
    if current_completion_depth >= max_completion_depth:
        return None
    text = llm.generate(text, 1)[0]
    score = score_func(text)
    if score is not None:
        if score > 0 and can_be_solution(text, min_lines, check_func):
            montecarlo.solution = text
        return (text, score)
    else:
        return generate_complete(text, montecarlo, current_completion_depth + 1)


def child_finder(node, montecarlo):
    if limit_depth(node):
        return

    (text, score) = generate_complete(node.state, montecarlo)
    if score < 0:
        (snippet, err) = short_verifier_feedback(node.state, text)
        if err:
            code = filter_code(text+"```")
            r = reflection.reflect_code(code, snippet, err)
            hint = None
            if r:
                text = "```"+LANG+"\n"+r
                score = score_func(text)
                if score is not None and score > 0:
                    hint = text
        else:
            hint = None
        if not hint:
            node.update_win_value(-1)
            return
        else:
            text = hint
            montecarlo.stats_failed_expansion_count += 1

    child = Node(text)
    node.add_child(child)
    child.update_win_value(1)
    child.update_policy_value(1)

    child = Node(node.state)
    node.add_child(child)
    child.update_policy_value(0.2)

def main(mins_timeout = None):
    montecarlo = MonteCarlo(Node(prompt), mins_timeout)
    montecarlo.child_finder = child_finder

    montecarlo.simulate(expansion_count)

    print("CHOSEN SOLUTION")
    print(montecarlo.solution)

    stats(montecarlo)
    print('cache stats', cache_stats)

    return cache_stats

if __name__ == '__main__':
    main()
