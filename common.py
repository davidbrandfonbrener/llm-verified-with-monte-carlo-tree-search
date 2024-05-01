from prompts import max_depth
import llm

from cmdline import args

max_completion_depth = args.max_completion_depth


def count_depth(node, f=lambda x: x):
    depth = 1
    curr = node

    while curr.parent is not None:
        if f(curr.state) != f(curr.parent.state):
            depth += 1
        curr = curr.parent

    return depth


# Prevents models from attempting to expand children past the EOS token
def string_contains_eos(node, f=lambda x: x):
    return llm.eos_token in f(node.state)


def limit_depth(node, f=lambda x: x):
    if max_depth is not None:
        depth = count_depth(node, f)
        if depth >= max_depth or string_contains_eos(node, f):
            node.update_win_value(-max_depth)
            return True
    return False


def limit_tokens(montecarlo=None):
    # Force montecarlo to exit if token limit is reached by setting solution
    # Note: call this at the end of child finder
    if args.token_limit is not None and llm.token_counter > args.token_limit:
        if montecarlo is not None and montecarlo.solution is None:
            montecarlo.solution = "Token limit reached"
        print("Token limit reached, no solution found")
        return True
    else:
        return False
