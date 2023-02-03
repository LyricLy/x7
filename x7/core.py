import copy
import inspect
import sys
import math
import itertools
from collections import namedtuple
from enum import Enum
from fractions import Fraction


class Raise(Exception):
    def __init__(self, reason):
        self.reason = reason

class Raisoid(Exception):
    pass

class RaiseInfo(Raisoid):
    def __init__(self, reason, s, i, state):
        self.reason = reason
        self.s = s
        self.i = i
        self.state = state

class Mask(Raisoid):
    __match_args__ = ("inner",)

    def __init__(self, inner):
        self.inner = inner

def propagate_mask(e):
    match e:
        case Mask(x):
            raise x

class State:
    def __init__(self, funcs):
        self.funcs = funcs
        self.stack = []
        self.groups = []
        self.last_popped = []
        self.variables = {}

    def _top_group(self):
        if self.groups:
            x, y = self.groups[-1]
            if len(self.stack) == y:
                return x
        if not self.stack:
            raise Raise("pop from empty stack")
        return len(self.stack)-1

    def peek_group(self):
        return [copy.copy(x) for x in self.stack[self._top_group():]]

    def pop_group(self):
        x = self._top_group()
        l = self.stack[x:]
        del self.stack[x:]
        self.last_popped[-1].extend(l)
        return l

    def push_group(self, group):
        if len(group) > 1:
            self.groups.append((len(self.stack), len(self.stack) + len(group)))
        self.stack.extend(group)

    def pop(self, n=1):
        low = len(self.stack) - n
        while self.groups and self.groups[-1][1] > low:
            self.groups.pop()
        self.last_popped[-1].extend(self.stack[low:])
        del self.stack[low:]

    def clone(self):
        return copy.deepcopy(self)

    def restore(self, clone):
        self.stack = clone.stack
        self.groups = clone.groups
        self.variables = clone.variables
        self.last_popped = clone.last_popped

    def execute(self, block):
        for cmd in block:
            cmd(self)

    def try_execute(self, block, save=None):
        save = save or self
        try:
            self.execute(block)
        except Raisoid as e:
            propagate_mask(e)
            self.restore(save.clone())
            return True
        return False

    def mask(self, block):
        try:
            self.execute(block)
        except Raisoid as e:
            raise Mask(e)


Instruction = namedtuple("Instruction", "func blocks")

instructions = {}

def instruction(name):
    def instruction_deco(func):
        instructions[name] = Instruction(func, len(inspect.getfullargspec(func).args)-1)
        return func
    return instruction_deco

def int_literal(n):
    f = Fraction(n)
    def push_(state):
        state.stack.append(f)
    push_.__name__ += str(n)
    push_.__qualname__ += str(n)
    return push_

def get_var(v, s, og_i):
    def get_var(state):
        try:
            state.stack.append(state.variables[v])
        except KeyError:
            raise RaiseInfo("variable not defined", s, og_i+1, state)
    return get_var

def call_func(n, s, og_i):
    def call_(state):
        try:
            state.execute(state.funcs[n-1])
        except IndexError:
            raise RaiseInfo("function not defined", s, og_i+1, state)
    call_.__name__ += str(n)
    call_.__qualname__ += str(n)
    return call_

def set_var(v, s, og_i):
    def set_var(state):
        try:
            state.variables[v] = state.stack.pop()
        except IndexError:
            raise RaiseInfo("pop from empty stack", s, og_i, state)
    return set_var

def run_inst(f, s, blocks, og_i):
    def func(state):
        state.last_popped.append([])
        try:
            f(state, *blocks)
        except Raise as r:
            raise RaiseInfo(r.reason, s, og_i, state)
        state.last_popped.pop()
    func.__name__ = f.__name__
    func.__qualname__ = f.__qualname__
    return func

class Rule(Enum):
    IGNORE = 0
    CONSUME = 1
    SEE = 2

def parse_number(s, i):
    if not s[i].isdigit():
        return None
    n = 0
    while (d := s[i:i+1]).isdigit():
        n = n*10 + int(d)
        i += 1
    return n, i

def parse_block(s, i, close_brackets, backticks=Rule.CONSUME):
    code = []
    while i < len(s) and s[i] != "\n":
        c = s[i]
        og_i = i
        i += 1
        if c == "}":
            match close_brackets:
                case Rule.CONSUME:
                    break
                case Rule.SEE:
                    i -= 1
                    break
        elif c == "`":
            match backticks:
                case Rule.CONSUME:
                    break
                case Rule.SEE:
                    i -= 1
                    break
                case Rule.IGNORE:
                    print("warning: redundant backtick", file=sys.stderr)
                    print_pos(s, og_i)
        elif c == "{":
            block, i = parse_block(s, i, Rule.CONSUME, Rule.SEE)
            code.extend(block)
        elif r := parse_number(s, og_i):
            n, i = r
            code.append(int_literal(n))
        elif c == ";":
            if r := parse_number(s, i):
                n, i = r
                code.append(call_func(n, s, og_i))
            else:
                v = s[i]
                i += 1
                code.append(get_var(v, s, og_i))
        elif c == ":":
            v = s[i]
            i += 1
            code.append(set_var(v, s, og_i))
        elif inst := instructions.get(c):
            blocks = []
            if inst.blocks:
                for _ in range(inst.blocks-1):
                    block, i = parse_block(s, i, Rule.CONSUME)
                    blocks.append(block)
                block, i = parse_block(s, i, Rule.SEE)
                blocks.append(block)
            code.append(run_inst(inst.func, s, blocks, og_i))
        elif c != " ":
            print(f"warning: unknown instruction '{c}'", file=sys.stderr)
    return code, i

def parse_program(s):
    i = 0
    funcs = []
    while i < len(s):
        func, i = parse_block(s, i, Rule.IGNORE, Rule.IGNORE)
        funcs.append(func)
        i += 1
    return funcs

def run_program(s):
    state = State(parse_program(s))
    if state.funcs:
        state.execute(state.funcs[-1])
    return state.stack

def render_stack(stack):
    return " ".join(render(x) for x in stack) if stack else "<empty>"

def print_pos(s, i):
    l = s.rfind("\n", 0, i)+1
    r = s.find("\n", i)
    line = s[l:r if r != -1 else None]
    print(line, file=sys.stderr)
    print(" "*(i - l) + "^", file=sys.stderr)

def print_raise(e):
    print(f"Instruction raised: {e.reason}", file=sys.stderr)
    stack = e.state.stack.copy()
    for p in e.state.last_popped:
        stack.extend(p)
    print(f"stack: {render_stack(stack)}", file=sys.stderr)
    print_pos(e.s, e.i)


Box = namedtuple("Box", "val")
List = namedtuple("List", "type")

def render(val):
    match val:
        case list():
            return f"[{', '.join(render(v) for v in val)}]"
        case x, y:
            return f"({render(x)}, {render(y)})"
        case _:
            return str(val)

def typeof(val):
    match val:
        case list([_, *_]):
            return List(typeof(val[0]))
        case list():
            return List(None)
        case a, b:
            return typeof(a), typeof(b)
        case _:
            return type(val)

def compatible(x, y):
    match x, y:
        case (None, _) | (_, None):
            return True
        case List(x), List(y):
            return compatible(x, y)
        case (x1, y1), (x2, y2):
            return compatible(x1, x2) and compatible(y1, y2)
        case _:
            return x == y

def get_types(state, *args):
    l = len(args)
    top = state.stack[-l:]
    if len(top) != l:
        raise Raise("pop from empty stack")
    elif any(not compatible(typeof(x), y) for x, y in zip(top, args)):
        raise Raise("type error")
    state.pop(l)
    return top


@instruction("+")
def add(state):
    x, y = get_types(state, Fraction, Fraction)
    state.stack.append(x + y)

@instruction("-")
def sub(state):
    x, y = get_types(state, Fraction, Fraction)
    state.stack.append(x - y)

@instruction("*")
def mul(state):
    x, y = get_types(state, Fraction, Fraction)
    state.stack.append(x * y)

@instruction("D")
def true_div(state):
    x, y = get_types(state, Fraction, Fraction)
    try:
        state.stack.append(x / y)
    except ZeroDivisionError:
        raise Raise("division by zero")

@instruction("Q")
def int_div(state):
    x, y = get_types(state, Fraction, Fraction)
    if x.denominator != 1 or y.denominator != 1:
        raise Raise("arguments must be integers")
    try:
        state.stack.append(Fraction(x // y))
    except ZeroDivisionError:
        raise Raise("division by zero")

@instruction("R")
def int_mod(state):
    x, y = get_types(state, Fraction, Fraction)
    if x.denominator != 1 or y.denominator != 1:
        raise Raise("arguments must be integers")
    try:
        state.stack.append(Fraction(x % y))
    except ZeroDivisionError:
        raise Raise("modulo by zero")

@instruction("F")
def floor(state):
    x, = get_types(state, Fraction)
    state.stack.append(math.floor(x))

@instruction("C")
def ceil(state):
    x, = get_types(state, Fraction)
    state.stack.append(math.ceil(x))

@instruction("<")
def lt(state):
    x, y = get_types(state, None, None)
    if not x < y:
        raise Raise("assertion failed")

@instruction("L")
def le(state):
    x, y = get_types(state, None, None)
    if not x <= y:
        raise Raise("assertion failed")

@instruction("=")
def eq(state):
    x, y = get_types(state, None, None)
    if not x == y:
        raise Raise("assertion failed")

@instruction("/")
def ne(state):
    x, y = get_types(state, None, None)
    if not x != y:
        raise Raise("assertion failed")

@instruction("G")
def ge(state):
    x, y = get_types(state, None, None)
    if not x >= y:
        raise Raise("assertion failed")

@instruction(">")
def gt(state):
    x, y = get_types(state, None, None)
    if not x > y:
        raise Raise("assertion failed")

@instruction("b")
def unbox(state):
    x, = get_types(state, Box)
    state.stack.append(x.val)

@instruction("B")
def box(state):
    x, = get_types(state, None)
    state.stack.append(Box(x))

@instruction(",")
def pair(state):
    x, y = get_types(state, None, None)
    state.stack.append((x, y))

@instruction("%")
def unpair(state):
    x, = get_types(state, (None, None))
    state.stack.extend(x)

@instruction("[")
def empty(state):
    state.stack.append([])

@instruction("]")
def enlist(state):
    state.stack.append(get_types(state, None))

@instruction(".")
def concat(state):
    x, y = get_types(state, None, None)
    match typeof(x), typeof(y):
        case List(a), List(b) if compatible(a, b):
            x.extend(y)
        case List(a), b if compatible(a, b):
            x.append(y)
        case a, List(b) if compatible(a, b):
            x = [x] + y
        case a, b if compatible(a, b):
            x = [x, y]
        case _:
            raise Raise("types incompatible")
    state.stack.append(x)

@instruction("i")
def iota(state):
    n, = get_types(state, Fraction)
    if n.denominator != 1 or n < 0:
        raise Raise("argument must be a nonnegative integer")
    state.stack.append([Fraction(x) for x in range(n.numerator)])

@instruction("W")
def while_(state, block):
    while True:
        if state.try_execute(block):
            break

@instruction("F")
def for_(state, block):
    l, = get_types(state, List(None))
    for x in l:
        state.stack.append(x)
        state.execute(block)

@instruction("M")
def map(state, block):
    l, = get_types(state, List(None))
    res = []
    ty = None
    for x in l:
        save = state.clone()
        state.stack.append(x)
        if not state.try_execute(block, save):
            v, = get_types(state, None)
            v_ty = typeof(v)
            if compatible(v_ty, ty):
                ty = ty or v_ty
                res.append(v)
    state.stack.append(res)

@instruction("T")
def times(state, block):
    n, = get_types(state, Fraction)
    if n.denominator != 1 or n < 0:
        raise Raise("argument must be a nonnegative integer")
    for _ in range(int(n)):
        state.execute(block)

@instruction("P")
def pick(state, rest):
    l, = get_types(state, List(None))
    save = state.clone()
    for x in l:
        state.stack.append(x)
        if not state.try_execute(rest, save):
            return
    raise Raise("all choices raised")

@instruction("&")
def tie(state):
    y, x = state.pop_group(), state.pop_group()
    state.push_group(x + y)

@instruction("d")
def dup(state):
    state.push_group(state.peek_group())

@instruction("p")
def pop(state):
    state.pop()

@instruction("_")
def under(state, rest):
    x = state.pop_group()
    state.execute(rest)
    state.push_group(x)

@instruction("l")
def lift(state, b1, b2):
    save = state.clone()
    state.execute(b2)
    g = state.pop_group()
    state.restore(save)
    state.execute(b1)
    state.push_group(g)

@instruction("f")
def flip(state):
    y, x = state.pop_group(), state.pop_group()
    state.push_group(y)
    state.push_group(x)

@instruction("^")
def yank(state):
    y, x = state.pop_group(), state.peek_group()
    state.push_group(y)
    state.push_group(x)

@instruction("~")
def permute(state, rest):
    save = state.clone()
    state.groups = []
    for perm in itertools.permutations(state.stack):
        state.stack[:] = perm
        if not state.try_execute(rest, save):
            return
    raise Raise("all permutations raised")

@instruction("r")
def raise_(state):
    raise Raise("explicit raise")

@instruction("e")
def except_(state, rest, except_):
    if state.try_execute(rest):
        state.execute(except_)

@instruction("s")
def suppress(state, rest):
    state.try_execute(rest)

@instruction("!")
def invert(state, rest):
    # no rewinding
    try:
        state.execute(rest)
    except Raisoid as e:
        propagate_mask(e)
    else:
        raise Raise("block didn't raise")

@instruction("m")
def mask(state, rest):
    state.mask(rest)

@instruction("v")
def debug(state):
    print(f"stack: {render_stack(state.stack)}", file=sys.stderr)

@instruction("V")
def debug_raises(state, rest):
    try:
        state.execute(rest)
    except Raisoid as r:
        propagate_mask(r)
        print_raise(r)
        raise


def run():
    with open(sys.argv[1]) as f:
        s = f.read()
    try:
        print(render_stack(run_program(s)))
    except Raisoid as r:
        while isinstance(r, Mask):
            r = r.inner
        print_raise(r)
        sys.exit(1)
