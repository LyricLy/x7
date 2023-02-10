import copy
import inspect
import sys
import math
import functools
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

    def peek(self, n=1):
        if len(self.stack) < n:
            raise Raise("pop from empty stack")
        return self.stack[-n:]

    def pop(self, n=1):
        low = len(self.stack) - n
        if low < 0:
            raise Raise("pop from empty stack")
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
        clone = self.clone() if not save else None
        try:
            self.execute(block)
        except Raisoid as e:
            propagate_mask(e)
            self.restore(save.clone() if save else clone)
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
        if name in instructions:
            raise KeyError("instruction name conflict")
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
            r = f(state, *blocks)
        except Raise as r:
            raise RaiseInfo(r.reason, s, og_i, state)
        if r is not None:
            state.stack.append(r)
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
            if v.isdigit():
                print("warning: variables named after digits are unusable", file=sys.stderr)
                print_pos(s, i)
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
View = namedtuple("View", "xs assign depth")
List = namedtuple("List", "type")
Mark = namedtuple("Mark", "x")

NEVER = 0
FROM_TOP = 1
FROM_TOP_PAIR = 2
FROM_SINGLE = 3
TO_SINGLE = 4

TOP = 0
TOP_PAIR = 1
SINGLE = 2
DEEP = 3

def render(val):
    match val:
        case list():
            return f"[{', '.join(render(v) for v in val)}]"
        case Box(v):
            return f"&{render(v)}"
        case View():
            return render(val.assign([Mark(x) for x in val.xs]))
        case Mark(v):
            return f"<{render(v)}>"
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
    top = state.peek(l)
    if any(not compatible(typeof(x), y) for x, y in zip(top, args)):
        raise Raise("type error")
    state.pop(l)
    return top

def assert_int(x):
    if x.denominator != 1:
        raise Raise("non-integer argument")
    return x.numerator

def assert_nonneg_int(x):
    if x < 0:
        raise Raise("negative argument")
    return assert_int(x)

def view_type(v):
    return typeof(v.xs[0]) if v.xs else None

def get_view(state, ty=None, *, drill):
    v, = get_types(state, None)
    if not isinstance(v, View):
        v = View([v], lambda xs: xs[0], TOP)
    while drill > v.depth and (n := flatten_view(v)):
        v = n
    if not compatible(view_type(v), ty) or drill > v.depth and drill != TO_SINGLE:
        raise Raise("type error")
    return v

def flatten_view(v):
    if not compatible(view_type(v), List(None)):
        return None
    xs = []
    ixs = []
    for x in v.xs:
        ixs.append((len(xs), len(xs)+len(x)))
        xs.extend(x)
    assign = lambda ys: v.assign([[x for x in ys[a:b] if x is not None] for a, b in ixs])
    return View(xs, assign, DEEP)

@instruction("+")
def add(state):
    x, y = get_types(state, Fraction, Fraction)
    return x + y

@instruction("-")
def sub(state):
    x, y = get_types(state, Fraction, Fraction)
    return x - y

@instruction("*")
def mul(state):
    x, y = get_types(state, Fraction, Fraction)
    return x * y

@instruction("D")
def true_div(state):
    x, y = get_types(state, Fraction, Fraction)
    try:
        return x / y
    except ZeroDivisionError:
        raise Raise("division by zero")

@instruction("Q")
def int_div(state):
    x, y = get_types(state, Fraction, Fraction)
    assert_int(x)
    assert_int(y)
    try:
        return Fraction(x // y)
    except ZeroDivisionError:
        raise Raise("division by zero")

@instruction("R")
def int_mod(state):
    x, y = get_types(state, Fraction, Fraction)
    assert_int(x)
    assert_int(y)
    try:
        return Fraction(x % y)
    except ZeroDivisionError:
        raise Raise("modulo by zero")

@instruction("N")
def negate(state):
    x, = get_types(state, Fraction)
    return -x

@instruction("J")
def floor(state):
    x, = get_types(state, Fraction)
    return math.floor(x)

@instruction("K")
def ceil(state):
    x, = get_types(state, Fraction)
    return math.ceil(x)

@instruction("<")
def lt(state):
    x, y = get_types(state, None, None)
    if not compatible(typeof(x), typeof(y)):
        raise Raise("type error")
    if not x < y:
        raise Raise("assertion failed")

@instruction("L")
def le(state):
    x, y = get_types(state, None, None)
    if not compatible(typeof(x), typeof(y)):
        raise Raise("type error")
    if not x <= y:
        raise Raise("assertion failed")

@instruction("=")
def eq(state):
    x, y = get_types(state, None, None)
    if not compatible(typeof(x), typeof(y)):
        raise Raise("type error")
    if not x == y:
        raise Raise("assertion failed")

@instruction("/")
def ne(state):
    x, y = get_types(state, None, None)
    if not compatible(typeof(x), typeof(y)):
        raise Raise("type error")
    if not x != y:
        raise Raise("assertion failed")

@instruction("G")
def ge(state):
    x, y = get_types(state, None, None)
    if not compatible(typeof(x), typeof(y)):
        raise Raise("type error")
    if not x >= y:
        raise Raise("assertion failed")

@instruction(">")
def gt(state):
    x, y = get_types(state, None, None)
    if not compatible(typeof(x), typeof(y)):
        raise Raise("type error")
    if not x > y:
        raise Raise("assertion failed")

@instruction("b")
def unbox(state):
    x, = get_types(state, Box)
    return x.val

@instruction("B")
def box(state):
    x, = get_types(state, None)
    return Box(x)

@instruction(",")
def pair(state):
    x, y = get_types(state, None, None)
    return (x, y)

@instruction("[")
def empty(state):
    return []

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
    return x

@instruction("i")
def iota(state):
    n, = get_types(state, Fraction)
    return [Fraction(x) for x in range(assert_nonneg_int(n))]

@instruction("h")
def head(state):
    v = get_view(state, (None, None), drill=TO_SINGLE)
    return View([x[0] for x in v.xs], lambda ys: v.assign([(y, x[1]) for x, y in zip(v.xs, ys)]), max(v.depth, TOP_PAIR))

@instruction("t")
def tail(state):
    v = get_view(state, (None, None), drill=TO_SINGLE)
    return View([x[1] for x in v.xs], lambda ys: v.assign([(x[0], y) for x, y in zip(v.xs, ys)]), max(v.depth, TOP_PAIR))

@instruction("j")
def join(state):
    v = get_view(state, List(None), drill=FROM_TOP)
    return flatten_view(v)

@instruction("n")
def nth(state):
    i, = get_types(state, Fraction)
    i = assert_nonneg_int(i)
    v = get_view(state, List(None), drill=NEVER)
    def assign(xs, x):
        if x is not None:
            xs[i] = x
        else:
            del xs[i]
        return xs
    try:
        return View([x[i] for x in v.xs], lambda ys: v.assign([assign(x, y) for x, y in zip(v.xs, ys)]), max(v.depth, SINGLE))
    except IndexError:
        raise Raise("index out of bounds")

@instruction("s")
def select(state):
    ixs = get_view(state, Fraction, drill=FROM_SINGLE).xs
    v = get_view(state, List(None), drill=NEVER)
    def assign_many(xs, ys):
        for i, y in zip(ixs, ys):
            xs[i.numerator] = y
        return [x for x in xs if x is not None]
    try:
        return View([x[assert_nonneg_int(i)] for x in v.xs for i in ixs], lambda ys: v.assign([assign_many(x, ys[j:j+len(ixs)]) for x, j in zip(v.xs, range(0, len(ys), len(ixs)))]), DEEP)
    except IndexError:
        raise Raise("index out of bounds")

@instruction("w")
def where(state, block):
    v = get_view(state, drill=FROM_SINGLE)
    xs = []
    ixs = []
    for i, x in enumerate(v.xs):
        save = state.clone()
        state.stack.append(x)
        if not state.try_execute(block, save):
            ixs.append(len(xs))
            xs.append(x)
        else:
            ixs.append(None)
    return View(xs, lambda ys: v.assign([ys[j] if j is not None else v.xs[i] for i, j in enumerate(ixs)]), v.depth)

@instruction("@")
def only(state):
    v = get_view(state, drill=FROM_TOP)
    if len(v.xs) != 1:
        raise Raise("wrong number of elements in view")
    return v.xs[0]

@instruction("]")
def enlist(state):
    v = get_view(state, drill=NEVER)
    return v.xs

@instruction("c")
def count(state):
    v = get_view(state, drill=FROM_SINGLE)
    return Fraction(len(v.xs))

@instruction("$")
def set(state):
    x, = get_types(state, None)
    v = get_view(state, drill=FROM_TOP)
    return v.assign([copy.copy(x) for _ in range(len(v.xs))])

@instruction("X")
def eighty_six(state):
    v = get_view(state, drill=FROM_TOP_PAIR)
    return v.assign([None] * len(v.xs))

@instruction("P")
def paste(state):
    b = get_view(state, drill=FROM_TOP_PAIR)
    a = get_view(state, drill=FROM_TOP_PAIR)
    return a.assign(b.xs + [None]*(len(a.xs)-len(b.xs)))

@instruction("u")
def u_turn(state):
    v = get_view(state, drill=FROM_SINGLE)
    return v.assign(v.xs[::-1])

@instruction("M")
def map(state, block):
    v = get_view(state, drill=FROM_TOP)
    res = []
    ty = None
    for x in v.xs:
        save = state.clone()
        state.stack.append(x)
        if not state.try_execute(block, save):
            y, = get_types(state, None)
            y_ty = typeof(y)
            if compatible(y_ty, ty):
                ty = ty or y_ty
                res.append(y)
        else:
            res.append(None)
    return v.assign(res)

@instruction("Z")
def zip_with(state, block):
    a = get_view(state, drill=FROM_TOP)
    b = get_view(state, drill=FROM_TOP)
    res = []
    ty = None
    for x, y in zip(a.xs, b.xs):
        save = state.clone()
        state.stack.append(x)
        state.stack.append(y)
        if not state.try_execute(block, save):
            y, = get_types(state, None)
            y_ty = typeof(y)
            if compatible(y_ty, ty):
                ty = ty or y_ty
                res.append(y)
        else:
            res.append(None)
    res.extend([None]*(len(a.xs)-len(res)))
    return a.assign(res)

@instruction("E")
def each(state, block):
    v = get_view(state, drill=FROM_SINGLE)
    for x in v.xs:
        state.stack.append(x)
        state.execute(block)

@instruction("C")
def choose(state, rest):
    v = get_view(state, drill=FROM_SINGLE)
    save = state.clone()
    for x in v.xs:
        state.stack.append(x)
        if not state.try_execute(rest, save):
            return
    raise Raise("all choices raised")

@instruction("W")
def while_(state, block):
    while True:
        if state.try_execute(block):
            break

@instruction("T")
def times(state, block):
    n, = get_types(state, Fraction)
    for _ in range(assert_nonneg_int(n)):
        state.execute(block)

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

@instruction("q")
def quiet(state, rest):
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
