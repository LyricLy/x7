import copy
import inspect
import sys
import itertools
from collections import namedtuple
from enum import Enum
from fractions import Fraction


class Raise(Exception):
    pass

class Mask(Exception):
    pass

def catch_raise(e):
    match e:
        case Raise():
            return True
        case Mask(args=[x]):
            raise x
    return False

class RewindManager:
    def __init__(self, state, mask=False):
        self.state = state
        self.mask = mask

    def __enter__(self):
        self.clone = self.state.clone()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.mask and exc_type in (Raise, Mask):
            raise Mask(exc_value)
        if catch_raise(exc_value):
            self.rewound = True
            self.state.restore(self.clone)
            return True
        else:
            self.rewound = False

class State:
    def __init__(self):
        self.stack = []
        self.groups = []

    def _top_group(self):
        if self.groups:
            x, y = self.groups[-1]
            if len(self.stack) == y:
                return x
        if not self.stack:
            raise Raise
        return len(self.stack)-1

    def peek_group(self):
        return self.stack[self._top_group():]

    def pop_group(self):
        x = self._top_group()
        l = self.stack[x:]
        del self.stack[x:]
        return l

    def push_group(self, group):
        if len(group) > 1:
            self.groups.append((len(self.stack), len(self.stack) + len(group)))
        self.stack.extend(group)

    def pop(self, n=1):
        low = len(self.stack) - n
        while self.groups and self.groups[-1][1] > low:
            self.groups.pop()
        del self.stack[low:]

    def clone(self):
        return copy.deepcopy(self)

    def restore(self, clone):
        self.stack = clone.stack
        self.groups = clone.groups

    def execute(self, block):
        for cmd in block:
            cmd(self)

    def rewind(self):
        return RewindManager(self)

    def try_execute(self, block):
        with self.rewind() as r:
            self.execute(block)
        return r.rewound

    def mask(self, block):
        with RewindManager(self, True):
            self.execute(block)


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

class Rule(Enum):
    IGNORE = 0
    CONSUME = 1
    SEE = 2

def parse_block(s, i, close_brackets, backticks=Rule.CONSUME):
    code = []
    while i < len(s):
        c = s[i]
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
        elif c == "{":
            block, i = parse_block(s, i, Rule.CONSUME, Rule.SEE)
            code.extend(block)
        elif c.isdigit():
            i -= 1
            n = 0
            while (d := s[i:i+1]).isdigit():
                n = n*10 + int(d)
                i += 1
            code.append(int_literal(n))
        elif inst := instructions.get(c):
            blocks = []
            if inst.blocks:
                for _ in range(inst.blocks-1):
                    block, i = parse_block(s, i, Rule.CONSUME)
                    blocks.append(block)
                block, i = parse_block(s, i, Rule.SEE)
                blocks.append(block)
            func = lambda s, f=inst.func, blocks=blocks: f(s, *blocks)
            func.__name__ = inst.func.__name__
            func.__qualname__ = inst.func.__qualname__
            code.append(func)
        elif not c.isspace():
            print(f"warning: unknown instruction '{c}'", file=sys.stderr)
    return code, i

def parse_program(s):
    func, i = parse_block(s, 0, Rule.IGNORE, Rule.SEE)
    if s[i:i+1] not in ("", "}"):
        print("error: unexpected `", file=sys.stderr)
        sys.exit(1)
    return func

def run_program(s):
    state = State()
    state.execute(parse_program(s))
    return state.stack


Box = namedtuple("Box", "val")
List = namedtuple("List", "type")

def render(val):
    match val:
        case list():
            return f"[{', '.join(render(v) for v in val)}]"
        case _:
            return str(val)

def typeof(val):
    match val:
        case list([_]):
            return List(typeof(val[0]))
        case list():
            return List(None)
        case _:
            return type(val)

def compatible(x, y):
    match x, y:
        case (None, _) | (_, None):
            return True
        case List(x), List(y):
            return compatible(x, y)
        case _:
            return x == y

def get_types(state, *args):
    l = len(args)
    top = state.stack[-l:]
    if len(top) != l or any(not compatible(typeof(x), y) for x, y in zip(top, args)):
        raise Raise
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
        raise Raise

@instruction("/")
def int_div(state):
    x, y = get_types(state, Fraction, Fraction)
    if x.denominator != 1 or y.denominator != 1:
        raise Raise
    try:
        state.stack.append(Fraction(x // y))
    except ZeroDivisionError:
        raise Raise

@instruction("%")
def int_mod(state):
    x, y = get_types(state, Fraction, Fraction)
    if x.denominator != 1 or y.denominator != 1:
        raise Raise
    try:
        state.stack.append(Fraction(x % y))
    except ZeroDivisionError:
        raise Raise

@instruction("b")
def unbox(state):
    x, = get_types(state, Box)
    state.stack.append(x.val)

@instruction("B")
def box(state):
    x, = get_types(state, None)
    state.stack.append(Box(x))

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
            raise Raise
    state.stack.append(x)

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
        with state.rewind() as r:
            state.stack.append(x)
            state.execute(block)
        if not r.rewound:
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
        raise Raise
    for _ in range(int(n)):
        state.execute(block)

@instruction("&")
def tie(state):
    y, x = state.pop_group(), state.pop_group()
    state.push_group(x + y)

@instruction("d")
def dup(state):
    state.push_group([copy.copy(x) for x in state.peek_group()])

@instruction("p")
def pop(state):
    state.pop_group()

@instruction("_")
def under(state, rest):
    x = state.pop_group()
    state.execute(rest)
    state.push_group(x)

@instruction("f")
def flip(state):
    y, x = state.pop_group(), state.pop_group()
    state.push_group(y)
    state.push_group(x)

@instruction("~")
def permute(state, rest):
    state.groups = []
    for perm in itertools.permutations(state.stack):
        state.stack[:] = perm
        if not state.try_execute(rest):
            return
    raise Raise

@instruction("r")
def raise_(state):
    raise Raise

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
    except (Raise, Mask) as e:
        catch_raise(e)
    else:
        raise Raise

@instruction("m")
def mask(state, rest):
    state.mask(rest)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        s = f.read()
    print(*[render(x) for x in run_program(s)])
