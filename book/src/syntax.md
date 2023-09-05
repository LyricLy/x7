## Basic syntax

x7 is a stack-based language. If you've ever written Forth, used the `dc` calculator, or looked at any of the myriad stack-based esolangs before, you should feel right at home with this form of expressions.
There is a single stack of [values](types.md) onto which intermediate results are placed, and instructions pop arguments from it to manipulate before pushing the result. 

### Instruction index
| Symbol | Name | Brief |
:-: | - | -
`+` | add      | Add two numbers.
`*` | multiply | Multiply two numbers.
`T` | times    | Run a block a given number of times.
`e` | except   | Try to run a given block, falling back to a different one if it raises.
`r` | raise    | Raise.

### Intro
Here's a simple example. We compute `2*3+1` in RPN, leaving the number `7` on the stack.
```x7
1 2 3*+
> 7
```
Note the format of this example: the output after running the code is shown below it, with a `>` preceding it.
We will use this convention for all examples in the reference from this point onward. Examples may also be followed by pseudocode showing what it is equivalent to.

We learn some things about the language from this example. Each instruction is a single character long, so we can write `*` and `+` next to each other without ambiguity. However,
spaces are needed between decimal literals, because if they were written next to each other, it would be read as a single number. There is one exception to this: because
leading zeroes before decimals are not allowed, there is no ambiguity after a `0`.
```x7
01 23
> 0 1 23
```

### Variables
`:x` stores a value in a variable called `x`, while `;x` retrieves it. Variable names can be any single character that is not a digit.
```x7
42:x 22:y ;x ;y ;x
> 42 22 42
```
This code would usually be written without spaces, as `42:x22:y;x;y;x`. It has been formatted this way to make it easier to read.

### Functions
By default, only the last line in a x7 file is executed. (It acts as an entry point, like `main`.) Other lines can be executed by using `;` with the line number.
```x7
3 4
1 2;1
> 1 2 3 4
```
You can call any line (including the main line) any number of times, and they may recurse.

### Blocks
Some instructions take *blocks* of source code. In the simplest case, this means reading the code in front of the instruction until reaching a backtick.
As an example, the `T` (times) instruction pops a natural number from the stack and runs a block that many times.

```x7
1 10T2*`
> 1024
```
In Ruby:
```ruby
n = 1
10.times do
    n *= 2
end
```

We push a `1` to the stack, then use `T` to double it 10 times, hence calculating \\(2^{10} = 1024\\).

Uses of `T` can nest as though `T` was an opening bracket and `` ` `` closed it:
```x7
1 4T2T2*``
> 256
```
```ruby
n = 1
4.times do
    2.times do
        n *= 2
    end
end
```

By doubling twice (multiplying by 4), 4 times, we get \\((2^2)^4 = 4^4 = 256\\) in a similar fashion to the previous example.

#### Auto-closing
Blocks close themselves at the end of a line, which is convenient for closing many at once:
```x7
0 10T10T10T1+
> 1000
```
```ruby
n = 0
10.times do
    10.times do
        10.times do
            n += 1
        end
    end
end
```

This power can be harnessed explicitly within a line by writing a closing brace.
```x7
0 10T10T1+}2*
> 200
```
```ruby
n = 0
10.times do
    10.times do
        n += 1
    end
end
n *= 2
```
Closing the `T` blocks early prevents the `2*` from being placed inside them, so the result is `200` instead of `2535301200456458802993406410750`.

`}` itself can also be capped with `{`:
```x7
0 2T{10T10T1+}2*
> 600
```
```ruby
n = 0
2.times do
    10.times do
        10.times do
            n += 1
        end
    end
    n *= 2
end
```
Here we keep the `2*` outside of the two inner `10T` loops, but *inside* the `2T` loop on the outside. `{` prevents `}` from being able to close blocks to the left of the `{`.

#### Multiple blocks
When an instruction takes more than one block, the syntax of all blocks other than the last one is different.

We need an example of an instruction that takes two blocks, so say hello to `r` (raise), which raises, and `e` (except), which acts like a `try/catch`.
```x7
e0}1
> 0
er}1
> 1
```
Here we see two examples where `e` is used. It tries to run the first block, but if and only if it raises, it runs the second block instead.

Note that the two blocks are separated with `}`. This is mandatory for instructions that take multiple blocks. The exact behaviour of this parsing can be seen here:
```x7
2Te2T0}1`2
> 0 0 2 0 0 2
```
```ruby
stack = []
2.times do
    begin
        2.times do
            stack.push(0)
    rescue
        stack.push(1)
    end
    stack.push(2)
end
```

The first `}` closes the second `2T` (because it is inside the `e`) but not the first (because it is not inside the `e`), and is then consumed to begin parsing the second block.
The second block is closed by the `` ` ``, followed (still inside a `2T`) by the `2`. As a result, the whole code executes twice, pushing two `0`s, ignoring the `1` (because there is no raise)
and adding a `2`.
