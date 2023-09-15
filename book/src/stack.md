## Stack manipulation

This chapter is a primer on how to use the language's facilities to shuffle values around on the stack.

### Instructions used in this chapter
| Symbol | Name | Brief |
:-: | - | -
`d` | dup      | Duplicate the top group on the stack.
`p` | pop      | Discard the top value on the stack.
`f` | flip     | Swap the top two groups on the stack.
`^` | yank     | Duplicate the second group from the top of the stack.
`&` | group    | Concatenate the top two groups on the stack.
`_` | under    | Run some code "under" the group on top of the stack.
`l` | lift     | Run separate blocks on the same arguments and push both the results.

### Basics
If you've used any stack-based language before, you'll be expecting these mainstays.

Duplicate a value (Forth's `DUP`):
```x7
1d
> 1 1
```

Get rid of one (Forth's `DROP`):
```x7
1p
>
```

Flip the top two (Forth's `SWAP`):
```x7
1 2f
> 2 1
```

Duplicate the second from the top (Forth's `OVER`):
```x7
1 2^
> 1 2 1
```

We're missing some common choices from Forth, which we'll address in the next section.

### Temporary groups
It is sometimes convenient to use stack utilities on multiple values at once, treating them like a "group" of consecutive values.
This can be done with `&` (group), which joins values together into larger groups that can be as large as you want.
```x7
1 2& 1 2 3&&
> 1&2 1&2&3
```

When an instruction refers to "groups" instead of "values" in its documentation, it means it will treat these groups specially:
```x7
1 2&d
1&2 1&2
```

Any other instruction will simply dissolve the group entirely when it needs a value from it.
```x7
1 2 3&&4+
> 1 2 7
```

We can use this to replicate some other Forth words, like `ROT`:
```x7
1 2 3&f
> 2&3 1
```

### Dipping
The last 2 tools included for the stack take blocks. The first is `_` (under, often called `dip` in other languages),
which pops a group, executes the block, and pushes the popped group again. This lets you do work lower down on the stack while preserving the top values.
```x7
1 2 3_+
> 3 3
```

Finally, `2SWAP`:
```x7
1 2 3 4&_&`f
> 3&4 1&2
```

### Lifting
`l` (lift) takes *two* blocks, executes the second one, pops a group, [rewinds](raises.md#rewinding) to before it ran, runs the first block, then pushes the group that was popped from the first branch.
It sounds complicated, but all it really does is let you execute multiple functions on the same input and get both results:
```x7
3 3l+}*
> 6 9
```
Here we use it to compute both \\(3 + 3 = 6\\) and \\(3 * 3 = 9\\) without having to duplicate the inputs. This enables idioms reminiscent of a [fork](https://aplwiki.com/wiki/Train#3-trains)
in APL or Haskell's [`liftA2`](https://hackage.haskell.org/package/base-4.18.0.0/docs/Prelude.html#v:liftA2) function used in the `(a ->)` functor, for which the instruction is named.
