## Data types

This is a list of all of the base types for values in x7.

### Instruction index
| Symbol | Name | Brief |
:-: | - | -
`N` | negate | Negate a number.
`D` | divide | Divide two rationals.
`[` | list   | Push an empty list.
`.` | concat | Form two values into a list, add an element to a list from either end, or concatenate two lists.
`]` | enlist | Enclose a value in a list.
`,` | pair   | Form a pair from two values.

### Numbers
All numbers are represented as fractions. They can be constructed with a combination of decimal literals and the instructions `D` (divide) and `N` (negate). They are displayed in a variety of forms by the interpreter:
```x7
1
> 1

1N
> -1

1 2D
> 0.5

1 3D
> 1.(3)

1 95D
> 0.0(105263157894736842)

52 58D
> 26/29

102 58D
> 1+22/29

102 58DN
> -1-22/29
```

### Lists
Finite sequences of values. Built with `[` (list), `.` (concat), and `]` (enlist).
```x7
1 2.3.
> [1,2,3]
```

Although x7 is dynamically-typed, lists are homogenous. Two values can only be in a list together if they are *compatible*, meaning that one of the following holds:
- They are both rationals
- They are both pairs, and their respective elements are compatible
- They are both lists, and one or both of them is empty
- They are both non-empty lists, and the elements inside them are all compatible

### Pairs
Two values. Together. Constructued with `,` (pair).
```x7
1 2,
> (1,2)
```
