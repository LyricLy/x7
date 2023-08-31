## Raises
x7's namesake and one of its most standout features. Here's how to use them.

### Instruction index
| Symbol | Name | Brief |
:-: | - | -
`r` | raise            | Raise.
`e` | except           | Try to run a given block, falling back to a different one if it raises.
`s` | suppress         | Ignore raises in a certain block.
`m` | mask             | Mask raises in a block, protecting them from one layer of being caught.
`!` | invert           | Logical NOT for raising.
`<` | less than        | Compare two values, expecting `first < second`.
`G` | not less than    | Compare two values, expecting `first >= second`.
`=` | equal to         | Compare two values, expecting `first == second`.
`/` | not equal to     | Compare two values, expecting `first != second`.
`>` | greater than     | Compare two values, expecting `first > second`.
`L` | not greater than | Compare two values, expecting `first <= second`.

### Intro
Raises happen all the time in x7 for all sorts of reasons. They can be invoked explicitly, but they often happen as an error case of another operation. Here is a non-exhaustive list of ways raises can occur:
- Type errors, like trying to add a pair to a number
- Stack underflow as a result of trying to access more values from it than it has
- Division by zero
- Reading an undefined variable
- Explicitly raising with `r`
- Failed comparisons
