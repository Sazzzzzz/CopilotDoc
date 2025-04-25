## TODO

- [ ] Common games like 24 points, managing water, Eulerian Path
- [ ] How to write switch arguments like `method: Literal["Internal", "External"]`
- [ ] Is there any way to write code snippets without defining a function inside a function?
- [ ] What is the `send` method of `Generator`
- [ ] What is overhead
- [ ] How to type for numbers 
- [ ] Usage of `dataclass`
- [ ] All things from `datetime`
- [ ] All things from `typing`
- [ ] How to write unittest
- [ ] decision tree
- [ ] What is `abcmeta`
- [ ] multithread stack

## Naming Conventions in Python

|       Elements       |             Naming Convention              |
| :------------------: | :----------------------------------------: |
|  Functions/Methods   | `snake_case`, action oriented, descriptive |
|      Variables       |   `snake_case`, descriptive but concise    |
|      Constants       |  `UPPERCASE_WITH_UNDERSCORE`, descriptive  |
|       Classes        |                `PascalCase`                |
| Modules and Packages |                `snake_case`                |
**Avoid Hungarian notation**: Do not include the type of the variable in its name.

## Time Complexity of Some built-in Functions/Methods

### List Generation

`range` < List Comprehension < `append` < `__add__`

### Methods of List

|      Operation      | Time Complexity |
|:-------------------:|:---------------:|
|    `l.append()`     |     $O(1)$      |
|       `l[k]`        |     $O(1)$      |
|    `l[k] = elem`    |     $O(1)$      |
|      `l.pop()`      |     $O(1)$      |
|     `l.pop(k)`      |     $O(n)$      |
| `l.insert(i, elem)` |     $O(n)$      |
|     `del l[k]`      |     $O(n)$      |
|   `for elem in l`   |     $O(n)$      |
|      `x in l`       |     $O(n)$      |
|      `l[a:b]`       |     $O(n)$      |
|   `l[a:b] = sth`    |   $O(n + k)$    |
|      `len(l)`       |     $O(n)$      |
|     `l.sort()`      |  $O(n \log n)$  |
|        `l*k`        |     $O(nk)$     |
|     `l.reverse`     |     $O(n)$      |
|       `l + m`       |     $O(n)$      |
|    `del l[a:b]`     |     $O(n)$      |
### Methods of Dictionaries

|    Operation     | Time Complexity |
| :--------------: | :-------------: |
|     `d[key]`     |     $O(1)$      |
| `d[key] = value` |     $O(1)$      |
|   `del d[key]`   |     $O(1)$      |
|    `key in d`    |     $O(1)$      |
|   `d.get(key)`   |     $O(1)$      |
|   `d.pop(key)`   |     $O(1)$      |
|  `d.popitem()`   |     $O(1)$      |
|     `len(d)`     |     $O(1)$      |
|   `d.clear()`    |     $O(n)$      |
|    `d.copy()`    |     $O(n)$      |
|   `d.values()`   |     $O(n)$      |
|   `d.items()`    |     $O(n)$      |
|    `d.keys()`    |     $O(n)$      |
## Mutable and Immutable Objects

- [ ] list multiplication
- [ ] SupportsIndex

## Multiple Inheritance

In Python, every class whether built-in or user-defined is derived from the object class and all the objects are instances of the class object. In the case of multiple inheritance, a given attribute is first searched in the current class if it’s not found then it’s searched in the parent classes. The parent classes are searched in a left-right fashion and each class is searched once.  
If we see the above example then the order of search for the attributes will be Derived, Base1, Base2, object. The order that is followed is known as a linearization of the class Derived and this order is found out using a set of rules called **Method Resolution Order (MRO).**

`super().method()` refers to the first parent class that implements `method()` in MRO.  
```python
A.mro() # returns a list
A.__mro__ # returns a tuple
```

When `super` function is called, the method called within `super` function follows `mro` order. If method from parent class is needed, call them directly with `SuperClass.method(self, args)`
## Iterator

There are two kinds: `Iterable` and `Iterator`
+ `Iterable`: A object that implements `__iter__`
+ `Iterator`: A object that implements `__iter__` and `__next__`

`__iter__` returns a object that is a iterator, which implements `__next__`. As long as a object defines `__iter__`, it is considered iterable. 
For a object that implements both `__iter__` and `__next__`, typically the `__iter__` returns the object itself, while the `__next__` method will be called during iteration. The object itself is an iterator.
Generators are just a special kind of iterator that use `yield` keywords to iterate. 

`yield from x` is syntax sugar of `for i in x yield x`

In development, `iter()` first before `next()`

### `Itertools`
Infinite iterators:  
count(start=0, step=1) --> start, start+step, start+2*step, …  
cycle(p) --> p0, p1, … plast, p0, p1, …  
repeat(elem [,n]) --> elem, elem, elem, … endlessly or up to n times
*product*: Cartesian product of iterators

## Hash Objects

Only immutable objects with `__hash__` and `__eq__` can be hashed.

`__hash__` returns the value of `hash(obj)`
`__eq__` are used to compare if two objects are identical. This is crucial for resolving hash conflicts. If not implemented, all objects are considered different based on their memory location.

## u-string f-string and r-string

+ u-string for unicode string
+ r-string for raw string, where `\` means nothing more than a backslash(except when before a quote). Useful for file directory, regex where escape sequence are not needed.
+ f-string format string
## `random` and `re`

### `random`
Commonly used functions are:
+ `randint()`
+ `randon()`
+ `sample`
+ `choice`
+ `choices`
To get a random element from a set/dictionary, use `sorted` first to convert them to `lists`

### `re`


## File Operation
| Character | Meaning                                                         |
| --------- | --------------------------------------------------------------- |
| 'r'       | open for reading (default)                                      |
| 'w'       | open for writing, truncating the file first                     |
| 'x'       | create a new file and open it for writing                       |
| 'a'       | open for writing, appending to the end of the file if it exists |
| 'b'       | binary mode                                                     |
| 't'       | text mode (default)                                             |
| '+'       | open a disk file for updating (reading and writing)             |
|           |                                                                 |

## Type Hints

+ `typing.Self` useful for referring to current subclass
+ `typing.reveal_type` to see object type

### Type from existing types
+  Type that are equivalent
```python
type A list[int]
A=list[int]
A:TypeAlias=int
```
+ Type that are subtype: Use `typing.NewType`
    `NewType` can not be subclassed
### `TypedDict`
Annotation for dictionarys, generics supported.
```python
class Point[T](TypedDict, total=True):
    x: NotRequired(int)
Point=TypedDict('Point',{'x':NotRequired(int)})
```
### Import from Built-in objects

`_typeshed` contains Built-in functions, types, exceptions, and other objects. It's not commonly used. 

During runtime, there may be error: `ModuleNotFoundError: No module named '_typeshed'`, following guard could solve the problem:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	import _typeshed
```

### Force arguments with desired type:
There are three approaches: `cast`, `TypeGuard`, and `assert`

| Feature              | cast | assert  | TypeGuard |
| -------------------- | :--: | :-----: | :-------: |
| Type Checker Support |  ✅   |    ✅    |     ✅     |
| Runtime Safety       |  ❌   |    ✅    |     ✅     |
| Reusability          |  ❌   |    ❌    |     ✅     |
| Performance Impact   | None | Minimal |  Minimal  |
| Freedom of Type      |  ✅   |    ❌    |     ✅     |
The syntax for `TypeGuard` is as follows:

```python
def is_str_list(l:list)->TypeGuard[list[str]]:
	return all(isinstance(x, str) for elem in l)
```

### Supports?

Take `SupportsIndex`for example:
`SupportsIndex`: An abstract base class. Used when object implements `__index__` method. `__index__` method returns `int`. Useful for syntax `range(A,B,C)`

Others include:
+ `SupportsAbs`
+ `SupportsBytes`
+ `SupportsComplex`
+ `SupportsFloat`
+ `SupportsInt`
+ `SupportsRound`
- [ ] suppress mypy

### Annotation for `Callable`

There's something called "type parameter list", experimental feature in mypy that frees you from writing `T=TypeVar("T")`

`typing.Callable` is deprecated, replace them with `Collections.abc.Callable`

The subscription syntax must always be used with exactly two values: the argument list and the return type. The argument list must be a list of types, a `ParamSpec`, `Concatenate`, or an ellipsis. The return type must be a single type.

`Concatenate` is currently only valid when used as the first argument to a Callable. The last parameter to `Concatenate` must be a `ParamSpec` or ellipsis (`…`).

`Concatenate` can be used in conjunction with Callable and `ParamSpec` to annotate a higher-order callable which adds, removes, or transforms parameters of another callable.

Parameter specification variables exist primarily for the benefit of static type checkers. They are used to forward the parameter types of one callable to another callable, parameter specifications can be declared with two asterisks (`**`)
### Pre-annotation

In addition, one **cannot annotate variables used in a `for` or `with` statement**; they can be annotated ahead of time, in a similar manner to tuple unpacking *(PEP 526)*

```python
i: int
for i in range(5):
    pass
```

## Overhead

The meaning of the word can differ a lot with context. In general, it's resources (most often memory and CPU time) that are used, which do not contribute directly to the intended result, but are required by the technology or method that is being used. Examples:

- Protocol overhead**: Ethernet frames, IP packets and TCP segments all have headers, TCP connections require handshake packets. Thus, you cannot use the entire bandwidth the hardware is capable of for your actual data. You can reduce the overhead by using larger packet sizes and UDP has a smaller header and no handshake.
- **Data structure memory overhead**: A linked list requires at least one pointer for each element it contains. If the elements are the same size as a pointer, this means a 50% memory overhead, whereas an array can potentially have 0% overhead.
- **Method call overhead**: A well-designed program is broken down into lots of short methods. But each method call requires setting up a stack frame, copying parameters and a return address. This represents CPU overhead compared to a program that does everything in a single monolithic function. Of course, the added maintainability makes it very much worth it, but in some cases, excessive method calls can have a significant performance impact.
## Miscs
+ `collections.Counter` Create a counting object
+ `itertools.permutations` Return the permutation of certain objects
+ `timeit.timeit(expr:str|Callable, setup=expr, number:int)` Measure the time used of running the expr
+ `callable(obj) -> bool`: Check if `obj` is callable
+ Create functions with functions:
	+ Nested `def` creates currying of function
	+ `functool.partial` fix certain parameters and create partial functions
	+ `lambda`
	+ Set default parameters creates functions
+ `slice` creates a slice object with attribute `start`, `stop`, `step` and `indices`, the following codes are equivalent. `indices(len(iter))` calculates `start`, `stop`, and `step` according to the length of `iter`. Algorithrm is shown below:
- [ ] a graph

```python
list[slice(1,3,2)]
list[1:3:2]
```
+ `str.join(Iter)` method can concatenate any number of strings.
+ There's only `deepcopy`, `copy` and `Error` inside `copy` module
	+ `deepcopy` cannot copy generators (which is meant for lazy evaluation)
- `sum` with `start` positional arguments specified could do sequential addition. (`str` not supported)
- `breakpoint()` are directly available in VSCode.
- `os`: `getenv`, `makedir`, 
- When switching modes, use literal strings for readability and dictionary mapping for internal status.
```python
def func(option: Literal["A","B"]):
	l = {
	"A": 1	
	"B": 2
	}.get(option, default)
```

- [ ] eliminate duplicate keys in python
- [ ] join split

### f-string
1. In f-strings, a single `{}` is used for variable interpolation
2. When you need to output a literal curly brace `{` or `}` in an f-string, you need to double them

### Comment Syntax

```python
# type: ignore: ignore all mistake for current line
# type: ignore[misc]: ignore specific mistake for current line
# type: int
# cSpell: words sth1, sth2
# cSpell: ignore sth1, sth2
# cSpell: diable/enable
# cSpell: disable-line/disable-next-line
```

### `for … else…`

`for … else …` is absolutely legal python code. It's normal applications encompass:**
+ Loop completion validation
+ Exception handle when loop completed undesirably
+ Avoid flagship variables that would require further tracking.