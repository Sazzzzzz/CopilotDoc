## TODO
- [ ] What does `AST` do?
- [ ] Is there a data structure that supports 3 layer hash table?
- [ ] What is `time.perf_counter`
- [ ] What is GIL
- [ ] async producer-consumer pattern
- [ ] internal event loop mechnism
## Remarks
### Remarks on `stack`:
A stack is a data structure that implements last-in, first-out (LIFO) behavior, making it ideally suited for tasks such as postfix calculation and base conversion.

The essential principles for converting infix expressions to postfix expressions are:
+   Operator precedence exhibits local monotonic increase for consecutive stack entries.
+   While parentheses are theoretically handled via recursive function calls, they adhere to the same local rule and can be effectively processed within the main function without significant impact.

> **Advices from my implementation**
> + Don't use regex to parse expressions
> + `**` is right-assosiative, requires additional handling

## `Collections`

This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers.

| [`namedtuple()`](https://docs.python.org/3/library/collections.html#collections.namedtuple "collections.namedtuple")  | factory function for creating tuple subclasses with named fields                                     |
| --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [`deque`](https://docs.python.org/3/library/collections.html#collections.deque "collections.deque")                   | list-like container with fast appends and pops on either end                                         |
| [`ChainMap`](https://docs.python.org/3/library/collections.html#collections.ChainMap "collections.ChainMap")          | dict-like class for creating a single view of multiple mappings                                      |
| [`Counter`](https://docs.python.org/3/library/collections.html#collections.Counter "collections.Counter")             | dict subclass for counting [hashable](https://docs.python.org/3/glossary.html#term-hashable) objects |
| [`OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict "collections.OrderedDict") | dict subclass that remembers the order entries were added                                            |
| [`defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict "collections.defaultdict") | dict subclass that calls a factory function to supply missing values                                 |
| [`UserDict`](https://docs.python.org/3/library/collections.html#collections.UserDict "collections.UserDict")          | wrapper around dictionary objects for easier dict subclassing                                        |
| [`UserList`](https://docs.python.org/3/library/collections.html#collections.UserList "collections.UserList")          | wrapper around list objects for easier list subclassing                                              |
| [`UserString`](https://docs.python.org/3/library/collections.html#collections.UserString "collections.UserString")    | wrapper around string objects for easier string subclassing                                          |
+ `OrderedDict`
    + `popitem(last=True)`
    + `move_to_end(last=True)`
    The rest of it is the same as normal dictionary

---
This part is dedicated on making code run faster

## What is concurrency?

==**Parallelism**== consists of performing multiple operations at the same time. **Multiprocessing** is a means to effect parallelism, and it entails spreading tasks over a computer’s central processing units (CPUs, or cores). Multiprocessing is well-suited for CPU-bound tasks: tightly bound and mathematical computations usually fall into this category.

==**Concurrency**== is a slightly broader term than parallelism. It suggests that multiple tasks have the ability to run in an overlapping manner. (There’s a saying that concurrency does not imply parallelism.)

==**Threading**== is a concurrent execution model whereby multiple take turns executing tasks. One process can contain multiple threads.

==**Asynchronous Programming**== is a single-threaded, single-process design: it uses **cooperative multitasking** async IO is a style of concurrent programming, but it is not parallelism. It’s more closely aligned with threading than with multiprocessing but is very much distinct from both of these and is a standalone member in concurrency’s bag of tricks.
## Asynchronous Programming in Python

`await`: Special break point

`Coroutine`: A function that can yield control via `await`.It won't run asynchronously without being in an event loop

`Future`: A placeholder for something will be done.

`Tasks`: While `Coroutine` is just a function, `Tasks` is a subclass of `Future` wrapped a function
- Wraps a coroutine and schedules it to run on the event loop
- Represents an execution unit that's being managed by the event loop
- Provides a way to monitor the coroutine's status (pending, done, cancelled)
- Has methods for cancellation and callbacks
### Syntax
- The syntax `async def` introduces **native coroutine**. The expressions `async with` and `async for` are also valid.
- The keyword `await` passes function control back to the event loop. (It suspends the execution of the surrounding coroutine.) 
- A function that you introduce with `async def` is a coroutine. It may use `await`, `return`, or `yield`, but all of these are optional. Declaring `async def noop(): pass` is valid:
    - Using `await` and/or `return` creates a coroutine function. To call a coroutine function, you must `await` it to get its results.
    - It is less common to use `yield` in an `async def` block. This creates an [asynchronous generator](https://www.python.org/dev/peps/pep-0525/), which you iterate over with `async for`. 
    - Anything defined with `async def` may not use `yield from`.
- `async with` `async for` `async while` will not run iterations concurrently. It just enables the async feature.

Most programs will contain small, modular coroutines and one wrapper function that serves to chain each of the smaller coroutines together. [`main()`](https://realpython.com/python-main-function/) is then used to gather tasks (futures) by mapping the central coroutine across some iterable or pool.

### `Asyncio` Methods
This is what's happening in async programs
**Overall Running**
+ `run`: It ==creates an event loop==, executes/await the function and return results.
+ `gather`: It adds passed event to the current event loop,runs them concurrently and waits for it to finish,but does't create its own event loop. If direct corourtines are pass, it automatically convert them to tasks. ==*This is recommended for timing issue with already running tasks*==
**Task Specifics**
+ `create_task`: Create a task that runs immediately, but don't await them to finish
+ `wait_for`: wait for a single future or coroutine to finish with time constrain.
+ `cancel`: cancel tasks
Normally you just `gather` and `run`, further topics include: `lock` `stream`

> *For the first time the runtime gives me warning instead of errors*

### Design Patterns
+ Chaining coroutines
+ Producer-Consumer pattern
### Implementation
#### Why is a coroutine awaitable?
Coroutine is actually a generator. `await` is just like `yield from`, and by the generator's `send` syntax it is able to receive outside infomation

#### Internals of event loop

### Usage Scenerios
+ When you could find proper library
+ Avoid blocking function such as web IO.

## Garbage Collect

There is indeed probability when objects form circular reference and break the reference counter, causing resource waste or memory leakage.
However, Python's module `gc` *(Cyclic Garbage Collector)* identifies groups of objects that reference each other but are not reachable from anywhere else in the program (outside the cycle).
## ANSI Escape Code

ANSI escape sequences were introduced in the 1970s as a standard to style text terminals with color, font styling, and other options.

|    Part     |                               `\x1b[`                               |                                           `31`                                           |                                      `m`                                      |
|:-----------:|:-------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
| Description | Starts sequence, also called the Control Sequence Introducer (CSI). | Color code for various text and background colors, e.g. between `30`-`49` or `90`-`109`. | Ends sequence and calls the graphics function Select Graphic Rendition (SGR). |
I'll just use [`colorist`](https://jakob-bagterp.github.io/colorist-for-python/) for convinence.

## Miscs
+ Use list comprehension and unpacking to pass parameters to complex functions
+ Iterable unpacking cannot be used in comprehension
+ `time.perfcounter` provides Performance counter for benchmarking.
+ `os.urandom` creates random bytes
+ `os.argv` returns a list of args passed to the script
+ `tqdm.tqdm` for progress bar
+ Callback: Just a function passed into another function and gets called.
+ `list.clear()` clears all items in list.