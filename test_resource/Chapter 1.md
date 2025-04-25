Basic stuff about Python.
*Basic understanding of the content in this chapter is suffice enough to handle my future learning*
(*The Todos in this chapter are not suited for learning specifically, it should be learnt one at a time over the course of coding*)

## Todo list

- [x] What are the built-types in Python ✅ 2024-11-18
- [ ] What are common errors in Python
- [ ] What's `ctypes`
- [ ] Methods for `string`; `dictionary` and other built in types. 
- [ ] What is `type` structure in Python
- [ ] inherit methods in Python
- [ ] Metaprogramming
- [ ] class structure in Python
- [ ] How to overload in Python?
	Materials can be found in *Python Cookbook* at page 388
	Implement an error-free `multidispatch`
- [ ] Python naming guidelines
- [ ] The usage of `typing` module
- [ ] What is `mro`
- [x] What's duck method and `ABC` ✅ 2024-11-28
- [x] What is `Generic` ✅ 2024-12-01
- [ ] Exercises on Backtracking

## Python Basics

### Keywords

| Keyword | Keyword  | Keyword  | Keyword |
| ------- | -------- | -------- | ------- |
| False   | None     | True     | and     |
| as      | assert   | async    | await   |
| break   | class    | continue | def     |
| del     | elif     | else     | except  |
| finally | for      | from     | global  |
| if      | import   | in       | is      |
| lambda  | nonlocal | not      | or      |
| pass    | raise    | return   | try     |
| while   | with     | yield    |         |
The only two keywords that I don't normally use is `async` and `await`, which is tightly related to basic concepts like **Thread**. The following shows an example:

```python
import asyncio
import time


async def sleep():
    print(f"Time: {time.time() - start:.2f}")
    await asyncio.sleep(1)


async def sum(name, numbers):
    total = 0
    for number in numbers:
        print(f"Task {name}: Computing {total}+{number}")
        await sleep()
        total += number
    print(f"Task {name}: Sum = {total}\n")


start = time.time()

loop = asyncio.get_event_loop()
tasks = [
    loop.create_task(sum("A", [1, 2])),
    loop.create_task(sum("B", [1, 2, 3])),
]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()

end = time.time()
print(f"Time: {end-start:.2f} sec")
```

### Data Structures

### Composit Data Structures

`set`; `list`; `tuple`; `frozenset`; `dict`; `slice`
- `frozenset`: Typically used when you need an immutable set of unique elements.
- `bytes` :an immutable sequence of bytes. It is used to handle binary data, such as files, network communications, or any other data that is not text. Supports indexing, slicing, iteration `find()`, `split()`, `join()`, and more. Use `decode` and `encode` to convert between `str` and `bytes`
- `bytearray` Mutable version of `bytes`
+ `memoryview`: Allows you to access the memory of a data type without copying data. Supports `slice` `getitem` and so on.
There's no built-in data structure that's both mutable and length-fixed. `Array` from module `array` only supports limited data types.

### Name Management in Python

#### Methods

**Single Underscore (`_method`)**:
A single underscore before a method or attribute name is a convention to indicate that it is intended for internal use. It is a hint to the programmer that the method or attribute is private and should not be accessed directly from outside the class. However, it is not enforced by Python, and the method or attribute can still be accessed if needed. 

**Double Underscore (`__method`)**:
A double underscore before a method or attribute name triggers name mangling. This means that the name of the method or attribute is changed internally to include the class name, making it harder to accidentally access or override it from outside the class. This is used to avoid name conflicts in subclasses.

### `@property`

**When to use `@property`**:
- When you need to add validation logic for setting an attribute.
- When you want to compute the value of an attribute dynamically. (Lazy evaluation and Computed Properties)
- When you want to control access to the attribute (e.g., making it read-only).

**When to use direct attribute access**:
- When the attribute is simple and does not require any special processing.
- When performance is a critical concern and you want to avoid the overhead of property methods.

**Syntax**: `var = property(fget, fset, fdel, doc)`
`@property` is a property which looks like a method. It is based on existing struct of the class and does not define a real attribute.

> [!warning] MyPy Issue
> Normal way to modify inherited property is as follows:
> ```python
> class Base:
> 	@property
> 	def attr(self):...
> class Sub(Base):
> 	@Base.attr.setter
> 	def attr(self, data): ...
> ```
> Due to some error in `MyPy`, the workaround is: 
> ```python
> class Base:
> 	@property
> 	def attr(self):...
> class Sub(Base):
> 	@property
> 	def attr(self):
> 		return super().attr()
> 	@attr.setter
> 	def attr(self, data): ...
> ```

`def attr()`中会定义属性初始值

## Advanced Grammar

### Decorator

Functions in Python are first class items.
- A function is an instance of the Object type.
- You can store the function in a variable.
- You can pass the function as a parameter to another function.
- You can return the function from a function.
- You can store them in data structures such as hash tables, lists, …
Decorators are actually syntax sugar for:
`f = decorator(f)`

- A *wrapper* is the inner function defined within a decorator that actually performs the added functionality.
- A *decorator* is the outer function that takes a function as an argument, defines a wrapper function to modify it, and returns the wrapper.

Use `functool.wraps` to preserve the function signature
Decorators can be used in various ways such as:
+ Decorators with parameters
+ Stacked decorators
+ Class decorators
## Type Hints

### Basic Grammar

| Syntax           | Usage                                           |
| ---------------- | ----------------------------------------------- |
| `Type`           | The class itself rather the object of the class |
| `TypeVar`        | A type variable                                 |
| `Optional[type]` | A type or None                                  |

`TypeVar`: `TypeVar` is a variable that can represent any types.
+ **Bounded TypeVar**: A typevar represents tupe that meets certain criteria, typically by inheriting from a specific class or implementing a specific protocol.
+ **Constrained TypeVar**: A typevar that's constrained to limited sets of types. Notice that different from `|`, a typevar can only represent one type at once.
+ **Covariant (`covariant=True`)**: Allows a type to be replaced with its subtype. Used for return types.
- **Contravariant (`contravariant=True`)**: Allows a type to be replaced with its supertype. Used for input parameters.
`Sequence`:`Sequence`is a type like list but supports covariant typing. However, it does not supports index 
`tuple`: Syntax:
```python
a: Tuple[int, ...]
a: Tuple[int, bool, str]
```

### Generics
`Generic`: `Generic` is a class to be inherited by other class that can work with multiple data types.
`Generic` restricts the object of the class to deal with only one explicit kind of data at once. 
```python
T = TypeVar('T')
class Stack(Generic[T]): ...
```
Generic class can be inherited, and subclasses are also generic classes.
However, subscripted generic class cannot be used in type checking or `match…case…` statements.
When defining generic class, write subscript when you know the type and when you don't subscript it automatically falls into the right place

Use `get_origin` to get original class of a subscripted class.
### Self-reference Classes

#### String Literals

By quoting types, types are stored as string literals and will be evaluated unless necessary. 
#### `from __future__ import annotations`

The statement `from __future__ import annotations` is used to enable postponed evaluation of type annotations. This means that the type annotations in your code will be stored as string literals and not evaluated until necessary. This can help avoid issues with forward references and circular imports.

### Protocol

One can find normal protocols in `collections.abc`, or one can define their own protocol with `Protocol` from `typing` module.
Protocol works with self-reference classes
**Syntax**
```python
from typing import Protocol, TypeVar, ClassVar
T = TypeVar("T")
# Generatic Protocol
class A(Protocol[T]):
	# All kinds of protocol members
	# class attributes are defined with `ClassVar`
	class_attribute: ClassVar[int]
	# instance attributes must be declared at the class level
	instance_attribute: str = ""
    @classmethod
    def class_method(cls) -> str:...
    @staticmethod
    def static_method(arg: int) -> str:...
    # This includes setters and getters
    @property
    def property_name(self) -> str:...
    # Even abstract methods
    @abstractmethod
    def abstract_method(self) -> str:...
# Subclass Protocol
class B(A, Protocol): ...
```

| Class                                                                                                       | Methods                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`Container`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Container)             | `.__contains__()`                                                                                                                                                                                     |
| [`Hashable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)               | `.__hash__()`                                                                                                                                                                                         |
| [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)               | `.__iter__()`                                                                                                                                                                                         |
| [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)               | `.__next__()` and `.__iter__()`                                                                                                                                                                       |
| [`Reversible`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Reversible)           | `.__reversed__()`                                                                                                                                                                                     |
| [`Generator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)             | `.send()`, `.throw()`, `.close()`, `.__iter__()`, and `.__next__()`                                                                                                                                   |
| [`Sized`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sized)                     | `.__len__()`                                                                                                                                                                                          |
| [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)               | `.__call__()`                                                                                                                                                                                         |
| [`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)           | `.__contains__()`, `.__iter__()`, and `.__len__()`                                                                                                                                                    |
| [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)               | `.__getitem__()`, `.__len__()`, `.__contains__()`, `.__iter__()`, `.__reversed__(), `.index()`, and `.count()`                                                                                        |
| [`MutableSequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSequence) | `.__getitem__()`, `.__setitem__()`, `.__delitem__()`, `.__len__()`, `.insert()`, `.append()`, `.clear()`, `.reverse()`, `.extend()`, `.pop()`, `.remove()`, and `.__iadd__()`                         |
| [`ByteString`](https://docs.python.org/3/library/collections.abc.html#collections.abc.ByteString)           | `.__getitem__()` and `.__len__()`                                                                                                                                                                     |
| [`Set`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Set)                         | `.__contains__()`, `.__iter__()`, `.__len__()`, `.__le__()`, `.__lt__()`, `.__eq__()`, `.__ne__()`, `.__gt__()`, `.__ge__()`, `.__and__()`, `.__or__()`, `.__sub__()`, `.__xor__()`,  `.isdisjoint()` |
| [`MutableSet`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSet)           | `.__contains__()`, `.__iter__()`, `.__len__()`, `.add()`, `.discard()`, `.clear()`, `.pop()`, `.remove()`, `.__ior__()`, `.__iand__()`, `.__ixor__()`, and `.__isub__()`                              |
| [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)                 | `.__getitem__()`, `.__iter__()`, `.__len__()`, `.__contains__()`, `.keys()`, `.items()`, `.values()`, `.get()`, `.__eq__()`, and `.__ne__()`                                                          |
| [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)   | `.__getitem__()`, `.__setitem__()`, `.__delitem__()`, `.__iter__()`, `.__len__()`, `.pop()`, `.popitem()`, `.clear()`, `.update()`, and `.setdefault()`                                               |
| [`AsyncIterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.AsyncIterable)     | `.__aiter__()`                                                                                                                                                                                        |
| [`AsyncIterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.AsyncIterator)     | `.__anext__()` and `.__aiter__()`                                                                                                                                                                     |
| [`AsyncGenerator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.AsyncGenerator)   | `.asend()`, `.athrow()`, `.aclose()`, `.__aiter__()`, and `.__anext__()`                                                                                                                              |
| [`Buffer`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Buffer)                   | `.__buffer__()`                                                                                                                                                                                       |

+ ABCs are suitable when you have control over the class hierarchy and want to define a consistent interface across subclasses. 
+ Protocols are useful in scenarios where modifying class hierarchies is impractical or when there’s no clear inheritance relationship between classes.
+ Without formal inheritance relationship, objects may accidentally pass the type checker with the right methods and attributes. 
+ Without formal inheritance relationship, `isinstance` cannot detect whether an object satisfy a protocol or not. Add `@runtime_checkable` to fix.
+ When creating classes on the fly, type annotation based on inheritance would break. Structual type hinding would be particularly useful.
+ Code reuse in Protocol class can be difficult. Use mixin class, utility function and modular design to overcome this issue. 
+ Protocol class can only be used standalone and cannot be used along with `abc`. The relationship of these two requires further investigation.
+ However, protocol class can be directly inherited and achieve code reuse in some way.

### ABC (Abstract Base Classes)

Using `@abstractmethod` and `ABC` one can create abstract base classes.
+ All subclasses must implement all abstract methods
+ Abstract base class cannot be instantiated.
+ Concrete methods inside abstract classes are allowed

**Advantage**: Explict interface. Eased maintainence cost. 
**Disadvantage**: Coupled class implementation. 

The use of `ABC` is tightly knotted with `isinstance()`

+ Overload implemented in abstract class can not be recognized

There are times when defining abstract methods whose arguments or output type varies based on the subclass type. In this situation (Expecially when abc is a generic class), following solution are available:
+ Using `typing.Self` to refer to the class type. 
	Notice `Self` has identical type as self so this may be not available when generic types are needed
+ Using self-bounding TypeVar. Declar a TypeVar with bound, And use this for type annotation in abstract class.
	Notice TypeVars are unsubscriptable and this weakens the strct type checking of original abc.
	Type annotations for `self` are sometimes needed.
## Programming Paradigms
### Metaprogramming

> [!Quote] Tim Peters
> *Metaclasses are deeper magic that 99% of users should never worry about. If you wonder whether you need them, you don’t.*

#### Concept
All classes in Python are instances of class `type`, with their super class being `object`, while all objects are instances of their individual classes.

The usage of `type` are demostrated below:
```python
type(a_object) -> type #Type of the object
type(name, base_classes:tuple, methods_and_attributes:dict()) -> type
```
One can use this to create classes on the fly.
Class dynamically created would not conflict with names in the global scope, because it cannot be called using the class name. 
When using `type` for class creation, one should first design interfaces, and add type annotation by annotate `self` parameter. 

#### Example
Metaclass can actually be anything callable with the same parameter as `type`

**Syntax**
```python
# To create metaclasses:
class my_metaclass(type): ...
# To create class with certain metaclass:
class my_class(object, metaclass=my_meta_class): ...
# which is equvailent to:
class my_class(metaclass=my_meta_class): ...
# Add attributes
class Foo(object, metaclass=something, kwarg1=value1, kwarg2=value2):
    ...
# Use metaclass at module level:
__metaclass__ = something_callable
```

#### Usage
By modifying the metaclass, typically `__new__`  and `__init__` methods, one could change the behavior of the class and all subclasses.
You could intercept code, modify them and return a new class
- metaclasses propagate down the inheritance hierarchies. It will affect all the subclasses as well. If we have such a situation, then we should use metaclasses.
- If we want to change class automatically, when it is created, we use metaclasses
- For API development, we might use metaclasses
- Normally you wouldn't be using metaclasses, you could use monkey patching or decorators to change the behavior of a class

### Inherit

**Basic Traits in OOP (Object Oriented Programming)**: Inheritance, Encapsulation, Polymorphism.

#### Basic Principles in Design Patterns
+ **Single Duty Principle**
	A Class is only for one duty. Comply with different responsibilities and package different responsibilities into different classes or modules
	+ Improved readability and maintainability
	+ Decoupled code
+ **Interface Isolation Principle**
	The client should not rely on interfaces it does not need; the dependency of one class on another should be based on the smallest interface.
	Create a single interface, do not build a large and bloated interface, try to refine the interface, the method in the interface is as small as possible. That is, we need to create a dedicated interface for each class, rather than trying to build a very large interface for all classes that depend on it.
	(**Single Duty Principle** and **Interface Isolation Principle** are similar)
+ **Liskov Substitution Principle**
	Instances of subclasses must be able to be treated as a instance of the parent class without error. 
	- Subclasses can add their own unique methods.
	- When used for polymorphism, subclasses can implement abstract methods of the parent class.
	- When used for code reuse, or subclasses are inherited by concrete classes, they cannot override the non-abstract methods of the parent class, if they do, must follow criterias below.
	- When a subclass’s method overrides the parent’s method, the method’s precondition (that is, the method’s formal parameter) is more lenient than the parent class’s input parameter.
	- When a subclass’s method implements the abstract method of the parent class, the postcondition of the method (that is, the return value of the method) is stricter than the parent class.
	- Subclass can only throw error types in the scope of error types of the parent class.
+ **Law of Demeter**
	The less a class uses interfaces from another class, the better.
	The aim of this rule is to decrease the level of coupling between classes. Use the smallest amount of interfaces or encapsule interfaces together. But we should not abuse the rule by creating massive intermediate interfaces, increasing overall system complexity. Therefore, when using the Dimitte rule, we must repeatedly weigh the balance, so that the structure is clear, but also high cohesion and low coupling.
+ **Dependency Inversion Principle**
	High-level modules should not rely on low-level modules, both of which should rely on their abstractions; abstractions should not rely on details; details should rely on abstractions.
	- Low-level modules should have abstract classes or interfaces, or both.
	- The declared type of a variable is as much as possible an abstract class or interface.
+ **Open-Closed Principle**
	+ A software entity such as classes, modules, and functions should be open to extensions and closed to modifications.
	+ When software needs to change, try to implement changes by extending the behavior of the software entities, rather than modifying the existing code to implement the changes.

Objects of a subclass are also instances of the parent class
#### Type Hints
It uses a *nominal* type checking system. See at [[课外探索/8. Python学习笔记/Python数据结构与算法分析/Chapter 1#ABC (Abstract Base Classes)]]


### Duck Typing

There are several flaws in OOP. One of the most eminent is the coupled interface.
Duck typing avoid such flaws. 
The use of duct typing is tightly knotted with `hasattr`

#### Type Hints

It uses a *structual* type checking system. See at [[课外探索/8. Python学习笔记/Python数据结构与算法分析/Chapter 1#Protocol]]

## Miscs

+ `locals()`: returns a dictionary of current local variables
+ `all(iter)`: checks if all objects in `iter` is `True`
+ `any(iter)`: checks if any objects in `iter` is `True` 
+ `match…case…`: sometimes it can be used as a alternative for `if…else…` statement, but it is especially useful for type match
```python
match var:
	case 1: ...
	# Notice: This grammar cannnot be used with the syntax below
	case 1|2: ...
	case [*a, 1]:...
	case list() if all(isinstance(x, Numeric) for x in obj): ...
	case {**a, "name"="me"}: ...
	case A_dataclass("name"=1):...
```
+ Arguments of a function cannot be paratheslized
+ `Callable[[Arg1type, Arg2type, …], output]`
	+ `Iterable/Iterator[ItemType]`
	+ `Generator[YieldType, SendType, ReturnType]`
		+ `YieldType` is the type of values yielded by the generator.
		- `SendType` is the type of values that can be sent to the generator (using the `send` method).
		- `ReturnType` is the type of the value returned when the generator terminates.
+ Memory process for objects in python: Python hold account for reference count. When a *temporary object* with no *reference count*, Python's garbage collector deallocates the memory used by these objects.
+ `Args` in Python is a tuple for positional arguments
	`kwargs` in Python is a dictionary for keyword arguments

```python
match args:
    case ():
        return CustomLogicProcessor()
   # You can't directly state that elem is a list
    case ([*elem],) if all(isinstance(x, bool) for x in elem):
        return func(elem)
```

+ `NamedTuple` in `type`, namedtuple in collections
+ when debugging with vsc, the str property is instantly called, may cause trouble with running and debugging
+ when defining class dynamically, type can create class at file level while nested class 

## Back Tracking

### Usage
- ***Decision Problems***: Here, we search for a feasible solution.
- ***Optimization Problems:*** For this type, we search for the best solution.
- ***Enumeration Problems:*** We find set of all possible feasible solutions to the problems of this type.
Example: N Queen problem, Rat in a Maze problem, Knight’s Tour Problem, Sudoku solver, and Graph coloring problems.

### Difference with Other Algorithm

Back tracking works for all every constraint satisfaction problem.
+ Dynamic Programming: Work only if subproblems or previous states has effect on proceding states.
+ Greedy: Works if there's a bias between different possibilities. 

### Syntax with Backtracking

```python
# Backtracking where the attempted operation is easily reverted
def backtracking(attempt_args):
	# Pre-operation you want to do in each layer of decision
	attempts()
	for decision_lists:
		attempted_operation(attempt_args)
		if result := backtracking(current_args):
			# Revert attempts
			return results
	return None

```

In conclusion, backtracking is a special kind of traverse. 
+ `for` loop to traverse all possibility space
+ `return` statement to shortcut the traverse, which is unnecessary for enumeration Problems.
+ Reverting arguments or encapsuling parameters and continuing with traverse for backtracking.