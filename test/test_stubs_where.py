"""Test the 'where' clause functionality in stubs.py for object instances."""

from sandbox.guest import stubs


def test_object_instance_with_where_clause():
    """Test that object instances show type annotation + where clause with class definition."""

    class MyClass:
        """A test class with some fields."""

        x: int
        y: str

        def __init__(self, x: int, y: str):
            self.x = x
            self.y = y

        def do_something(self) -> None:
            """Do something."""
            pass

    obj = MyClass(42, "hello")
    result = stubs.format_definition(obj)

    # print("", "#" * 20, sep="\n")
    # print(result)
    # print("#" * 20)

    # The result should have the object annotation, then "where", then the class definition
    assert "obj: MyClass =" in result or "_: MyClass" in result
    assert "where" in result
    assert "class MyClass:" in result
    assert 'x: int' in result
    assert 'y: str' in result
    assert "def __init__(self, x: int, y: str): ..." in result
    assert "def do_something(self) -> None:" in result

    # Verify structure: annotation, blank line, "where", blank line, class
    lines = result.strip().split('\n')
    assert 'where' in lines


def test_multiple_objects_with_where_clause():
    """Test multiple object instances create a single where clause with all class definitions."""

    class MyClass:
        """A test class."""

        x: int

        def __init__(self, x: int):
            self.x = x

    class AnotherClass:
        """Another test class."""

        def method(self, arg: int) -> str:
            """A method."""
            return str(arg)

    obj1 = MyClass(42)
    obj2 = AnotherClass()

    result, _ = stubs.emit_stubs({'obj1': obj1, 'obj2': obj2}, exclude_private=False)

    # print("", "#" * 20, sep="\n")
    # print(result)
    # print("#" * 20)

    # Both object annotations should be present
    assert "obj1: MyClass" in result
    assert "obj2: AnotherClass" in result

    # Single "where" clause (check for the actual keyword, not substring matches)
    assert result.count("\nwhere\n") == 1

    # Both class definitions should be in the where clause
    assert "class MyClass:" in result
    assert "class AnotherClass:" in result

    # Verify the classes are after "where"
    where_index = result.index("\nwhere\n")
    myclass_index = result.index("class MyClass:")
    anotherclass_index = result.index("class AnotherClass:")
    assert myclass_index > where_index
    assert anotherclass_index > where_index


def test_no_where_clause_for_non_objects():
    """Test that functions and classes don't trigger a where clause."""

    def my_function(x: int) -> str:
        """A function."""
        return str(x)

    class MyClass:
        """A class."""

        pass

    result, _ = stubs.emit_stubs(
        {'my_function': my_function, 'MyClass': MyClass}, exclude_private=False
    )

    # print("", "#" * 20, sep="\n")
    # print(result)
    # print("#" * 20)

    # Should not have a where clause since we're not showing object instances
    assert "where" not in result
    assert "def my_function(x: int) -> str:" in result
    assert "class MyClass:" in result


def test_builtin_objects_no_where_clause():
    """Test that built-in type instances don't trigger a where clause."""
    result, context = stubs.emit_stubs(
        {'x': 42, 'y': "hello", 'z': [1, 2, 3]}, exclude_private=False
    )

    # Built-in types should not create a where clause
    assert "where" not in result
    assert "x: int = 42" in result
    assert 'y: str' in result
    assert 'z: list' in result


def test_show_definition_with_where_clause(capsys):
    """Test that show_definition prints the correct output with where clause."""

    class Vehicle:
        """A vehicle with make and model."""

        make: str
        model: str

        def __init__(self, make: str, model: str):
            self.make = make
            self.model = model

        def description(self) -> str:
            """Return the vehicle description."""
            return f"{self.make} {self.model}"

    car = Vehicle("Toyota", "Camry")

    # show_definition prints to stdout
    stubs.show_definition(car)
    captured = capsys.readouterr()

    # print("", "#" * 20, sep="\n")
    # print(captured.out)
    # print("#" * 20)

    # Verify the printed output has the where clause structure
    assert "_: Vehicle" in captured.out or "car: Vehicle" in captured.out
    assert "where" in captured.out
    assert "class Vehicle:" in captured.out
    assert "make: str" in captured.out
    assert "model: str" in captured.out
    assert "def __init__(self, make: str, model: str): ..." in captured.out
    assert "def description(self) -> str:" in captured.out


def test_function_type_annotations_no_where_clause():
    """Test that function type annotations don't trigger a where clause."""

    class Person:
        """A person with a name."""

        name: str

        def __init__(self, name: str):
            self.name = name

    def greet(person: Person) -> str:
        """Greet a person."""
        return f"Hello, {person.name}!"

    result, _ = stubs.emit_stubs({'greet': greet}, exclude_private=False)

    # Function with custom type annotation should NOT trigger where clause
    assert "where" not in result
    assert "def greet(person: Person) -> str:" in result
    # The Person class should NOT be included in output
    assert "class Person:" not in result


def test_mixed_function_and_object():
    """Test that where clause only appears for objects, not function type annotations."""

    class Point:
        """A 2D point."""

        x: int
        y: int

        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

    def distance(p: Point) -> float:
        """Calculate distance from origin."""
        return (p.x**2 + p.y**2) ** 0.5

    point = Point(3, 4)

    result, _ = stubs.emit_stubs({'distance': distance, 'point': point}, exclude_private=False)

    # Should have where clause because of the object instance
    assert "where" in result
    assert "point: Point" in result
    assert "def distance(p: Point) -> float:" in result
    assert "class Point:" in result

    # Verify structure: definitions first, then where clause
    where_index = result.index("where")
    class_index = result.index("class Point:")
    # Class definition should be after "where"
    assert class_index > where_index
