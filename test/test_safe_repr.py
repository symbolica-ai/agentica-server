# fmt: off

from sandbox.guest.safe_repr import safe_repr


def repr2(obj):
    return safe_repr(obj, False, 2)

def repr5(obj):
    return safe_repr(obj, False, 5)

def repr10(obj):
    return safe_repr(obj, False, 10)

def repr20(obj):
    return safe_repr(obj, False, 20)

def repr30(obj):
    return safe_repr(obj, False, 30)

def repr40(obj):
    return safe_repr(obj, False, 40)


def test_strlike():
    a = 'foo'
    assert repr20(a) == "'foo'"
    assert repr5(a) == "'foo'"
    assert repr2(a) == "'foo'"
    a = b'foo'
    assert repr20(a) == "b'foo'"
    assert repr5(a) == "b'foo'"
    assert repr2(a) == "b'foo'"


def test_long_strlike():
    a = '*' * 100
    assert repr20(a) == "'*********..*********' <100 chars>"
    assert repr5(a) == "<str len=100>"
    assert repr2(a) == "<str>"
    a = b'*' * 100
    assert repr20(a) == "<bytes len=100>"
    assert repr5(a) == "<bytes len=100>"
    assert repr2(a) == "<bytes>"


def test_repr_tuple():
    a = tuple(range(20))
    assert repr20(a) == '(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ..) <20 items>'
    assert repr5(a) == '(0, 1, 2, ..) <20 items>'
    assert repr2(a) == '<tuple len=20>'


def test_repr_single():
    assert repr5(['hello']) == "['hello']"
    assert repr5(['hello', 0]) == "[<str>, ..] <2 items>"


def test_repr_small():
    for fn in (repr2, repr5, repr20):
        assert fn(set()) == 'set()'
        assert fn('') == "''"
        assert fn(b'') == "b''"
        assert fn({}) == '{}'
        assert fn([]) == '[]'
        assert fn(()) == 'tuple()'
        assert fn((1,)) == '(1,)'

    assert repr20((1, 2)) == '(1, 2)'
    assert repr5((1, 2)) == '(1, 2)'
    assert repr2((1, 2)) == '<tuple len=2>'


def test_repr_list():
    a = list(range(20))
    assert repr20(a) == '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ..] <20 items>'
    assert repr5(a) == '[0, 1, 2, ..] <20 items>'
    assert repr2(a) == '<list len=20>'


def test_repr_set():
    a = set(range(20))
    assert repr20(a) == '{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ..} <20 items>'
    assert repr5(a) == '{0, 1, 2, ..} <20 items>'
    assert repr2(a) == '<set len=20>'


def test_repr_dict():
    a = dict(zip(range(20), range(20)))
    assert repr20(a) == '{0: 0, 1: 1, 2: 2, ..} <20 items>'
    assert repr5(a) == '{0: 0, ..} <20 items>'
    assert repr2(a) == '<dict len=20>'
    a = dict(foo=1, bar=2, baz=3, bam='hello world')
    assert repr20(a) == "dict(foo=1, bar=2, baz=3, bam=..)"
    assert repr5(a) == "dict(foo=1, bar=2, ..) <4 items>"
    assert repr2(a) == "<dict len=4>"
    assert repr20({'923': 1}) == "{'923': 1}"


class MyObject: ...

def test_repr_unknown():
    my_object = MyObject()
    a = [my_object] * 10
    assert repr40(a) == "[<'MyObject' object>, <'MyObject' object>, ..] <10 items>"
    assert repr5(a) == "[<'MyObject' object>, ..] <10 items>"
    assert repr2(a) == "<list len=10>"


class FakeVirtObject:

    def __init__(self):
        from agentica_internal.warpc.attrs import VHDL
        from agentica_internal.warpc.resource.handle import ResourceHandle
        setattr(self, VHDL, ResourceHandle())

    def __repr__(self):
        return '~virtual repr~'


def test_repr_virtual():
    virt_obj = FakeVirtObject()
    assert safe_repr(virt_obj, False, 20) == "<'FakeVirtObject' object>"
    assert safe_repr(virt_obj, True, 20) == "~virtual repr~"


def test_dataclass():
    from dataclasses import dataclass
    @dataclass
    class Foo:
        x: int
        y: str
    a = Foo(5, 'x'*15)
    assert repr30(a) == "Foo(x=5, y='xxxxxxxxxxxxxxx')"
    assert repr20(a) == "Foo(x=5, y=<str len=15>)"
    assert repr10(a) == "Foo(x=5, y=..)"
    assert repr5(a) == "<'Foo' object>"
    class Bar:
        __slots__ = ('a','b','c','d','e','f')
    b = Bar()
    b.a = 1; b.b = [2, 3, 4]; b.c = 5; b.d = 'xyz'; b.e = True; b.f = False
    assert repr30(b) == "Bar(a=1, b=[2, 3, 4], c=5, d='xyz', e=.., ..)"
    assert repr20(b) == "Bar(a=1, b=[2, 3, 4], c=5, d=.., ..)"
    assert repr10(b) == "Bar(a=1, b=.., c=5, ..)"
    assert repr5(b) == "<'Bar' object>"
