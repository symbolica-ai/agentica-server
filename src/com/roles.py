from abc import ABC
from typing import Any, ClassVar, Literal

__all__ = [
    'GenRole',
    'UserRole',
    'AgentRole',
    'SystemRole',
    'RoleName',
    'RoleType',
]


type RoleName = Literal['user', 'agent', 'system']
type RoleType = RoleName | UserRole | AgentRole | SystemRole


class GenRole(ABC):
    name: ClassVar[RoleName]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, GenRole) and self.name == other.name

    @staticmethod
    def from_name(name: RoleType) -> 'GenRole':
        if isinstance(name, GenRole):
            return name
        match name:
            case 'user':
                return UserRole()
            case 'agent':
                return AgentRole()
            case 'system':
                return SystemRole()
        raise ValueError(name)

    def __str__(self) -> str:
        return f"<|{self.name}|>"


class UserRole(GenRole):
    name = 'user'
    username: str | None

    def __init__(self, username: str | None = None):
        self.username = username

    def __hash__(self) -> int:
        return hash((self.name, self.username))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, UserRole)
            and self.name == other.name
            and self.username == other.username
        )

    def __str__(self) -> str:
        return f"<|{self.name}: {self.username}|>"


class AgentRole(GenRole):
    name = 'agent'


class SystemRole(GenRole):
    name = 'system'
