from dataclasses import dataclass
from typing import cast, get_args

from .version_policy import SDK


@dataclass(kw_only=True)
class MagicProtocol:
    sdk: SDK
    version: str

    @classmethod
    def parse(cls, protocol: str | None) -> 'MagicProtocol':
        if protocol is None:
            return MagicProtocol.default()
        parts = protocol.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid protocol format: {protocol!r} (expected 'sdk/version')")
        sdk, version = parts
        if sdk not in get_args(SDK):
            raise ValueError(f"Invalid SDK: {sdk}")
        return cls(
            sdk=cast(SDK, sdk),
            version=version,
        )

    @classmethod
    def default(cls) -> 'MagicProtocol':
        return cls(
            sdk='python',
            version='0.0.0-dev',
        )
