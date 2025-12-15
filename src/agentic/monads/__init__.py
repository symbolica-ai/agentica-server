__all__ = [
    'AgentMonads',
    'model_router',
]

from dataclasses import dataclass
from types import ModuleType
from typing import Callable

from agentica_internal.session_manager_messages import PromptTemplate

from com.abstract import HistoryMonad

from ..models import ProviderModel

# DISABLED:
# from .json_tool.base import MultiTurnJSON
from .repl_tool.multi_turn import anthropic, openai


@dataclass
class AgentMonads:
    init_monad: Callable[[str | None, str | PromptTemplate | None], HistoryMonad[None]]
    interaction_monad: HistoryMonad[None]
    user_monad: Callable[[str | PromptTemplate, str | PromptTemplate | None], HistoryMonad[None]]

    @classmethod
    def from_multi_turn_repl(cls, multi_turn_repl: ModuleType) -> 'AgentMonads':
        return cls(
            init_monad=multi_turn_repl.system_monad,
            user_monad=multi_turn_repl.user_monad,
            interaction_monad=multi_turn_repl.interaction_monad(),
        )

    @classmethod
    def from_multi_turn_json(cls, multi_turn_json: ModuleType) -> 'AgentMonads':
        return cls(
            init_monad=multi_turn_json.system_monad,
            user_monad=multi_turn_json.user_monad,
            interaction_monad=multi_turn_json.interaction_monad(),
        )


def model_router(model: ProviderModel, json: bool) -> AgentMonads:
    if json:
        raise ValueError("JSON mode is disabled")
        # return model_router_json(model)
    else:
        return model_router_code(model)


# DISABLED:
# def model_router_json(_model: ProviderModel) -> AgentMonads:
#     return AgentMonads.from_multi_turn_json(MultiTurnJSON)


def model_router_code(model: ProviderModel) -> AgentMonads:
    match (model.provider, model.model):
        case ('anthropic', _):
            return AgentMonads.from_multi_turn_repl(anthropic)
        case _:
            return AgentMonads.from_multi_turn_repl(openai)
