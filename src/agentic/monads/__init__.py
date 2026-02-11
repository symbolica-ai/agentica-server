__all__ = [
    'AgentMonads',
]

from dataclasses import dataclass
from typing import Callable

from agentica_internal.session_manager_messages import PromptTemplate

from com.abstract import HistoryMonad

from ..models import ProviderModel
from .repl_tool.multi_turn import anthropic, openai


@dataclass
class AgentMonads:
    init_monad: Callable[[str | None, str | PromptTemplate | None], HistoryMonad[None]]
    interaction_monad: HistoryMonad[None]
    user_monad: Callable[[str | PromptTemplate, str | PromptTemplate | None], HistoryMonad[None]]

    @classmethod
    def from_model(cls, model: ProviderModel) -> 'AgentMonads':
        match (model.provider, model.model):
            case ('anthropic', _):
                monad = anthropic
            case _:
                monad = openai

        return AgentMonads(
            init_monad=monad.system_monad,
            user_monad=monad.user_monad,
            interaction_monad=monad.interaction_monad(),
        )
