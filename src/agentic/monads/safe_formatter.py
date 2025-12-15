import re

IDENT = r'[A-Z][A-Z_]*'
R_SAFE_VAR = re.compile(IDENT)
R_TEMPLATE_VAR = re.compile(r'\{\{\s*(' + IDENT + r')\s*\}\}')


class SafeFormatter:
    base: dict[str, str]

    def __init__(self, base: dict[str, str] | None = None, **kwargs: str) -> None:
        base = base | kwargs if base else kwargs
        self._verify_kwargs(base)
        self.base = base

    def format(self, template: str, /, extra: dict[str, str] | None = None, **kwargs: str) -> str:
        extra = extra | kwargs if extra else kwargs
        self._verify_kwargs(extra)
        all_vars = {**self.base, **extra}

        def replace_match(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name not in all_vars:
                raise KeyError(f"Variable '{var_name}' not found in formatter context")
            return all_vars[var_name]

        return R_TEMPLATE_VAR.sub(replace_match, template)

    def _verify_kwargs(self, kwargs: dict[str, str]) -> None:
        for k, v in kwargs.items():
            if not R_SAFE_VAR.fullmatch(k):
                raise ValueError(f"Invalid variable name: {k}")
            if not isinstance(v, str):
                kwargs[k] = str(v)
