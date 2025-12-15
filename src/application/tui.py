"""
Interactive TUI for the Session Manager using Textual + Python's InteractiveInterpreter.

Provides:
- Full Python REPL with proper multi-line support
- Tab completion with popup grid
- Rich print support for formatted output
- Logs displayed in scrolling area above
- Input box fixed at the bottom
"""

import asyncio
import builtins
import code
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

HISTORY_FILE = Path.home() / ".session_manager_history"
HISTORY_MAX_SIZE = 1000

from rich.columns import Columns
from rich.console import Console
from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.suggester import Suggester
from textual.widgets import Input, RichLog, Static

if TYPE_CHECKING:
    from application.main import SessionManager


def _sort_completions(completions: list[str]) -> list[str]:
    """Sort completions with _ and __ prefixed items last."""

    def sort_key(name: str) -> tuple[int, str]:
        # Get the last part after any dots for sorting
        last_part = name.rsplit(".", 1)[-1]
        if last_part.startswith("__"):
            return (2, name.lower())  # Dunder last
        elif last_part.startswith("_"):
            return (1, name.lower())  # Single underscore second
        else:
            return (0, name.lower())  # Public first

    return sorted(completions, key=sort_key)


class PythonSuggester(Suggester):
    """Suggester that shows gray autosuggestion for Python completions."""

    def __init__(self, namespace: dict[str, Any]):
        super().__init__(use_cache=False, case_sensitive=True)
        self.namespace = namespace

    async def get_suggestion(self, value: str) -> str | None:
        """Get a suggestion for the current input."""
        if not value:
            return None

        # Find the word being completed (last token)
        word = ""
        for i in range(len(value) - 1, -1, -1):
            c = value[i]
            if c.isalnum() or c in "_.":
                word = c + word
            else:
                break

        if not word:
            return None

        # Get first completion
        completions = self._get_completions(word)
        if completions:
            prefix = value[: len(value) - len(word)]
            return prefix + completions[0]
        return None

    def _get_completions(self, word: str) -> list[str]:
        """Get completions for a word."""
        completions = []

        if "." in word:
            parts = word.rsplit(".", 1)
            prefix = parts[0]
            partial = parts[1] if len(parts) > 1 else ""
            try:
                obj = eval(prefix, self.namespace)
                for attr in dir(obj):
                    if attr.startswith(partial):
                        completions.append(f"{prefix}.{attr}")
            except Exception:
                pass
        else:
            for name in self.namespace:
                if name.startswith(word):
                    completions.append(name)
            for name in dir(builtins):
                if name.startswith(word) and name not in completions:
                    completions.append(name)

        return _sort_completions(completions)


class TUILogHandler(logging.Handler):
    """Log handler that writes to the Textual RichLog widget."""

    def __init__(self, log_widget: RichLog):
        super().__init__()
        self.log_widget = log_widget
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                self.log_widget.write(f"[bold red]{msg}[/bold red]")
            elif record.levelno >= logging.WARNING:
                self.log_widget.write(f"[yellow]{msg}[/yellow]")
            elif record.levelno <= logging.DEBUG:
                self.log_widget.write(f"[dim]{msg}[/dim]")
            else:
                self.log_widget.write(f"[cyan]{msg}[/cyan]")
        except Exception:
            self.handleError(record)


class RichLogFile:
    """A file-like object that writes to a RichLog widget."""

    def __init__(self, log_widget: RichLog, style: str = ""):
        self.log_widget = log_widget
        self.style = style
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                if self.style:
                    self.log_widget.write(f"[{self.style}]{line}[/{self.style}]")
                else:
                    self.log_widget.write(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            if self.style:
                self.log_widget.write(f"[{self.style}]{self._buffer}[/{self.style}]")
            else:
                self.log_widget.write(self._buffer)
            self._buffer = ""


class TUIInterpreter(code.InteractiveInterpreter):
    """Interactive interpreter that outputs to a RichLog widget."""

    def __init__(self, log_widget: RichLog, locals: dict[str, Any]):
        super().__init__(locals)
        self.log_widget = log_widget
        self.buffer: list[str] = []
        # Create a Rich console that writes to the log widget
        self._rich_file = RichLogFile(log_widget, style="yellow")
        self._rich_console = Console(file=self._rich_file, force_terminal=True, width=120)
        # Inject into namespace for user access
        locals["console"] = self._rich_console

    def write(self, data: str) -> None:
        """Override to write errors to the RichLog."""
        if data.strip():
            self.log_widget.write(f"[bold red]{data.rstrip()}[/bold red]")

    def _displayhook(self, value: Any) -> None:
        """Custom displayhook that uses Rich pretty printing."""
        if value is None:
            return
        # Store in _ like the standard REPL
        self.locals["_"] = value
        # Pretty print using Rich
        self.log_widget.write(Pretty(value, indent_guides=True, expand_all=True))

    def runcode(self, code_obj) -> None:
        """Execute code and capture stdout."""
        # Create file wrappers for stdout/stderr
        stdout_file = RichLogFile(self.log_widget, style="yellow")
        stderr_file = RichLogFile(self.log_widget, style="bold red")

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_displayhook = sys.displayhook

        try:
            sys.stdout = stdout_file  # type: ignore
            sys.stderr = stderr_file  # type: ignore
            sys.displayhook = self._displayhook
            exec(code_obj, self.locals)
        except SystemExit:
            raise
        except:
            # Restore before showing traceback
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.displayhook = old_displayhook
            self.showtraceback()
        else:
            # Flush any remaining buffered output
            stdout_file.flush()
            stderr_file.flush()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.displayhook = old_displayhook

    def push(self, line: str) -> bool:
        """Push a line of code. Returns True if more input is needed."""
        self.buffer.append(line)
        source = "\n".join(self.buffer)
        more = self.runsource(source, "<input>")
        if not more:
            self.buffer = []
        return more


class CompletionState:
    """Tracks completion cycling state."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.completions: list[str] = []
        self.index = 0
        self.original_value = ""
        self.original_cursor = 0
        self.word_start = 0
        self.column_count = 1  # Calculated when showing completions

    def move_right(self) -> None:
        """Move right one item."""
        self.index = (self.index + 1) % len(self.completions)

    def move_left(self) -> None:
        """Move left one item."""
        self.index = (self.index - 1) % len(self.completions)

    def move_down(self) -> None:
        """Move down one row."""
        new_index = self.index + self.column_count
        if new_index < len(self.completions):
            self.index = new_index
        else:
            # Wrap to top of column
            col = self.index % self.column_count
            self.index = col

    def move_up(self) -> None:
        """Move up one row."""
        new_index = self.index - self.column_count
        if new_index >= 0:
            self.index = new_index
        else:
            # Wrap to bottom of column
            col = self.index % self.column_count
            # Find last row with this column
            total = len(self.completions)
            last_row_start = ((total - 1) // self.column_count) * self.column_count
            new_index = last_row_start + col
            if new_index >= total:
                new_index -= self.column_count
            if new_index >= 0:
                self.index = new_index


class CompletionInput(Input):
    """Input that defers arrow keys to app when completions are active."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completion_state: CompletionState | None = None

    async def _on_key(self, event: Key) -> None:
        """Intercept keys for readline-like behavior and completions."""
        # Readline shortcuts
        if event.key == "ctrl+a":
            # Go to beginning of line
            self.cursor_position = 0
            event.prevent_default()
            event.stop()
            return
        if event.key == "ctrl+e":
            # Go to end of line
            self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()
            return
        if event.key == "ctrl+k":
            # Kill to end of line
            self.value = self.value[: self.cursor_position]
            event.prevent_default()
            event.stop()
            return
        if event.key == "ctrl+u":
            # Kill to beginning of line
            self.value = self.value[self.cursor_position :]
            self.cursor_position = 0
            event.prevent_default()
            event.stop()
            return

        # Defer arrow keys when completions are active
        if self.completion_state and self.completion_state.active:
            if event.key in ("up", "down", "left", "right"):
                # Let these bubble up to the app handler
                return
        # Otherwise, let Input handle it normally
        await super()._on_key(event)


class SessionManagerApp(App):
    """A Textual app for the Session Manager REPL."""

    # Disable mouse capture to allow native terminal text selection
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    RichLog {
        height: 1fr;
        border: solid green;
        background: transparent;
    }
    #input-area {
        dock: bottom;
        height: auto;
        max-height: 12;
        border: solid cyan;
        padding: 0 1;
        background: transparent;
    }
    #completions {
        height: auto;
        max-height: 8;
        border-bottom: solid magenta;
        padding: 0;
        display: none;
        background: transparent;
    }
    #completions.visible {
        display: block;
    }
    #input-row {
        height: 1;
        padding: 0;
        background: transparent;
    }
    #prompt {
        width: 4;
        height: 1;
        content-align: right middle;
        color: green;
        background: transparent;
    }
    #input {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0;
        background: transparent;
    }
    """

    BINDINGS = [
        Binding("ctrl+d", "quit", "Exit"),
        Binding("ctrl+c", "clear_input", "Clear", show=False),
        Binding("pageup", "scroll_up", "Scroll Up", show=False),
        Binding("pagedown", "scroll_down", "Scroll Down", show=False),
        Binding("escape", "toggle_mouse", "Select Mode"),
    ]

    def __init__(self, session_manager: "SessionManager"):
        super().__init__()
        self.session_manager = session_manager
        ie = session_manager.inference_endpoint_client()
        self.local_ns: dict[str, Any] = {
            "sm": session_manager,
            "ssm": session_manager._ssm,
            "app": session_manager._app,
            "ie": ie,
            "asyncio": asyncio,
        }
        self._suggester = PythonSuggester(self.local_ns)
        self._interpreter: TUIInterpreter | None = None
        self._log_handler: TUILogHandler | None = None
        self._server_task: asyncio.Task | None = None
        self._history: list[str] = self._load_history()
        self._history_index = len(self._history)
        self._multiline_mode = False
        self._completion = CompletionState()
        self._mouse_enabled = True

    def _load_history(self) -> list[str]:
        """Load command history from file."""
        if HISTORY_FILE.exists():
            try:
                return HISTORY_FILE.read_text().splitlines()[-HISTORY_MAX_SIZE:]
            except Exception:
                pass
        return []

    def _save_history(self) -> None:
        """Save command history to file."""
        try:
            # Keep only the last HISTORY_MAX_SIZE entries
            history = self._history[-HISTORY_MAX_SIZE:]
            HISTORY_FILE.write_text("\n".join(history))
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", highlight=True, markup=True, wrap=True)
        with Vertical(id="input-area"):
            yield Static(id="completions")
            with Horizontal(id="input-row"):
                yield Static(">>> ", id="prompt")
                input_widget = CompletionInput(
                    id="input",
                    suggester=self._suggester,
                )
                input_widget.completion_state = self._completion
                yield input_widget

    async def on_mount(self) -> None:
        """Set up when the app starts."""
        log = self.query_one("#log", RichLog)
        input_widget = self.query_one("#input", Input)
        input_widget.focus()

        # Add log widget to namespace
        self.local_ns["log"] = log

        # Create interpreter
        self._interpreter = TUIInterpreter(log, self.local_ns)

        # Set up log handler
        self._log_handler = TUILogHandler(log)
        self._log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self._log_handler)

        # Welcome message
        log.write("[bold cyan]" + "=" * 50 + "[/bold cyan]")
        log.write("[bold cyan]Session Manager Interactive Console[/bold cyan]")
        log.write("[bold cyan]" + "=" * 50 + "[/bold cyan]")
        log.write("")
        log.write("[dim]Available objects:[/dim]")
        log.write("  [magenta]sm[/magenta]      - SessionManager instance")
        log.write("  [magenta]ssm[/magenta]     - ServerSessionManager")
        log.write("  [magenta]ie[/magenta]      - httpx.Client endpoint (https://openrouter.ai/)")
        log.write("  [magenta]app[/magenta]     - Litestar application")
        log.write(
            "  [magenta]console[/magenta] - Rich Console (use console.print for formatted output)"
        )
        log.write("")
        log.write(
            "[dim]Tab to complete, Up/Down for history, Esc to toggle mouse mode, Ctrl+D to exit[/dim]"
        )
        log.write("")

        # Start the server
        self._server_task = asyncio.create_task(self.session_manager._server.serve())

    async def on_unmount(self) -> None:
        """Clean up when the app stops."""
        # Save history
        self._save_history()

        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)

        if self._server_task:
            self.session_manager._server.should_exit = True
            try:
                await asyncio.wait_for(self._server_task, timeout=3.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        text = event.value
        log = self.query_one("#log", RichLog)
        input_widget = self.query_one("#input", Input)

        # Hide completions and reset state
        self._hide_completions()
        self._completion.reset()

        if not self._interpreter:
            return

        # Determine prompt style
        prompt = "..." if self._multiline_mode else ">>>"

        # Echo the input
        log.write(f"[bold green]{prompt} {text}[/bold green]")

        # Clear input
        input_widget.value = ""

        # Push to interpreter
        self._multiline_mode = self._interpreter.push(text)

        # Update prompt based on mode
        prompt = self.query_one("#prompt", Static)
        if self._multiline_mode:
            prompt.update("... ")
        else:
            prompt.update(">>> ")
            # Add to history only when command is complete
            if text.strip():
                self._history.append(text)
                self._history_index = len(self._history)

    def action_clear_input(self) -> None:
        """Clear the input field and reset multiline mode."""
        input_widget = self.query_one("#input", Input)
        input_widget.value = ""
        if self._interpreter:
            self._interpreter.buffer = []
        self._multiline_mode = False
        self._hide_completions()
        self._completion.reset()
        prompt = self.query_one("#prompt", Static)
        prompt.update(">>> ")

    def action_scroll_up(self) -> None:
        """Scroll log up."""
        log = self.query_one("#log", RichLog)
        log.scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        """Scroll log down."""
        log = self.query_one("#log", RichLog)
        log.scroll_down(animate=False)

    def action_toggle_mouse(self) -> None:
        """Toggle mouse mode between scroll and select."""
        self._mouse_enabled = not self._mouse_enabled
        # Write directly to the original stdout (bypassing any redirects)
        import sys

        out = sys.__stdout__
        if self._mouse_enabled:
            # Enable mouse tracking
            out.write("\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1006h")
            out.flush()
            self.notify("Mouse: scroll mode (Esc for select)")
        else:
            # Disable mouse tracking
            out.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l")
            out.flush()
            self.notify("Mouse: select mode (Esc for scroll)")

    def _get_all_completions(self, word: str) -> list[str]:
        """Get all completions for a word."""
        completions = []

        if "." in word:
            # Attribute completion
            parts = word.rsplit(".", 1)
            prefix = parts[0]
            partial = parts[1] if len(parts) > 1 else ""
            try:
                obj = eval(prefix, self.local_ns)
                for attr in dir(obj):
                    if attr.startswith(partial):
                        completions.append(f"{prefix}.{attr}")
            except Exception:
                pass
        else:
            # Namespace completion
            for name in self.local_ns:
                if name.startswith(word):
                    completions.append(name)
            # Builtins
            for name in dir(builtins):
                if name.startswith(word) and name not in completions:
                    completions.append(name)

        return _sort_completions(completions)

    def _apply_completion(self, input_widget: Input, completion: str) -> None:
        """Apply a completion to the input widget."""
        cs = self._completion
        prefix = cs.original_value[: cs.word_start]
        suffix = cs.original_value[cs.original_cursor :]
        input_widget.value = prefix + completion + suffix
        input_widget.cursor_position = len(prefix + completion)

    def _show_completions(self) -> None:
        """Show completion grid in the completions widget."""
        completions_widget = self.query_one("#completions", Static)
        cs = self._completion

        if cs.completions:
            # Build styled text items for grid
            items = []
            for i, comp in enumerate(cs.completions):
                if i == cs.index:
                    items.append(Text(comp, style="bold cyan reverse"))
                else:
                    items.append(Text(comp, style="white"))

            # Get the widget width for rendering
            try:
                width = completions_widget.size.width - 2
            except Exception:
                width = 80

            # Create Columns and render it to extract actual column count
            columns = Columns(items, equal=False, expand=False)

            # Render to a temporary console to get the table and its column count
            import io

            from rich.console import Console as RichConsole

            temp_console = RichConsole(file=io.StringIO(), width=width, force_terminal=True)
            # Render to get the generator result (a Table)

            options = temp_console.options.update_width(width)
            for table in columns.__rich_console__(temp_console, options):
                # The table's columns list tells us how many columns
                cs.column_count = max(1, len(table.columns))
                break

            completions_widget.update(columns)
            completions_widget.add_class("visible")

    def _hide_completions(self) -> None:
        """Hide the completions widget."""
        completions_widget = self.query_one("#completions", Static)
        completions_widget.remove_class("visible")
        completions_widget.update("")

    async def on_key(self, event) -> None:
        """Handle key presses."""
        input_widget = self.query_one("#input", Input)
        cs = self._completion

        if event.key == "ctrl+d":
            # Exit the app
            event.prevent_default()
            event.stop()
            self.exit()
            return

        if event.key == "tab":
            event.prevent_default()
            event.stop()

            if cs.active:
                # Cycle forward through completions
                cs.index = (cs.index + 1) % len(cs.completions)
                self._apply_completion(input_widget, cs.completions[cs.index])
                self._show_completions()
            else:
                # Start new completion
                value = input_widget.value
                cursor = input_widget.cursor_position
                text_before = value[:cursor]

                # Find the word being completed
                word = ""
                word_start = cursor
                for i in range(len(text_before) - 1, -1, -1):
                    c = text_before[i]
                    if c.isalnum() or c in "_.":
                        word = c + word
                        word_start = i
                    else:
                        break

                completions = self._get_all_completions(word) if word else []

                if len(completions) == 1:
                    # Single completion - apply directly
                    prefix = text_before[:word_start]
                    suffix = value[cursor:]
                    input_widget.value = prefix + completions[0] + suffix
                    input_widget.cursor_position = len(prefix + completions[0])
                elif completions:
                    # Multiple completions - start cycling
                    cs.active = True
                    cs.completions = completions
                    cs.index = 0
                    cs.original_value = value
                    cs.original_cursor = cursor
                    cs.word_start = word_start
                    self._apply_completion(input_widget, completions[0])
                    self._show_completions()
                elif not word:
                    # No word - insert 4 spaces
                    input_widget.value = value[:cursor] + "    " + value[cursor:]
                    input_widget.cursor_position = cursor + 4

        elif event.key == "shift+tab":
            event.prevent_default()
            event.stop()

            if cs.active:
                # Cycle backward through completions
                cs.index = (cs.index - 1) % len(cs.completions)
                self._apply_completion(input_widget, cs.completions[cs.index])
                self._show_completions()

        elif event.key == "escape":
            if cs.active:
                # Cancel completion, restore original
                input_widget.value = cs.original_value
                input_widget.cursor_position = cs.original_cursor
                self._hide_completions()
                cs.reset()
                event.prevent_default()

        elif event.key == "enter" and cs.active:
            # Accept current completion and hide
            self._hide_completions()
            cs.reset()
            # Don't prevent - let the input handle submission

        elif event.key == "up" and cs.active:
            event.prevent_default()
            event.stop()
            cs.move_up()
            self._apply_completion(input_widget, cs.completions[cs.index])
            self._show_completions()

        elif event.key == "down" and cs.active:
            event.prevent_default()
            event.stop()
            cs.move_down()
            self._apply_completion(input_widget, cs.completions[cs.index])
            self._show_completions()

        elif event.key == "left" and cs.active:
            event.prevent_default()
            event.stop()
            cs.move_left()
            self._apply_completion(input_widget, cs.completions[cs.index])
            self._show_completions()

        elif event.key == "right" and cs.active:
            event.prevent_default()
            event.stop()
            cs.move_right()
            self._apply_completion(input_widget, cs.completions[cs.index])
            self._show_completions()

        elif event.key == "up" and self._history and not self._multiline_mode:
            # History navigation
            event.prevent_default()
            event.stop()
            if self._history_index > 0:
                self._history_index -= 1
                input_widget.value = self._history[self._history_index]
                input_widget.cursor_position = len(input_widget.value)

        elif event.key == "down" and self._history and not self._multiline_mode:
            # History navigation
            event.prevent_default()
            event.stop()
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                input_widget.value = self._history[self._history_index]
                input_widget.cursor_position = len(input_widget.value)
            elif self._history_index == len(self._history) - 1:
                self._history_index = len(self._history)
                input_widget.value = ""

        else:
            # Any other key resets completion state
            if cs.active:
                self._hide_completions()
                cs.reset()


async def run_with_tui(session_manager: "SessionManager") -> None:
    """Run the session manager with the interactive TUI."""
    app = SessionManagerApp(session_manager)
    await app.run_async()
