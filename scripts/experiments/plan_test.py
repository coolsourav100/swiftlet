from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll, Vertical
from textual.widgets import Input, Markdown, Collapsible, Label, Static

class SwiftletApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    #chat-pane {
        width: 3fr;
        height: 100%;
        border-right: solid green;
    }
    #sidebar {
        width: 26;
        height: 100%;
    }
    """
    def compose(self) -> ComposeResult:
        yield Horizontal(
            VerticalScroll(id="chat-pane"),
            Vertical(id="sidebar")
        )

app = SwiftletApp()
# print(app.CSS)
