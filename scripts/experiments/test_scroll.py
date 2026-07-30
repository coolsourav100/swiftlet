from textual.app import App, ComposeResult
from textual.widgets import Static
import asyncio

class TestApp(App):
    CSS = "#box { height: 5; overflow-y: scroll; border: solid green; }"
    def compose(self) -> ComposeResult:
        yield Static("", id="box")

    async def on_mount(self):
        self.text = ""
        self.set_interval(0.1, self.add_text)

    def add_text(self):
        self.text += "line\n"
        box = self.query_one("#box", Static)
        box.update(self.text)
        box.scroll_end(animate=False)

TestApp().run()
