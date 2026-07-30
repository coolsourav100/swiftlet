# Contributing to Swiftlet

Thank you for your interest in contributing to **Swiftlet**! We welcome bug reports, feature requests, and code contributions from the community.

## How to Contribute

### 1. Reporting Bugs
- Check the issue tracker to ensure the bug hasn't already been reported.
- Open a new issue using the **Bug Report** template.
- Include as much context as possible (macOS version, Mac model/RAM, error logs, etc.).

### 2. Suggesting Features
- Open a new issue using the **Feature Request** template.
- Explain *why* this feature would be useful and how it fits into the Swiftlet architecture (proxy layer vs UI layer).

### 3. Submitting Pull Requests
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
   - If changing the proxy routing logic (`swiftlet/orchestrator.py` or `swiftlet/cli.py`), ensure it still handles dynamic routing correctly.
   - If changing the UI (`swiftlet/app.py`), ensure it works in standard terminal sizes.
4. Test your changes locally on an Apple Silicon Mac using the `bash start.command` workflow.
5. Commit your changes (`git commit -m "feat: your feature description"`).
6. Push to your branch and open a Pull Request!

## Pull Request Review Process
We have an **AI Bot Reviewer** installed in this repository. When you open a PR, the bot will automatically review your code, summarize the changes, and leave helpful suggestions. Please address any critical issues flagged by the bot before requesting a review from the maintainers.

## Code Structure Overview
- `swiftlet/cli.py`: The HTTP Proxy engine that intercepts requests and manages `llama-server`.
- `swiftlet/orchestrator.py`: The brain that decides hardware splits based on prompt length vs expected output.
- `swiftlet/app.py`: The Textual-based terminal chat UI.

Happy coding!
