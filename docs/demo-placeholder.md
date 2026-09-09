# Demo recording

This folder is reserved for a short screen recording of the agent driving a browser.

## What to record

1. Run a live goal:
   ```bash
   python -m agentic "Find the pricing page" "https://example.com"
   ```
2. Show the page loading, the agent reading elements, clicking, and reaching `done`.
3. Keep it under 30 seconds. Headed mode helps: `AGENT_HEADED=1`.

## Tools

- Windows: Xbox Game Bar (`Win + G`) or OBS Studio.
- Convert to GIF with [ScreenToGif](https://www.screentogif.com/) or `ffmpeg`.

Save the final file as `docs/demo.gif` and it will render automatically below.

![demo](demo.gif)
