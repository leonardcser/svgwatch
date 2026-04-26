# svgwatch

GPU-accelerated SVG viewer built on [Vello](https://github.com/linebender/vello)
and [vello_svg](https://github.com/linebender/vello_svg). Live-reloads on file
change.

## Run

```bash
cargo run --release -- path/to/file.svg
```

## Controls

| Input                        | Action                            |
| ---------------------------- | --------------------------------- |
| Scroll                       | Pan (trackpad) / zoom (mouse)     |
| Cmd/Ctrl + scroll, pinch     | Zoom at cursor                    |
| Middle-click drag            | Pan                               |
| Arrow keys / `h` `j` `k` `l` | Step pan (Shift = larger)         |
| `f`, `0`                     | Fit to window                     |
| `+` / `-`                    | Zoom in / out                     |
| Cmd+F                        | Find text                         |
| Enter / Shift+Enter          | Next / previous match             |
| Esc                          | Close search                      |
| Ctrl+C, Ctrl+D, Cmd+W        | Quit                              |

## Notes

- Vello SVG is best-effort — some masks, filters, and pattern gradients may not
  render exactly. For spec-correct rendering use resvg + tiny-skia (CPU).
- Backed by wgpu (Metal on macOS, Vulkan on Linux, DX12 on Windows). Search UI
  is rendered with [egui](https://github.com/emilk/egui) on top of the same
  wgpu surface.
