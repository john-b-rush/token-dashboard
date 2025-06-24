# Token Usage Dashboard

A real-time terminal dashboard for tracking Claude and OpenAI token usage and costs across your projects when using Claude Code or Goose CLI. Monitor and analyze both model-based and project-based consumption patterns.

![Dashboard Screenshot](screenshot.png)

## Features

- 🔄 **Real-time monitoring** - Updates every 10 seconds
- 📊 **Visual charts** - Line graphs for usage trends, bar charts for daily breakdowns
- 💰 **Cost tracking** - Accurate pricing for all major models including cache costs
- 🎯 **Dual views** - Toggle between Model and Project perspectives
- 📈 **Multiple timeframes** - Lifetime, month-to-date, and daily spending
- 🚀 **Fast performance** - Parallel log scanning with intelligent caching
- 🔧 **Multi-source** - Supports both Claude Code and Goose CLI logs

## Supported Models

### Claude Models
- **Claude Sonnet 4** - Latest high-performance model
- **Claude Opus 4** - Most capable model
- **Claude Sonnet 3.7** - Updated Sonnet variant
- **Claude 3.5 Sonnet/Haiku** - Previous generation models

### OpenAI Models  
- **o3** - Latest reasoning model
- **o1** - Advanced reasoning model
- **GPT-4o** - Multimodal flagship
- **GPT-4** - Standard GPT-4 variants

## Installation

### Prerequisites
- **Rust** (latest stable)
- **Claude Code CLI** or **Goose** with conversation logs

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/token-dashboard.git
   cd token-dashboard
   ```

2. **Build the dashboard**
   ```bash
   cargo build --release
   ```

3. **Run the dashboard**
   ```bash
   cargo run
   ```

## Usage

### Running the Dashboard
```bash
# Standard usage
cargo run

# Or use the release build
./target/release/token-dashboard
```

### Controls
- **`M`** - Switch to Model View (group data by AI model)
- **`P`** - Switch to Project View (group data by codebase project)
- **`Q` or `Esc`** - Quit the dashboard
- **Auto-refresh** - Updates every 10 seconds

### Understanding the Display

The dashboard shows five main sections:

1. **Summary Panel** (top-left)
   - Lifetime spend and token counts
   - Month-to-date totals
   - Today's usage and costs

2. **Controls Panel** (top-right)
   - Available keyboard shortcuts
   - Shows current view mode

3. **Usage Trends** (middle-left chart)
   - Daily token usage over time
   - Color-coded by model or project
   - View changes based on current mode (M/P)

4. **Cost Trends** (middle-right chart)  
   - Daily cost over time
   - Same grouping as Usage Trends
   - Dollar amounts on Y-axis

5. **Daily Breakdowns** (bottom charts)
   - Today's usage by model or project
   - Bar charts showing current day totals

## Data Sources & Configuration

The dashboard automatically scans logs from:

- **Claude Code**: `~/.claude/projects/` - JSONL conversation files
- **Goose CLI**: `~/.local/state/goose/logs/cli/` - Session logs

Performance optimizations:
- **Caching**: `~/.cache/token-dashboard/` - Only re-parses modified files
- **Parallel Processing**: Uses all CPU cores for fast scanning
- **Clear cache**: `rm -rf ~/.cache/token-dashboard/` if needed

## Project Views

One of the key features is the ability to toggle between different perspectives:

- **Model View (M key)**: Groups all usage data by AI model
  - See which models cost the most across all projects
  - Identify high-token-consumption models

- **Project View (P key)**: Groups usage data by codebase project
  - Track which projects consume the most tokens
  - Monitor spending across different repositories

## Installation & Usage

```bash
# Build the dashboard
cargo build --release

# Run with cargo
cargo run

# Or use the release binary
./target/release/token-dashboard
```

## Troubleshooting

- **No data?** Verify log directories exist and contain files
- **Test data loading**: `cargo run -- --test`
- **Display issues**: Ensure terminal supports colors and is large enough

## License

MIT License

## Credits

Built with [ratatui](https://github.com/ratatui-org/ratatui) for terminal UI rendering.
