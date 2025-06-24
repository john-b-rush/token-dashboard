// Skeleton main.rs for ratatui dashboard
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    tty::IsTty,
};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::{process::Command, thread, time::Duration};

use chrono::Utc;
use crossterm::event::{poll, read, Event, KeyCode};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::*;
use ratatui::symbols;
use ratatui::text::Span;
use ratatui::widgets::*;
use ratatui::{backend::CrosstermBackend, Terminal};
use serde::Deserialize;
use std::io::stdout;
use std::time::Instant;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelStats {
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
    total_tokens: u64,
    input_cost: f64,
    output_cost: f64,
    cache_creation_cost: f64,
    cache_read_cost: f64,
    total_cost: f64,
}

type DailyModelReport = HashMap<String, HashMap<String, ModelStats>>; // date -> model -> stats

fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000_000 {
        format!("{:.1}B", tokens as f64 / 1_000_000_000.0)
    } else if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.0}K", tokens as f64 / 1_000.0)
    } else {
        format!("{}", tokens)
    }
}

fn normalize_model_name(model: &str) -> String {
    let model_lower = model.to_lowercase();

    // Map Claude model names to cleaner versions
    if model_lower.contains("opus-4") {
        "Opus 4".to_string()
    } else if model_lower.contains("sonnet-4") {
        "Sonnet 4".to_string()
    } else if (model_lower.contains("claude-3.7") && model_lower.contains("sonnet"))
        || model_lower.contains("3-7-sonnet")
    {
        "Sonnet 3.7".to_string()
    } else if model_lower.contains("claude-3-5-sonnet") || model_lower.contains("claude-3.5-sonnet")
    {
        "Claude 3.5 Sonnet".to_string()
    } else if model_lower.contains("claude-3-5-haiku") || model_lower.contains("claude-3.5-haiku") {
        "Claude 3.5 Haiku".to_string()
    } else if model_lower.contains("claude-3-opus") {
        "Claude 3 Opus".to_string()
    // Map OpenAI/goose model names
    } else if model_lower == "o3" || model_lower.contains("o3-2025") {
        "o3".to_string()
    } else if model_lower.contains("o1") {
        "o1".to_string()
    } else if model_lower.contains("gpt-4o") {
        "GPT-4o".to_string()
    } else if model_lower.contains("gpt-4") && !model_lower.contains("gpt-4o") {
        "GPT-4".to_string()
    } else if model_lower == "unknown" {
        "Unknown".to_string()
    } else {
        model.to_string() // Return original if no match
    }
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
}

impl TerminalGuard {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        if !std::io::stdin().is_tty() {
            eprintln!("Error: This application requires a TTY terminal to run.");
            std::process::exit(1);
        }

        enable_raw_mode().map_err(|e| {
            eprintln!("Failed to enable raw mode: {}", e);
            e
        })?;

        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen).map_err(|e| {
            let _ = disable_raw_mode();
            eprintln!("Failed to enter alternate screen: {}", e);
            e
        })?;

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).map_err(|e| {
            let _ = disable_raw_mode();
            eprintln!("Failed to create terminal: {}", e);
            e
        })?;

        Ok(TerminalGuard { terminal })
    }

    fn terminal_mut(&mut self) -> &mut Terminal<CrosstermBackend<std::io::Stdout>> {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

fn load_data() -> Option<DailyModelReport> {
    use std::fs;

    // Run the parser to generate the JSON file
    let output = Command::new("python3")
        .arg("parser.py")
        .arg("--format")
        .arg("json")
        .arg("--output")
        .arg("data.json")
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!(
            "Python parser failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    // Read the generated JSON file
    let text = fs::read_to_string("data.json").ok()?;

    match serde_json::from_str::<DailyModelReport>(&text) {
        Ok(data) => Some(data),
        Err(e) => {
            eprintln!("JSON parsing failed: {}", e);
            None
        }
    }
}

fn draw_ui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    data: &DailyModelReport,
) -> std::io::Result<()> {
    terminal.draw(|f| {
        let full = f.size();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Length(7), Constraint::Min(0)])
            .split(full);

        let summary_block = Block::default()
            .title(" Token Usage Dashboard ")
            .borders(Borders::ALL);

        let mut lifetime_cost = 0.0;
        let mut mtd_cost = 0.0;
        let mut today_cost = 0.0;

        let mut lifetime_input = 0u64;
        let mut lifetime_output = 0u64;
        let mut lifetime_cache_read = 0u64;
        let mut lifetime_cache_write = 0u64;

        let mut _mtd_input = 0u64;
        let mut _mtd_output = 0u64;
        let mut _mtd_cache_read = 0u64;
        let mut _mtd_cache_write = 0u64;

        let mut today_input = 0u64;
        let mut today_output = 0u64;
        let mut today_cache_read = 0u64;
        let mut today_cache_write = 0u64;

        let today = Utc::now().naive_utc().date();
        let today_str = today.format("%Y-%m-%d").to_string();
        let this_month = today.format("%Y-%m").to_string();

        for (date, model_map) in data.iter() {
            for (_model, stat) in model_map.iter() {
                // Lifetime totals
                lifetime_cost += stat.total_cost;
                lifetime_input += stat.input_tokens;
                lifetime_output += stat.output_tokens;
                lifetime_cache_read += stat.cache_read_tokens;
                lifetime_cache_write += stat.cache_creation_tokens;

                // Month-to-date
                if date.starts_with(&this_month) {
                    mtd_cost += stat.total_cost;
                    _mtd_input += stat.input_tokens;
                    _mtd_output += stat.output_tokens;
                    _mtd_cache_read += stat.cache_read_tokens;
                    _mtd_cache_write += stat.cache_creation_tokens;
                }

                // Today
                if date == &today_str {
                    today_cost += stat.total_cost;
                    today_input += stat.input_tokens;
                    today_output += stat.output_tokens;
                    today_cache_read += stat.cache_read_tokens;
                    today_cache_write += stat.cache_creation_tokens;
                }
            }
        }

        let summary_text = Paragraph::new(format!(
            "Lifetime Spend   ${:.2}\nLifetime Tokens  {}/{} ({}/{})\nMonth-to-Date    ${:.2}\nToday's Spend    ${:.2}\nToday's Tokens   {}/{} ({}/{})",
            lifetime_cost,
            format_tokens(lifetime_input),
            format_tokens(lifetime_output),
            format_tokens(lifetime_cache_read),
            format_tokens(lifetime_cache_write),
            mtd_cost,
            today_cost,
            format_tokens(today_input),
            format_tokens(today_output),
            format_tokens(today_cache_read),
            format_tokens(today_cache_write)
        ))
        .block(summary_block)
        .style(Style::default().fg(Color::Green));

        f.render_widget(summary_text, layout[0]);

        let chart_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(layout[1]);

        let top_row = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chart_layout[0]);

        let bottom_row = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chart_layout[1]);

        // Render actual charts instead of placeholder blocks
        render_stacked_bar(f, top_row[0], data, true); // Tokens/day
        render_stacked_bar(f, top_row[1], data, false); // Cost/day
        render_today_bar(f, bottom_row[0], data, true); // Today's tokens
        render_today_bar(f, bottom_row[1], data, false); // Today's cost
    })?;
    Ok(())
}

fn render_stacked_bar(
    f: &mut ratatui::Frame,
    area: Rect,
    data: &DailyModelReport,
    is_tokens: bool,
) {
    // Split area into legend (left) and chart (right) sections
    let sections = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(25), Constraint::Min(0)])
        .split(area);

    // Add left padding to legend area
    let legend_with_padding = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(2), Constraint::Min(0)])
        .split(sections[0]);

    let legend_area = legend_with_padding[1];
    let chart_area = sections[1];

    let mut dates: Vec<String> = data.keys().cloned().collect();
    dates.sort();

    if dates.is_empty() {
        return;
    }

    // Collect all unique models and calculate their total spend
    let mut model_totals: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for models_map in data.values() {
        for (model, stats) in models_map.iter() {
            *model_totals.entry(model.clone()).or_insert(0.0) += stats.total_cost;
        }
    }

    // Sort models by total spend (descending)
    let mut sorted_models: Vec<String> = model_totals.keys().cloned().collect();
    sorted_models.sort_by(|a, b| {
        model_totals
            .get(b)
            .unwrap_or(&0.0)
            .partial_cmp(model_totals.get(a).unwrap_or(&0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Create complete time series for ALL models
    let chart_data: Vec<(String, Vec<(f64, f64)>)> = sorted_models
        .iter()
        .map(|model| {
            let points: Vec<(f64, f64)> = dates
                .iter()
                .enumerate()
                .map(|(i, date)| {
                    let value = data
                        .get(date)
                        .and_then(|models| models.get(model))
                        .map(|stat| {
                            if is_tokens {
                                stat.total_tokens as f64
                            } else {
                                stat.total_cost
                            }
                        })
                        .unwrap_or(0.0);
                    (i as f64, value)
                })
                .collect();
            (model.clone(), points)
        })
        .collect();

    let colors = [
        Color::Cyan,
        Color::Yellow,
        Color::Green,
        Color::Red,
        Color::Blue,
        Color::Magenta,
        Color::LightCyan,
        Color::LightYellow,
    ];

    // Create datasets for ALL models (no legend on chart, we'll do it separately)
    let mut datasets: Vec<Dataset> = Vec::new();

    for (i, (_model, points)) in chart_data.iter().enumerate() {
        datasets.push(
            Dataset::default()
                .name("") // No names on chart datasets
                .marker(symbols::Marker::Braille)
                .graph_type(ratatui::widgets::GraphType::Line)
                .style(Style::default().fg(colors[i % colors.len()]))
                .data(points),
        );
    }

    // Calculate Y-axis bounds
    let mut max_val = 0.0;
    for (_, points) in &chart_data {
        for (_, val) in points {
            if *val > max_val {
                max_val = *val;
            }
        }
    }

    // Create X-axis labels
    let label_step = if dates.len() <= 10 {
        2
    } else {
        dates.len() / 5
    };
    let mut x_labels: Vec<Span> = Vec::new();

    for i in (0..dates.len()).step_by(label_step.max(1)) {
        let date = &dates[i];
        if date.len() >= 10 {
            x_labels.push(Span::raw(format!("{}-{}", &date[5..7], &date[8..10])));
        } else {
            x_labels.push(Span::raw(date.clone()));
        }
    }

    // Format Y-axis values
    let y_max = max_val * 1.1;
    let y_step = y_max / 5.0;
    let y_labels: Vec<Span> = (0..6)
        .map(|i| {
            let value = i as f64 * y_step;
            if is_tokens {
                if value >= 1_000_000.0 {
                    Span::raw(format!("{:.1}M", value / 1_000_000.0))
                } else if value >= 1_000.0 {
                    Span::raw(format!("{:.0}K", value / 1_000.0))
                } else {
                    Span::raw(format!("{:.0}", value))
                }
            } else {
                Span::raw(format!("${:.0}", value))
            }
        })
        .collect();

    // Render the chart
    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(if is_tokens {
            "Tokens/day"
        } else {
            "Cost/day ($)"
        }))
        .x_axis(
            Axis::default()
                .title("Date")
                .bounds([0.0, (dates.len().max(1) - 1) as f64])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title(if is_tokens { "Tokens" } else { "Cost" })
                .bounds([0.0, y_max])
                .labels(y_labels),
        );

    f.render_widget(chart, chart_area);

    // Render the legend separately with matching colors (vertical format)
    use ratatui::text::{Line, Text};

    let mut legend_lines = vec![
        Line::from("All Models:").style(Style::default().fg(Color::White)),
        Line::from(""), // Empty line for spacing
    ];

    for (i, model) in sorted_models.iter().enumerate() {
        let color = colors[i % colors.len()];
        let color_char = match color {
            Color::Cyan => "◆",
            Color::Yellow => "●",
            Color::Green => "▲",
            Color::Red => "■",
            Color::Blue => "♦",
            Color::Magenta => "★",
            Color::LightCyan => "▼",
            Color::LightYellow => "♠",
            _ => "●",
        };
        let cost = model_totals.get(model).unwrap_or(&0.0);
        let line_text = format!(
            "{} {} (${:.0})",
            color_char,
            normalize_model_name(model),
            cost
        );
        legend_lines.push(Line::from(line_text).style(Style::default().fg(color)));
    }

    let legend_text = Text::from(legend_lines);
    let legend_paragraph = Paragraph::new(legend_text)
        .block(Block::default())
        .style(Style::default())
        .wrap(ratatui::widgets::Wrap { trim: true })
        .alignment(ratatui::layout::Alignment::Left);

    f.render_widget(legend_paragraph, legend_area);
}

fn render_today_bar(f: &mut ratatui::Frame, area: Rect, data: &DailyModelReport, is_tokens: bool) {
    let today = Utc::now().naive_utc().date().format("%Y-%m-%d").to_string();
    let today_data = match data.get(&today) {
        Some(val) => val,
        None => {
            // Show empty chart if there's no data for today
            render_empty_bar(f, area, &today, is_tokens);
            return;
        }
    };

    let mut items: Vec<(&String, f64)> = today_data
        .iter()
        .map(|(model, stat)| {
            (
                model,
                if is_tokens {
                    stat.total_tokens as f64
                } else {
                    stat.total_cost
                },
            )
        })
        .collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let labels: Vec<Span> = items
        .iter()
        .map(|(m, _)| Span::raw(normalize_model_name(m)))
        .collect();
    let values: Vec<u64> = items
        .iter()
        .map(|(_, v)| {
            if is_tokens {
                *v as u64
            } else {
                // For costs, multiply by 100 to show cents as integers
                (*v * 100.0) as u64
            }
        })
        .collect();

    // Find max value for scaling
    let max_val = items.iter().map(|(_, v)| *v).fold(0.0, f64::max);

    let bar_data: Vec<(&str, u64)> = labels
        .iter()
        .zip(values.iter())
        .map(|(s, v)| (s.content.as_ref(), *v))
        .collect();

    let title = if is_tokens {
        format!(
            "Today: Tokens by Model (Max: {})",
            if max_val >= 1_000_000.0 {
                format!("{:.1}M", max_val / 1_000_000.0)
            } else if max_val >= 1_000.0 {
                format!("{:.0}K", max_val / 1_000.0)
            } else {
                format!("{:.0}", max_val)
            }
        )
    } else {
        format!("Today: Cost by Model (Max: ${:.2})", max_val)
    };

    let bars = BarChart::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .bar_width(10)
        .bar_gap(3)
        .value_style(
            Style::default()
                .fg(Color::White)
                .bg(Color::Black)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )
        .label_style(Style::default().fg(Color::White))
        .bar_style(Style::default().fg(if is_tokens {
            Color::Green
        } else {
            Color::Magenta
        }))
        .data(&bar_data);

    f.render_widget(bars, area);
}

fn render_empty_bar(f: &mut ratatui::Frame, area: Rect, _date: &str, is_tokens: bool) {
    let title = if is_tokens {
        "Today: Tokens by Model (No data)".to_string()
    } else {
        "Today: Cost by Model (No data)".to_string()
    };

    let bars = BarChart::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .bar_width(10)
        .bar_gap(3)
        .value_style(
            Style::default()
                .fg(Color::White)
                .bg(Color::Black)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )
        .label_style(Style::default().fg(Color::White))
        .bar_style(Style::default().fg(if is_tokens {
            Color::Green
        } else {
            Color::Magenta
        }))
        .data(&[]);

    f.render_widget(bars, area);
}

#[allow(dead_code)]
fn render_recent_bar(
    f: &mut ratatui::Frame,
    area: Rect,
    data: &HashMap<String, ModelStats>,
    date: &str,
    is_tokens: bool,
) {
    let mut items: Vec<(&String, f64)> = data
        .iter()
        .map(|(model, stat)| {
            (
                model,
                if is_tokens {
                    stat.total_tokens as f64
                } else {
                    stat.total_cost
                },
            )
        })
        .collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let labels: Vec<Span> = items
        .iter()
        .map(|(m, _)| Span::raw(normalize_model_name(m)))
        .collect();
    let values: Vec<u64> = items
        .iter()
        .map(|(_, v)| {
            if is_tokens {
                *v as u64
            } else {
                // For costs, multiply by 100 to show cents as integers
                (*v * 100.0) as u64
            }
        })
        .collect();

    // Find max value for scaling
    let max_val = items.iter().map(|(_, v)| *v).fold(0.0, f64::max);

    let bar_data: Vec<(&str, u64)> = labels
        .iter()
        .zip(values.iter())
        .map(|(s, v)| (s.content.as_ref(), *v))
        .collect();

    let title = if is_tokens {
        format!(
            "{}: Tokens by Model (Max: {})",
            date,
            if max_val >= 1_000_000.0 {
                format!("{:.1}M", max_val / 1_000_000.0)
            } else if max_val >= 1_000.0 {
                format!("{:.0}K", max_val / 1_000.0)
            } else {
                format!("{:.0}", max_val)
            }
        )
    } else {
        format!("{}: Cost by Model (Max: ${:.2})", date, max_val)
    };

    let bars = BarChart::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .bar_width(10)
        .bar_gap(3)
        .value_style(
            Style::default()
                .fg(Color::White)
                .bg(Color::Black)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )
        .label_style(Style::default().fg(Color::White))
        .bar_style(Style::default().fg(if is_tokens {
            Color::Green
        } else {
            Color::Magenta
        }))
        .data(&bar_data);

    f.render_widget(bars, area);
}

fn setup_signal_handler() -> Result<Arc<AtomicBool>, Box<dyn std::error::Error>> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    Ok(running)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let running = setup_signal_handler()?;
    let mut terminal_guard = TerminalGuard::new()?;

    let refresh_interval = Duration::from_secs(10);
    let mut last_update = Instant::now() - refresh_interval;

    // Load data immediately on startup
    let mut current_data = load_data();

    while running.load(Ordering::SeqCst) {
        // Input check first
        if poll(Duration::from_millis(100))? {
            if let Event::Key(key) = read()? {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                    break;
                }
            }
        }

        // Refresh data on interval
        if last_update.elapsed() >= refresh_interval {
            if let Some(data) = load_data() {
                current_data = Some(data);
            }
            last_update = Instant::now();
        }

        // Always try to render
        if let Some(ref data) = current_data {
            draw_ui(terminal_guard.terminal_mut(), data)?;
        } else {
            // Show loading/error screen
            terminal_guard.terminal_mut().draw(|f| {
                let block = Block::default()
                    .title(" Token Usage Dashboard - No Data ")
                    .borders(Borders::ALL);
                let paragraph =
                    Paragraph::new("Unable to load data from parser.\nPress 'q' or Esc to quit")
                        .block(block)
                        .style(Style::default().fg(Color::Red));
                f.render_widget(paragraph, f.size());
            })?;
        }

        // Avoid burning CPU
        thread::sleep(Duration::from_millis(50));
    }

    Ok(())
}
