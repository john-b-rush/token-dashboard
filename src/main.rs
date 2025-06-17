// Skeleton main.rs for ratatui dashboard
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    tty::IsTty,
};
use std::collections::HashMap;
use std::{process::Command, thread, time::Duration};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use chrono;
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

fn normalize_model_name(model: &str) -> String {
    let model_lower = model.to_lowercase();
    
    // Map Claude model names to cleaner versions
    if model_lower.contains("opus-4") {
        "Opus 4".to_string()
    } else if model_lower.contains("sonnet-4") {
        "Sonnet 4".to_string()
    } else if (model_lower.contains("claude-3.7") && model_lower.contains("sonnet")) 
              || model_lower.contains("3-7-sonnet") {
        "Sonnet 3.7".to_string()
    } else if model_lower.contains("claude-3-5-sonnet") || model_lower.contains("claude-3.5-sonnet") {
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
        eprintln!("Python parser failed:\n{}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    
    // Read the generated JSON file
    let text = fs::read_to_string("data.json").ok()?;
    
    match serde_json::from_str::<DailyModelReport>(&text) {
        Ok(data) => {
            Some(data)
        }
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
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(full);

        let summary_block = Block::default()
            .title(" Claude Token Dashboard ")
            .borders(Borders::ALL);

        let mut lifetime_cost = 0.0;
        let mut mtd_cost = 0.0;
        let today = Utc::now().naive_utc().date();
        let this_month = today.format("%Y-%m").to_string();

        for (date, model_map) in data.iter() {
            for (_model, stat) in model_map.iter() {
                lifetime_cost += stat.total_cost;
                if date.starts_with(&this_month) {
                    mtd_cost += stat.total_cost;
                }
            }
        }

        let summary_text = Paragraph::new(format!(
            "Lifetime: ${:.2}    MTD: ${:.2}",
            lifetime_cost, mtd_cost
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
    let mut dates: Vec<String> = data.keys().cloned().collect();
    dates.sort();
    
    if dates.is_empty() {
        return;
    }

    // Collect all unique models and sort them for consistent ordering
    let mut all_models: std::collections::HashSet<String> = std::collections::HashSet::new();
    for models_map in data.values() {
        for model in models_map.keys() {
            all_models.insert(model.clone());
        }
    }
    
    let mut sorted_models: Vec<String> = all_models.into_iter().collect();
    sorted_models.sort(); // Ensure consistent ordering
    
    // Create complete time series for each model (fill missing dates with 0)
    let chart_data: Vec<(String, Vec<(f64, f64)>)> = sorted_models
        .into_iter()
        .take(5) // Limit to 5 models for readability
        .map(|model| {
            let points: Vec<(f64, f64)> = dates
                .iter()
                .enumerate()
                .map(|(i, date)| {
                    let value = data.get(date)
                        .and_then(|models| models.get(&model))
                        .map(|stat| if is_tokens {
                            stat.total_tokens as f64
                        } else {
                            stat.total_cost
                        })
                        .unwrap_or(0.0);
                    (i as f64, value)
                })
                .collect();
            (model, points)
        })
        .collect();

    let colors = [Color::Cyan, Color::Yellow, Color::Green, Color::Red, Color::Blue];
    let datasets: Vec<Dataset> = chart_data
        .iter()
        .enumerate()
        .map(|(i, (model, points))| {
            Dataset::default()
                .name(normalize_model_name(model))
                .marker(symbols::Marker::Block)
                .style(Style::default().fg(colors[i % colors.len()]))
                .data(points)
        })
        .collect();

    // Calculate Y-axis bounds
    let mut max_val = 0.0;
    for (_, points) in &chart_data {
        for (_, val) in points {
            if *val > max_val {
                max_val = *val;
            }
        }
    }
    
    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(if is_tokens {
            "Tokens/day"
        } else {
            "Cost/day ($)"
        }))
        .x_axis(
            Axis::default()
                .title("Day")
                .bounds([0.0, (dates.len().max(1) - 1) as f64]),
        )
        .y_axis(Axis::default()
            .title(if is_tokens { "Tokens" } else { "$" })
            .bounds([0.0, max_val * 1.1]));

    f.render_widget(chart, area);
}

fn render_today_bar(f: &mut ratatui::Frame, area: Rect, data: &DailyModelReport, is_tokens: bool) {
    let today = Utc::now().naive_utc().date().format("%Y-%m-%d").to_string();
    let today_data = match data.get(&today) {
        Some(val) => val,
        None => {
            // Try to find the most recent date if today doesn't exist
            let mut dates: Vec<&String> = data.keys().collect();
            dates.sort();
            if let Some(recent_date) = dates.last() {
                if let Some(recent_data) = data.get(*recent_date) {
                    render_recent_bar(f, area, recent_data, recent_date, is_tokens);
                }
            }
            return;
        },
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
    let values: Vec<u64> = items.iter().map(|(_, v)| *v as u64).collect();

    let bar_data: Vec<(&str, u64)> = labels
        .iter()
        .zip(values.iter())
        .map(|(s, v)| (s.content.as_ref(), *v))
        .collect();
    let bars = BarChart::default()
        .block(Block::default().borders(Borders::ALL).title(if is_tokens {
            "Today: Tokens by Model"
        } else {
            "Today: Cost by Model"
        }))
        .bar_width(6)
        .bar_gap(1)
        .value_style(Style::default().fg(Color::Yellow))
        .label_style(Style::default().fg(Color::White))
        .bar_style(Style::default().bg(Color::Blue))
        .data(&bar_data);

    f.render_widget(bars, area);
}

fn render_recent_bar(f: &mut ratatui::Frame, area: Rect, data: &HashMap<String, ModelStats>, date: &str, is_tokens: bool) {
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
    let values: Vec<u64> = items.iter().map(|(_, v)| *v as u64).collect();

    let bar_data: Vec<(&str, u64)> = labels
        .iter()
        .zip(values.iter())
        .map(|(s, v)| (s.content.as_ref(), *v))
        .collect();
    let bars = BarChart::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            "{}: {} by Model", 
            date, 
            if is_tokens { "Tokens" } else { "Cost" }
        )))
        .bar_width(6)
        .bar_gap(1)
        .value_style(Style::default().fg(Color::Yellow))
        .label_style(Style::default().fg(Color::White))
        .bar_style(Style::default().bg(Color::Blue))
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
    let mut current_data: Option<DailyModelReport> = None;
    
    // Load data immediately on startup
    current_data = load_data();
    
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
                    .title(" Claude Token Dashboard - No Data ")
                    .borders(Borders::ALL);
                let paragraph = Paragraph::new("Unable to load data from parser.\nPress 'q' or Esc to quit")
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
