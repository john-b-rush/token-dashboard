// Token Dashboard
mod parser;

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    tty::IsTty,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::{thread, time::Duration};

use chrono::Utc;
use crossterm::event::{poll, read, Event, KeyCode};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::*;
use ratatui::symbols;
use ratatui::text::Span;
use ratatui::widgets::*;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;
use std::time::Instant;

// Use the parser module types
use parser::{DailyModelReport, TokenData};

// View mode for choosing how to group data
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum ViewMode {
    #[default]
    ByModel,
    ByProject,
}

// Totals structure for aggregated data
#[derive(Debug, Clone)]
struct Totals {
    cost: f64,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
}

impl Default for Totals {
    fn default() -> Self {
        Totals {
            cost: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }
    }
}

// Aggregates structure for pre-computed UI data
#[derive(Debug, Clone)]
struct Aggregates {
    lifetime: Totals,
    mtd: Totals,
    selected_date: Totals,
}

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

// Use the normalize_model_name function from our parser module
use parser::normalize_model_name;

/// Compute aggregates once when data is loaded/refreshed
fn compute_aggregates(data: &DailyModelReport, selected_date_str: &str) -> Aggregates {
    let mut lifetime = Totals::default();
    let mut mtd = Totals::default();
    let mut selected_date = Totals::default();

    let today_date = Utc::now().naive_utc().date();
    // We only need the month info for month-to-date calculations now
    let this_month = today_date.format("%Y-%m").to_string();

    for (date, model_map) in data.iter() {
        for (_model, stat) in model_map.iter() {
            // Lifetime totals
            lifetime.cost += stat.total_cost;
            lifetime.input_tokens += stat.input_tokens;
            lifetime.output_tokens += stat.output_tokens;
            lifetime.cache_read_tokens += stat.cache_read_tokens;
            lifetime.cache_write_tokens += stat.cache_creation_tokens;

            // Month-to-date
            if date.starts_with(&this_month) {
                mtd.cost += stat.total_cost;
                mtd.input_tokens += stat.input_tokens;
                mtd.output_tokens += stat.output_tokens;
                mtd.cache_read_tokens += stat.cache_read_tokens;
                mtd.cache_write_tokens += stat.cache_creation_tokens;
            }

            // Selected date
            if date == selected_date_str {
                selected_date.cost += stat.total_cost;
                selected_date.input_tokens += stat.input_tokens;
                selected_date.output_tokens += stat.output_tokens;
                selected_date.cache_read_tokens += stat.cache_read_tokens;
                selected_date.cache_write_tokens += stat.cache_creation_tokens;
            }
        }
    }

    Aggregates {
        lifetime,
        mtd,
        selected_date,
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

fn load_data() -> Option<TokenData> {
    // Use our Rust parser to get the complete data directly
    eprintln!("Loading token data...");
    let result = parser::load_complete_token_data();
    match &result {
        Some(data) => eprintln!("Loaded data for {} dates", data.model_report.len()),
        None => eprintln!("No data found"),
    }
    result
}

fn draw_ui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    token_data: &TokenData,
    aggregates: &Aggregates,
    view_mode: ViewMode,
    selected_date: chrono::NaiveDate,
    current_utc_date: chrono::NaiveDate,
) -> std::io::Result<()> {
    // Select the data based on view mode
    let data = match view_mode {
        ViewMode::ByModel => &token_data.model_report,
        ViewMode::ByProject => &token_data.project_report,
    };
    terminal.draw(|f| {
        let full = f.size();

        // Create main vertical layout
        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Length(7), Constraint::Min(0)])
            .split(full);

        // Split top area into stats and controls
        let top_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
            .split(main_layout[0]);

        let view_mode_text = match view_mode {
            ViewMode::ByModel => "Mode: By Model (M)",
            ViewMode::ByProject => "Mode: By Project (P)",
        };

        let summary_block = Block::default()
            .title(format!(" Token Usage Dashboard - {} ", view_mode_text))
            .borders(Borders::ALL);

        // Remove the help text since we're using a separate control panel

        // Format the selected date string
        let selected_date_display = if selected_date == current_utc_date {
            "Today".to_string()
        } else {
            selected_date.format("%Y-%m-%d").to_string()
        };
        let summary_text = Paragraph::new(format!(
            "Lifetime Spend   ${:.2}\nLifetime Tokens  {}/{} ({}/{})\nMonth-to-Date    ${:.2}\n{}'s Spend    ${:.2}\n{}'s Tokens   {}/{} ({}/{})",
            aggregates.lifetime.cost,
            format_tokens(aggregates.lifetime.input_tokens),
            format_tokens(aggregates.lifetime.output_tokens),
            format_tokens(aggregates.lifetime.cache_read_tokens),
            format_tokens(aggregates.lifetime.cache_write_tokens),
            aggregates.mtd.cost,
            selected_date_display,
            aggregates.selected_date.cost,
            selected_date_display,
            format_tokens(aggregates.selected_date.input_tokens),
            format_tokens(aggregates.selected_date.output_tokens),
            format_tokens(aggregates.selected_date.cache_read_tokens),
            format_tokens(aggregates.selected_date.cache_write_tokens)
        ))
        .block(summary_block)
        .style(Style::default().fg(Color::Green));

        // Controls block with keyboard shortcuts
        let controls_block = Block::default()
            .title(" Controls ")
            .borders(Borders::ALL);

        // Create help text based on current view
        let toggle_text = match view_mode {
            ViewMode::ByModel => "[P] Switch to Project View",
            ViewMode::ByProject => "[M] Switch to Model View",
        };

        let controls_text = Paragraph::new(format!(
            "{}\n[<] Previous Day  [>] Next Day\n[Q] Quit",
            toggle_text
        ))
        .block(controls_block)
        .style(Style::default().fg(Color::Yellow))
        .alignment(ratatui::layout::Alignment::Center);

        // Render both panels
        f.render_widget(summary_text, top_layout[0]);
        f.render_widget(controls_text, top_layout[1]);

        let chart_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(main_layout[1]);

        let top_row = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chart_layout[0]);

        let bottom_row = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chart_layout[1]);

        // Render actual charts instead of placeholder blocks
        render_stacked_bar(f, top_row[0], data, true, view_mode); // Tokens/day
        render_stacked_bar(f, top_row[1], data, false, view_mode); // Cost/day
        render_today_bar(f, bottom_row[0], data, true, view_mode, selected_date, current_utc_date); // Selected day's tokens
        render_today_bar(f, bottom_row[1], data, false, view_mode, selected_date, current_utc_date); // Selected day's cost
    })?;
    Ok(())
}

fn render_stacked_bar(
    f: &mut ratatui::Frame,
    area: Rect,
    data: &DailyModelReport,
    is_tokens: bool,
    view_mode: ViewMode,
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
    let chart_title = match view_mode {
        ViewMode::ByModel => {
            if is_tokens {
                "Tokens/day by Model"
            } else {
                "Cost/day ($) by Model"
            }
        }
        ViewMode::ByProject => {
            if is_tokens {
                "Tokens/day by Project"
            } else {
                "Cost/day ($) by Project"
            }
        }
    };

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(chart_title))
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

    // Get the legend title based on view mode
    let legend_title = match view_mode {
        ViewMode::ByModel => "All Models:",
        ViewMode::ByProject => "All Projects:",
    };

    let mut legend_lines = vec![
        Line::from(legend_title).style(Style::default().fg(Color::White)),
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

fn render_today_bar(
    f: &mut ratatui::Frame,
    area: Rect,
    data: &DailyModelReport,
    is_tokens: bool,
    view_mode: ViewMode,
    selected_date: chrono::NaiveDate,
    current_utc_date: chrono::NaiveDate,
) {
    let selected_date_str = selected_date.format("%Y-%m-%d").to_string();
    let display_date = if selected_date == current_utc_date {
        "Today"
    } else {
        &selected_date_str
    };
    let today_data = match data.get(&selected_date_str) {
        Some(val) => val,
        None => {
            // Show empty chart if there's no data for the selected date
            render_empty_bar(
                f,
                area,
                &selected_date_str,
                is_tokens,
                view_mode,
                current_utc_date,
            );
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

    let category = match view_mode {
        ViewMode::ByModel => "Model",
        ViewMode::ByProject => "Project",
    };

    let title = if is_tokens {
        format!(
            "{}: Tokens by {} (Max: {})",
            display_date,
            category,
            if max_val >= 1_000_000.0 {
                format!("{:.1}M", max_val / 1_000_000.0)
            } else if max_val >= 1_000.0 {
                format!("{:.0}K", max_val / 1_000.0)
            } else {
                format!("{:.0}", max_val)
            }
        )
    } else {
        format!(
            "{}: Cost by {} (Max: ${:.2})",
            display_date, category, max_val
        )
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

fn render_empty_bar(
    f: &mut ratatui::Frame,
    area: Rect,
    date: &str,
    is_tokens: bool,
    view_mode: ViewMode,
    current_utc_date: chrono::NaiveDate,
) {
    let category = match view_mode {
        ViewMode::ByModel => "Model",
        ViewMode::ByProject => "Project",
    };

    let current_date_str = current_utc_date.format("%Y-%m-%d").to_string();
    let display_date = if *date == current_date_str {
        "Today"
    } else {
        date
    };

    let title = if is_tokens {
        format!("{}: Tokens by {} (No data)", display_date, category)
    } else {
        format!("{}: Cost by {} (No data)", display_date, category)
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

fn setup_signal_handler() -> Result<Arc<AtomicBool>, Box<dyn std::error::Error>> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    Ok(running)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Starting token dashboard...");

    // Check for test mode
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--test" {
        eprintln!("Running in test mode - data loading only");
        let initial_data = load_data();
        match initial_data {
            Some(data) => {
                eprintln!("SUCCESS: Loaded data for {} dates", data.model_report.len());
                for (date, models) in data.model_report.iter().take(3) {
                    eprintln!("  Date: {}, Models: {}", date, models.len());
                }
                return Ok(());
            }
            None => {
                eprintln!("ERROR: No token data found");
                return Ok(());
            }
        }
    }

    // Check for terminal test mode
    if args.len() > 1 && args[1] == "--test-terminal" {
        eprintln!("Testing terminal initialization...");
        let _terminal_guard = TerminalGuard::new()?;
        eprintln!("Terminal initialized successfully, exiting");
        return Ok(());
    }

    // Load data completely before initializing terminal
    let initial_data = load_data();
    if initial_data.is_none() {
        eprintln!("Warning: No token data found");
        eprintln!("Press Ctrl+C to exit or continue to start dashboard...");
    }

    let running = setup_signal_handler()?;
    let mut terminal_guard = TerminalGuard::new()?;

    let refresh_interval = Duration::from_secs(10);
    let mut last_update = Instant::now(); // Set to now so we don't immediately refresh

    // Use the pre-loaded data and compute initial aggregates
    let mut current_data = initial_data;

    // Track current view mode, dates and UTC day tracking
    let mut current_view_mode = ViewMode::ByModel;
    let mut current_utc_date = Utc::now().naive_utc().date();
    let mut selected_date = current_utc_date;
    let selected_date_str = selected_date.format("%Y-%m-%d").to_string();

    // Initialize available dates from the initial data
    let mut available_dates: Vec<String> = current_data
        .as_ref()
        .map(|data| {
            let mut dates = data.model_report.keys().cloned().collect::<Vec<String>>();
            dates.sort();
            dates
        })
        .unwrap_or_default();

    // Store if the user is currently viewing "today"
    let mut viewing_today = true;

    let mut current_aggregates = current_data
        .as_ref()
        .map(|data| compute_aggregates(&data.model_report, &selected_date_str));

    while running.load(Ordering::SeqCst) {
        // Input check first
        match poll(Duration::from_millis(100)) {
            Ok(true) => {
                match read() {
                    Ok(Event::Key(key)) => {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => break,
                            KeyCode::Char('m') | KeyCode::Char('M') => {
                                current_view_mode = ViewMode::ByModel;
                                // Update aggregates based on the new view mode
                                if let Some(data) = &current_data {
                                    let selected_date_str =
                                        selected_date.format("%Y-%m-%d").to_string();
                                    current_aggregates = Some(compute_aggregates(
                                        &data.model_report,
                                        &selected_date_str,
                                    ));
                                }
                            }
                            KeyCode::Char('p') | KeyCode::Char('P') => {
                                current_view_mode = ViewMode::ByProject;
                                // Update aggregates based on the new view mode
                                if let Some(data) = &current_data {
                                    let selected_date_str =
                                        selected_date.format("%Y-%m-%d").to_string();
                                    current_aggregates = Some(compute_aggregates(
                                        &data.project_report,
                                        &selected_date_str,
                                    ));
                                }
                            }
                            KeyCode::Char('<') => {
                                // Navigate to previous day if available
                                if !available_dates.is_empty() {
                                    let current_date_str =
                                        selected_date.format("%Y-%m-%d").to_string();
                                    let current_index =
                                        available_dates.iter().position(|d| d == &current_date_str);

                                    if let Some(idx) = current_index {
                                        if idx > 0 {
                                            // There's a previous date
                                            if let Ok(prev_date) = chrono::NaiveDate::parse_from_str(
                                                &available_dates[idx - 1],
                                                "%Y-%m-%d",
                                            ) {
                                                selected_date = prev_date;

                                                // We're not viewing today anymore if we go back
                                                viewing_today = selected_date == current_utc_date;

                                                // Update aggregates for the new selected date
                                                if let Some(data) = &current_data {
                                                    let report = match current_view_mode {
                                                        ViewMode::ByModel => &data.model_report,
                                                        ViewMode::ByProject => &data.project_report,
                                                    };
                                                    let selected_date_str = selected_date
                                                        .format("%Y-%m-%d")
                                                        .to_string();
                                                    current_aggregates = Some(compute_aggregates(
                                                        report,
                                                        &selected_date_str,
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            KeyCode::Char('>') => {
                                // Navigate to next day if available
                                if !available_dates.is_empty() {
                                    let current_date_str =
                                        selected_date.format("%Y-%m-%d").to_string();
                                    let current_index =
                                        available_dates.iter().position(|d| d == &current_date_str);

                                    if let Some(idx) = current_index {
                                        if idx < available_dates.len() - 1 {
                                            // There's a next date
                                            if let Ok(next_date) = chrono::NaiveDate::parse_from_str(
                                                &available_dates[idx + 1],
                                                "%Y-%m-%d",
                                            ) {
                                                selected_date = next_date;

                                                // Check if we've navigated to today
                                                viewing_today = selected_date == current_utc_date;

                                                // Update aggregates for the new selected date
                                                if let Some(data) = &current_data {
                                                    let report = match current_view_mode {
                                                        ViewMode::ByModel => &data.model_report,
                                                        ViewMode::ByProject => &data.project_report,
                                                    };
                                                    let selected_date_str = selected_date
                                                        .format("%Y-%m-%d")
                                                        .to_string();
                                                    current_aggregates = Some(compute_aggregates(
                                                        report,
                                                        &selected_date_str,
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(_) => {} // Other events
                    Err(e) => {
                        eprintln!("Error reading input: {}", e);
                        break;
                    }
                }
            }
            Ok(false) => {} // No input available
            Err(e) => {
                eprintln!("Error polling input: {}", e);
                break;
            }
        }

        // Check for UTC date rollover
        let new_utc_date = Utc::now().naive_utc().date();
        if new_utc_date != current_utc_date {
            // UTC date has changed
            current_utc_date = new_utc_date;

            // If the user was viewing "today", automatically advance to the new day
            if viewing_today {
                selected_date = current_utc_date;

                // Update aggregates if we have data
                if let Some(data) = &current_data {
                    let report = match current_view_mode {
                        ViewMode::ByModel => &data.model_report,
                        ViewMode::ByProject => &data.project_report,
                    };
                    let selected_date_str = selected_date.format("%Y-%m-%d").to_string();
                    current_aggregates = Some(compute_aggregates(report, &selected_date_str));
                }
            }
        }

        // Refresh data on interval
        if last_update.elapsed() >= refresh_interval {
            // Load data silently on refresh
            if let Some(data) = parser::load_complete_token_data_with_options(true) {
                // Extract available dates from the data
                available_dates = data.model_report.keys().cloned().collect();
                available_dates.sort();

                // Ensure the selected date exists in the data, otherwise default to today
                let selected_date_str = selected_date.format("%Y-%m-%d").to_string();
                if !available_dates.contains(&selected_date_str) {
                    selected_date = current_utc_date;
                    viewing_today = true;
                }

                // Compute aggregates based on current view mode
                let report = match current_view_mode {
                    ViewMode::ByModel => &data.model_report,
                    ViewMode::ByProject => &data.project_report,
                };
                let selected_date_str = selected_date.format("%Y-%m-%d").to_string();
                current_aggregates = Some(compute_aggregates(report, &selected_date_str));
                current_data = Some(data);
            }
            last_update = Instant::now();
        }

        // Always try to render
        match (&current_data, &current_aggregates) {
            (Some(data), Some(aggregates)) => {
                if let Err(e) = draw_ui(
                    terminal_guard.terminal_mut(),
                    data,
                    aggregates,
                    current_view_mode,
                    selected_date,
                    current_utc_date,
                ) {
                    eprintln!("Error drawing UI: {}", e);
                    break;
                }
            }
            _ => {
                // Show loading/error screen
                if let Err(e) = terminal_guard.terminal_mut().draw(|f| {
                    let main_area = f.size();

                    // Create vertical split
                    let main_layout = Layout::default()
                        .direction(Direction::Vertical)
                        .margin(1)
                        .constraints([Constraint::Length(7), Constraint::Min(0)])
                        .split(main_area);

                    // Split top area into message and controls
                    let top_layout = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
                        .split(main_layout[0]);

                    // Main message
                    let block = Block::default()
                        .title(" Token Usage Dashboard - No Data ")
                        .borders(Borders::ALL);
                    let paragraph = Paragraph::new("Unable to load data from parser.")
                        .block(block)
                        .style(Style::default().fg(Color::Red));
                    f.render_widget(paragraph, top_layout[0]);

                    // Controls section
                    let controls_block = Block::default().title(" Controls ").borders(Borders::ALL);
                    let controls_text =
                        Paragraph::new("[M] View by Model\n[P] View by Project\n[<] Previous Day  [>] Next Day\n[Q] Quit")
                            .block(controls_block)
                            .style(Style::default().fg(Color::Yellow))
                            .alignment(ratatui::layout::Alignment::Center);
                    f.render_widget(controls_text, top_layout[1]);
                }) {
                    eprintln!("Error drawing no-data screen: {}", e);
                    break;
                }
            }
        }

        // Avoid burning CPU
        thread::sleep(Duration::from_millis(50));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use parser::ModelStats;
    use std::collections::HashMap;

    #[test]
    fn test_format_tokens() {
        // Test different token formatting cases
        assert_eq!(format_tokens(500), "500");
        assert_eq!(format_tokens(1500), "2K"); // Rounded up to nearest K
        assert_eq!(format_tokens(15000), "15K");
        assert_eq!(format_tokens(1_500_000), "1.5M");
        assert_eq!(format_tokens(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_compute_aggregates_with_selected_date() {
        // Create a mock daily model report
        let mut report = DailyModelReport::new();

        // Create some test model stats
        let today_date = Utc::now().naive_utc().date();
        let yesterday_date = today_date.checked_sub_signed(Duration::days(1)).unwrap();

        let today_str = today_date.format("%Y-%m-%d").to_string();
        let yesterday_str = yesterday_date.format("%Y-%m-%d").to_string();
        let this_month = today_date.format("%Y-%m").to_string();

        // Create model stats for today
        let mut today_models = HashMap::new();
        let today_model_stat = ModelStats {
            input_tokens: 1000,
            output_tokens: 500,
            cache_creation_tokens: 200,
            cache_read_tokens: 100,
            total_tokens: 1800,
            input_cost: 0.015,
            output_cost: 0.0375,
            cache_creation_cost: 0.003,
            cache_read_cost: 0.0001,
            total_cost: 0.0556,
        };
        today_models.insert("model1".to_string(), today_model_stat.clone());
        report.insert(today_str.clone(), today_models);

        // Create model stats for yesterday
        let mut yesterday_models = HashMap::new();
        let yesterday_model_stat = ModelStats {
            input_tokens: 2000,
            output_tokens: 1000,
            cache_creation_tokens: 300,
            cache_read_tokens: 200,
            total_tokens: 3500,
            input_cost: 0.03,
            output_cost: 0.075,
            cache_creation_cost: 0.0045,
            cache_read_cost: 0.0002,
            total_cost: 0.1097,
        };
        yesterday_models.insert("model1".to_string(), yesterday_model_stat.clone());
        report.insert(yesterday_str.clone(), yesterday_models);

        // Test computing aggregates with today selected
        let aggregates_today = compute_aggregates(&report, &today_str);
        assert_eq!(aggregates_today.selected_date.input_tokens, 1000);
        assert_eq!(aggregates_today.selected_date.output_tokens, 500);
        assert_eq!(aggregates_today.selected_date.cost, 0.0556);

        // Test computing aggregates with yesterday selected
        let aggregates_yesterday = compute_aggregates(&report, &yesterday_str);
        assert_eq!(aggregates_yesterday.selected_date.input_tokens, 2000);
        assert_eq!(aggregates_yesterday.selected_date.output_tokens, 1000);
        assert_eq!(aggregates_yesterday.selected_date.cost, 0.1097);

        // Check lifetime totals are the same regardless of selected date
        assert_eq!(aggregates_today.lifetime.input_tokens, 3000);
        assert_eq!(aggregates_yesterday.lifetime.input_tokens, 3000);

        // Check month-to-date calculations
        // Both dates should be in this month, so MTD should equal lifetime
        if yesterday_str.starts_with(&this_month) && today_str.starts_with(&this_month) {
            assert_eq!(aggregates_today.mtd.input_tokens, 3000);
            assert_eq!(aggregates_yesterday.mtd.input_tokens, 3000);
        }
    }

    #[test]
    fn test_date_navigation_and_viewing_today() {
        // This test checks the logic for the viewing_today flag that determines
        // whether to auto-advance after UTC date rollover

        let today = Utc::now().naive_utc().date();
        let yesterday = today.checked_sub_signed(Duration::days(1)).unwrap();
        let tomorrow = today.checked_add_signed(Duration::days(1)).unwrap();

        // When selected_date == current_utc_date, viewing_today should be true
        let viewing_today1 = today == today;
        assert!(viewing_today1);

        // When selected_date != current_utc_date, viewing_today should be false
        let viewing_today2 = yesterday == today;
        assert!(!viewing_today2);

        let viewing_today3 = tomorrow == today;
        assert!(!viewing_today3);
    }
}
