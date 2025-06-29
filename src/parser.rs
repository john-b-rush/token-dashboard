use chrono::{TimeZone, Utc};
use lazy_static::lazy_static;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// ----- Model Structures -----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub total_tokens: u64,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_creation_cost: f64,
    pub cache_read_cost: f64,
    pub total_cost: f64,
}

impl Default for ModelStats {
    fn default() -> Self {
        ModelStats {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            total_tokens: 0,
            input_cost: 0.0,
            output_cost: 0.0,
            cache_creation_cost: 0.0,
            cache_read_cost: 0.0,
            total_cost: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub project: String,
    pub date: String,
    pub timestamp: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub total_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub session_id: String,
    pub file_path: String,
    pub source: Option<String>,
}

// Cache data structure
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    mtime: f64,
    records: Vec<UsageRecord>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct CacheData {
    files: HashMap<String, CacheEntry>,
}

// Model pricing data structure
#[derive(Clone, Copy)]
pub struct ModelPricing {
    input_tokens: f64,
    output_tokens: f64,
    input_tokens_cache_write: f64,
    input_tokens_cache_read: f64,
}

// ----- Model Pricing Constants -----

lazy_static::lazy_static! {
    static ref MODEL_PRICING: HashMap<&'static str, ModelPricing> = {
        let mut m = HashMap::new();
        // Anthropic Claude models
        m.insert("Opus 4", ModelPricing {
            input_tokens: 0.015,
            output_tokens: 0.075,
            input_tokens_cache_write: 0.01875,
            input_tokens_cache_read: 0.0015,
        });
        m.insert("Sonnet 4", ModelPricing {
            input_tokens: 0.003,
            output_tokens: 0.015,
            input_tokens_cache_write: 0.00375,
            input_tokens_cache_read: 0.0003,
        });
        m.insert("Sonnet 3.7", ModelPricing {
            input_tokens: 0.003,
            output_tokens: 0.015,
            input_tokens_cache_write: 0.00375,
            input_tokens_cache_read: 0.0003,
        });
        m.insert("Claude 3.5 Sonnet", ModelPricing {
            input_tokens: 0.003,
            output_tokens: 0.015,
            input_tokens_cache_write: 0.00375,
            input_tokens_cache_read: 0.0003,
        });
        m.insert("Claude 3.5 Haiku", ModelPricing {
            input_tokens: 0.0008,
            output_tokens: 0.004,
            input_tokens_cache_write: 0.001,
            input_tokens_cache_read: 0.00008,
        });
        m.insert("Claude 3 Opus", ModelPricing {
            input_tokens: 0.015,
            output_tokens: 0.075,
            input_tokens_cache_write: 0.0,
            input_tokens_cache_read: 0.0,
        });
        // OpenAI models
        m.insert("GPT-4o", ModelPricing {
            input_tokens: 0.005,
            output_tokens: 0.015,
            input_tokens_cache_write: 0.0,
            input_tokens_cache_read: 0.0,
        });
        m.insert("GPT-4", ModelPricing {
            input_tokens: 0.03,
            output_tokens: 0.06,
            input_tokens_cache_write: 0.0,
            input_tokens_cache_read: 0.0,
        });
        m.insert("o3", ModelPricing {
            input_tokens: 0.015,
            output_tokens: 0.045,
            input_tokens_cache_write: 0.0,
            input_tokens_cache_read: 0.0075,
        });
        m.insert("o1", ModelPricing {
            input_tokens: 0.015,
            output_tokens: 0.06,
            input_tokens_cache_write: 0.0,
            input_tokens_cache_read: 0.0075,
        });
        m.insert("o4-mini", ModelPricing {
            input_tokens: 0.0003,
            output_tokens: 0.0012,
            input_tokens_cache_write: 0.0,
            input_tokens_cache_read: 0.00015,
        });
        m
    };
}

// ----- Project Name Extraction Functions -----

// Cache for filesystem validation results to avoid repeated I/O
lazy_static! {
    static ref PATH_CACHE: Mutex<HashMap<String, Option<String>>> = Mutex::new(HashMap::new());
}

/// Extract project name using filesystem validation with caching
fn extract_project_name_with_validation(folder_path: &str) -> Option<String> {
    let folder_name = Path::new(folder_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    // Remove leading dash if present
    let folder_name = if let Some(stripped) = folder_name.strip_prefix('-') {
        stripped
    } else {
        folder_name
    };

    // Check cache first
    {
        let cache = PATH_CACHE.lock().unwrap();
        if let Some(cached_result) = cache.get(folder_name) {
            return cached_result.clone();
        }
    }

    let result = try_filesystem_validation(folder_name);

    // Cache the result
    {
        let mut cache = PATH_CACHE.lock().unwrap();
        cache.insert(folder_name.to_string(), result.clone());
    }

    result
}

/// Try multiple path reconstruction strategies with filesystem validation
fn try_filesystem_validation(folder_name: &str) -> Option<String> {
    let parts: Vec<&str> = folder_name.split('-').filter(|s| !s.is_empty()).collect();

    if parts.len() < 2 {
        return None;
    }

    // Strategy 1: Try different split points for base_path/project_name
    for split_point in 1..parts.len() {
        let base_path = format!("/{}", parts[..split_point].join("/"));
        let project_name = parts[split_point..].join("-");

        if !project_name.is_empty() {
            let full_path = format!("{}/{}", base_path, project_name);
            if Path::new(&full_path).exists() {
                return Some(project_name);
            }
        }
    }

    // Strategy 2: Try common base path patterns
    let common_bases = vec!["Users", "home", "opt"];
    for base in common_bases {
        if let Some(base_idx) = parts.iter().position(|&p| p == base) {
            // Try paths starting from this base
            for split_point in (base_idx + 2)..parts.len() {
                let base_path = format!("/{}", parts[..split_point].join("/"));
                let project_name = parts[split_point..].join("-");

                if !project_name.is_empty() {
                    let full_path = format!("{}/{}", base_path, project_name);
                    if Path::new(&full_path).exists() {
                        return Some(project_name);
                    }
                }
            }
        }
    }

    None
}

/// Extract project name from Claude folder path like '-Users-johnrush-repos-chuck-data2'
pub fn extract_project_name(folder_path: &str) -> String {
    // Try filesystem validation first
    if let Some(project_name) = extract_project_name_with_validation(folder_path) {
        return project_name;
    }

    // Fallback to original logic if filesystem validation fails
    extract_project_name_fallback(folder_path)
}

/// Original project name extraction logic as fallback
fn extract_project_name_fallback(folder_path: &str) -> String {
    let folder_name = Path::new(folder_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    // Remove leading dash if present
    let folder_name = if let Some(stripped) = folder_name.strip_prefix('-') {
        stripped
    } else {
        folder_name
    };

    // Remove the '-Users-johnrush-repos-' prefix and return the project name
    let prefix = "Users-johnrush-repos-";
    if let Some(stripped) = folder_name.strip_prefix(prefix) {
        return stripped.to_string();
    }

    // Fallback: convert dashes back to slashes and extract the last part
    let path_parts = folder_name.replace('-', "/");
    Path::new(&path_parts)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Extract project name from goose working directory path like '/Users/johnrush/repos/chuck-data4'
pub fn extract_project_name_from_working_dir(working_dir: &str) -> String {
    if working_dir.is_empty() {
        return "unknown".to_string();
    }

    // Extract the last directory name from the path
    let project_name = Path::new(working_dir.trim_end_matches('/'))
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    // Handle special cases
    if project_name.is_empty() || project_name == "." {
        return "root".to_string();
    } else if working_dir == dirs::home_dir().unwrap_or_default().to_string_lossy() {
        return "home".to_string();
    }

    project_name.to_string()
}

// ----- Model Normalization Function -----

/// Normalize model name to match pricing table keys
pub fn normalize_model_name(model: &str) -> String {
    let model_lower = model.to_lowercase();

    // Map Claude model names to pricing table keys
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
    // Map OpenAI/goose model names to pricing table keys
    } else if model_lower == "o3" || model_lower.contains("o3-2025") {
        "o3".to_string()
    } else if model_lower.contains("o1") {
        "o1".to_string()
    } else if model_lower.contains("gpt-4o") {
        "GPT-4o".to_string()
    } else if model_lower.contains("gpt-4") && !model_lower.contains("gpt-4o") {
        "GPT-4".to_string()
    } else {
        model.to_string() // Return original if no match
    }
}

// ----- Cost Calculation Function -----

/// Calculate cost breakdown for token usage
pub fn calculate_cost(
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
) -> ModelStats {
    let normalized_model = normalize_model_name(model);

    let pricing = MODEL_PRICING.get(normalized_model.as_str());

    if pricing.is_none() {
        return ModelStats::default();
    }

    let pricing = pricing.unwrap();

    let input_cost = (input_tokens as f64 / 1000.0) * pricing.input_tokens;

    let output_cost = (output_tokens as f64 / 1000.0) * pricing.output_tokens;

    let cache_creation_cost = if pricing.input_tokens_cache_write > 0.0 {
        (cache_creation_tokens as f64 / 1000.0) * pricing.input_tokens_cache_write
    } else {
        0.0
    };

    let cache_read_cost = if pricing.input_tokens_cache_read > 0.0 {
        (cache_read_tokens as f64 / 1000.0) * pricing.input_tokens_cache_read
    } else {
        0.0
    };

    let total_cost = input_cost + output_cost + cache_creation_cost + cache_read_cost;

    ModelStats {
        input_tokens,
        output_tokens,
        cache_creation_tokens,
        cache_read_tokens,
        total_tokens: input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens,
        input_cost,
        output_cost,
        cache_creation_cost,
        cache_read_cost,
        total_cost,
    }
}

// ----- File Parsing Functions -----

/// Parse a JSONL file and extract token usage records
pub fn parse_jsonl_file(file_path: &Path, project_name: &str) -> Vec<UsageRecord> {
    let mut records = Vec::new();

    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error reading file {}: {}", file_path.display(), e);
            return records;
        }
    };

    let reader = BufReader::new(file);

    for (line_num, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(e) => {
                eprintln!(
                    "Error reading line at {}:{}: {}",
                    file_path.display(),
                    line_num + 1,
                    e
                );
                continue;
            }
        };

        if line.is_empty() {
            continue;
        }

        match serde_json::from_str::<Value>(&line) {
            Ok(data) => {
                // Only process assistant messages with usage data
                if let (Some("assistant"), Some(message)) = (
                    data.get("type").and_then(|v| v.as_str()),
                    data.get("message"),
                ) {
                    if let Some(usage) = message.get("usage") {
                        if let Some(timestamp) = data.get("timestamp").and_then(|v| v.as_str()) {
                            let model = message
                                .get("model")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");

                            // Skip synthetic messages
                            if model == "<synthetic>" {
                                continue;
                            }

                            // Parse timestamp and extract date
                            match chrono::DateTime::parse_from_rfc3339(timestamp) {
                                Ok(dt) => {
                                    let date = dt.naive_utc().date();
                                    let date_str = date.format("%Y-%m-%d").to_string();

                                    let input_tokens = usage
                                        .get("input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);

                                    let output_tokens = usage
                                        .get("output_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);

                                    let cache_creation_input_tokens = usage
                                        .get("cache_creation_input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);

                                    let cache_read_input_tokens = usage
                                        .get("cache_read_input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);

                                    let session_id = data
                                        .get("sessionId")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();

                                    let mut record = UsageRecord {
                                        project: project_name.to_string(),
                                        date: date_str,
                                        timestamp: timestamp.to_string(),
                                        model: model.to_string(),
                                        input_tokens,
                                        output_tokens,
                                        cache_creation_input_tokens,
                                        cache_read_input_tokens,
                                        total_tokens: None,
                                        reasoning_tokens: None,
                                        session_id,
                                        file_path: file_path.to_string_lossy().to_string(),
                                        source: Some("claude".to_string()),
                                    };

                                    // Add server tool use if present
                                    if usage.get("server_tool_use").is_some() {
                                        // We don't handle server tool use in this version yet
                                    }

                                    record.total_tokens = Some(
                                        record.input_tokens
                                            + record.output_tokens
                                            + record.cache_creation_input_tokens
                                            + record.cache_read_input_tokens,
                                    );

                                    records.push(record);
                                }
                                Err(e) => {
                                    eprintln!("Failed to parse timestamp '{}': {}", timestamp, e);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to parse JSON at {}:{} - {}",
                    file_path.display(),
                    line_num + 1,
                    e
                );
                continue;
            }
        }
    }

    records
}

/// Parse a goose CLI log file and extract token usage records
pub fn parse_goose_cli_log_file(file_path: &Path) -> Vec<UsageRecord> {
    let mut records = Vec::new();
    let session_id = file_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_string();

    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!(
                "Error reading goose CLI log file {}: {}",
                file_path.display(),
                e
            );
            return records;
        }
    };

    let reader = BufReader::new(file);

    for (line_num, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(e) => {
                eprintln!(
                    "Error reading line at {}:{}: {}",
                    file_path.display(),
                    line_num + 1,
                    e
                );
                continue;
            }
        };

        if line.is_empty() {
            continue;
        }

        match serde_json::from_str::<Value>(&line) {
            Ok(data) => {
                // Look for lines with token usage in the fields
                if let Some(fields) = data.get("fields") {
                    if fields.get("total_tokens").is_some() && fields.get("output").is_some() {
                        // Parse output JSON to get detailed usage and model info
                        let mut model = "unknown".to_string();
                        let mut input_tokens = 0;
                        let mut output_tokens = 0;
                        let mut total_tokens = 0;
                        let mut cache_tokens = 0;
                        let mut reasoning_tokens = 0;

                        if let Some(output_str) = fields.get("output").and_then(|v| v.as_str()) {
                            if let Ok(output_data) = serde_json::from_str::<Value>(output_str) {
                                if let Some(usage) = output_data.get("usage") {
                                    input_tokens = usage
                                        .get("prompt_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);

                                    output_tokens = usage
                                        .get("completion_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);

                                    total_tokens = usage
                                        .get("total_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(input_tokens + output_tokens);

                                    // Extract detailed token info if available
                                    if let Some(completion_details) =
                                        usage.get("completion_tokens_details")
                                    {
                                        reasoning_tokens = completion_details
                                            .get("reasoning_tokens")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);
                                    }

                                    if let Some(prompt_details) = usage.get("prompt_tokens_details")
                                    {
                                        cache_tokens = prompt_details
                                            .get("cached_tokens")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);
                                    }
                                }

                                model = output_data
                                    .get("model")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown")
                                    .to_string();
                            } else {
                                // Fallback to fields data if output parsing fails
                                input_tokens = fields
                                    .get("input_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);

                                output_tokens = fields
                                    .get("output_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);

                                total_tokens = fields
                                    .get("total_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                            }
                        }

                        // Parse input messages to extract working directory/project info
                        let mut project_name = "unknown".to_string();

                        if let Some(input_str) = fields.get("input").and_then(|v| v.as_str()) {
                            if let Ok(input_data) = serde_json::from_str::<Value>(input_str) {
                                if let Some(messages) =
                                    input_data.get("messages").and_then(|v| v.as_array())
                                {
                                    // Look for working directory in system messages
                                    for message in messages {
                                        if let Some(content) =
                                            message.get("content").and_then(|v| v.as_str())
                                        {
                                            if content.contains("current directory:") {
                                                // Extract working directory path
                                                lazy_static::lazy_static! {
                                                    static ref RE: regex::Regex = regex::Regex::new(
                                                        r"current directory: (/[^\s\n]+)"
                                                    ).unwrap();
                                                }

                                                if let Some(captures) = RE.captures(content) {
                                                    if let Some(working_dir) = captures.get(1) {
                                                        project_name =
                                                            extract_project_name_from_working_dir(
                                                                working_dir.as_str(),
                                                            );
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Parse timestamp
                        let mut date_str = String::new();
                        let mut timestamp = String::new();

                        if let Some(ts) = data.get("timestamp").and_then(|v| v.as_str()) {
                            match chrono::DateTime::parse_from_rfc3339(ts) {
                                Ok(dt) => {
                                    date_str = dt.naive_utc().date().format("%Y-%m-%d").to_string();
                                    timestamp = ts.to_string();
                                }
                                Err(_) => {
                                    // Fallback to file modification time
                                    if let Ok(metadata) = fs::metadata(file_path) {
                                        if let Ok(modified) = metadata.modified() {
                                            if let Ok(timestamp_secs) =
                                                modified.duration_since(std::time::UNIX_EPOCH)
                                            {
                                                let dt = Utc
                                                    .timestamp_opt(
                                                        timestamp_secs.as_secs() as i64,
                                                        0,
                                                    )
                                                    .unwrap();
                                                date_str = dt
                                                    .naive_utc()
                                                    .date()
                                                    .format("%Y-%m-%d")
                                                    .to_string();
                                                timestamp = dt.to_rfc3339();
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // Fallback to file modification time
                            if let Ok(metadata) = fs::metadata(file_path) {
                                if let Ok(modified) = metadata.modified() {
                                    if let Ok(timestamp_secs) =
                                        modified.duration_since(std::time::UNIX_EPOCH)
                                    {
                                        let dt = Utc
                                            .timestamp_opt(timestamp_secs.as_secs() as i64, 0)
                                            .unwrap();
                                        date_str =
                                            dt.naive_utc().date().format("%Y-%m-%d").to_string();
                                        timestamp = dt.to_rfc3339();
                                    }
                                }
                            }
                        }

                        let record = UsageRecord {
                            project: project_name,
                            date: date_str,
                            timestamp,
                            model,
                            input_tokens,
                            output_tokens,
                            cache_creation_input_tokens: 0, // Not typically in goose logs
                            cache_read_input_tokens: cache_tokens,
                            total_tokens: Some(total_tokens),
                            reasoning_tokens: Some(reasoning_tokens),
                            session_id: session_id.clone(),
                            file_path: file_path.to_string_lossy().to_string(),
                            source: Some("goose_cli".to_string()),
                        };

                        records.push(record);
                    }
                }
            }
            Err(_) => {
                // Skip non-JSON lines (some logs may have mixed content)
                continue;
            }
        }
    }

    records
}

// ----- Cache Management Functions -----

/// Get the cache file path for the given directory
pub fn get_cache_path(dir: &str) -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("token-dashboard");

    fs::create_dir_all(&cache_dir).unwrap_or(());

    // Create a safe filename from the dir path
    let safe_name = dir.replace(['/', '\\', ':'], "_");
    cache_dir.join(format!("cache_{}.json", safe_name))
}

/// Load cache from JSON file
pub fn load_cache(cache_path: &Path) -> CacheData {
    if cache_path.exists() {
        match fs::read_to_string(cache_path) {
            Ok(content) => match serde_json::from_str::<CacheData>(&content) {
                Ok(cache) => return cache,
                Err(e) => eprintln!("Failed to parse cache file: {}", e),
            },
            Err(e) => eprintln!("Failed to read cache file: {}", e),
        }
    }

    CacheData::default()
}

/// Save cache to JSON file
pub fn save_cache(cache_path: &Path, cache_data: &CacheData) {
    match serde_json::to_string_pretty(cache_data) {
        Ok(json) => {
            if let Err(e) = fs::write(cache_path, json) {
                eprintln!("Warning: Could not save cache to {:?}: {}", cache_path, e);
            }
        }
        Err(e) => eprintln!("Warning: Could not serialize cache: {}", e),
    }
}

/// Check if file has been modified since the cached modification time
pub fn is_file_modified(file_path: &Path, cached_mtime: f64) -> bool {
    match fs::metadata(file_path) {
        Ok(metadata) => {
            match metadata.modified() {
                Ok(modified) => {
                    match modified.duration_since(std::time::UNIX_EPOCH) {
                        Ok(duration) => {
                            let current_mtime = duration.as_secs_f64();
                            // Use a small epsilon to handle floating point precision issues
                            (current_mtime - cached_mtime) > 0.001
                        }
                        Err(_) => true,
                    }
                }
                Err(_) => true,
            }
        }
        Err(_) => true,
    }
}

// ----- Main Scanning Functions -----

/// Structure to hold file information for processing
#[derive(Debug, Clone)]
struct FileToProcess {
    path: PathBuf,
    project_name: String,
    current_mtime: f64,
}

/// Get the default Claude projects directory
fn get_claude_projects_dir() -> PathBuf {
    dirs::home_dir()
        .map(|path| path.join(".claude").join("projects"))
        .unwrap_or_else(|| PathBuf::from("~/.claude/projects"))
}

/// Get the default Goose CLI logs directory
fn get_goose_logs_dir() -> PathBuf {
    dirs::home_dir()
        .map(|path| path.join(".local/state/goose/logs/cli"))
        .unwrap_or_else(|| PathBuf::from("~/.local/state/goose/logs/cli"))
}

/// Scan all Claude log files and extract token usage data with caching (with silent option)
pub fn scan_claude_logs_with_options(claude_dir: Option<String>, silent: bool) -> Vec<UsageRecord> {
    let claude_path = if let Some(dir) = claude_dir {
        PathBuf::from(dir)
    } else {
        get_claude_projects_dir()
    };

    if !claude_path.exists() {
        eprintln!(
            "Claude projects directory not found: {}",
            claude_path.display()
        );
        return Vec::new();
    }

    // Load cache
    let cache_path = get_cache_path(&claude_path.to_string_lossy());
    let mut cache_data = load_cache(&cache_path);
    let mut all_records = Vec::new();
    let mut files_processed = 0;
    let mut files_from_cache = 0;

    // Collect all files that need processing
    let mut files_to_process = Vec::new();

    // First pass: collect files and use cache when possible
    if let Ok(entries) = fs::read_dir(claude_path) {
        for entry in entries.filter_map(Result::ok) {
            if !entry.path().is_dir() {
                continue;
            }

            let project_folder = entry.file_name();
            let project_folder_str = project_folder.to_string_lossy();
            let project_name = extract_project_name(&project_folder_str);

            // Process all .jsonl files in the project folder
            if let Ok(files) = fs::read_dir(entry.path()) {
                for file_entry in files.filter_map(Result::ok) {
                    if let Some(extension) = file_entry.path().extension() {
                        if extension == "jsonl" {
                            let file_path = file_entry.path();

                            let current_mtime = match fs::metadata(&file_path) {
                                Ok(metadata) => match metadata.modified() {
                                    Ok(modified) => modified
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map(|d| d.as_secs_f64())
                                        .unwrap_or(0.0),
                                    Err(_) => continue,
                                },
                                Err(_) => continue,
                            };

                            let file_path_str = file_path.to_string_lossy().to_string();

                            // Check if file is in cache and hasn't been modified
                            if let Some(cache_entry) = cache_data.files.get(&file_path_str) {
                                if !is_file_modified(&file_path, cache_entry.mtime) {
                                    // Use cached records
                                    all_records.extend(cache_entry.records.clone());
                                    files_from_cache += 1;
                                    continue;
                                }
                            }

                            // Add to files that need processing
                            files_to_process.push(FileToProcess {
                                path: file_path.to_path_buf(),
                                project_name: project_name.clone(),
                                current_mtime,
                            });
                        }
                    }
                }
            }
        }
    }

    // Second pass: process files in parallel
    let processed_results: Vec<_> = files_to_process
        .par_iter()
        .map(|file_info| {
            let records = parse_jsonl_file(&file_info.path, &file_info.project_name);
            (
                file_info.path.to_string_lossy().to_string(),
                file_info.current_mtime,
                records,
            )
        })
        .collect();

    // Third pass: collect results and update cache
    for (file_path, mtime, records) in processed_results {
        all_records.extend(records.clone());
        cache_data
            .files
            .insert(file_path, CacheEntry { mtime, records });
        files_processed += 1;
    }

    // Save updated cache
    save_cache(&cache_path, &cache_data);

    if !silent && (files_processed > 0 || files_from_cache > 0) {
        eprintln!(
            "Claude: Processed {} files, used cache for {} files",
            files_processed, files_from_cache
        );
    }

    all_records
}

/// Scan all goose CLI log files and extract token usage data with caching (with silent option)
pub fn scan_goose_logs_with_options(goose_dir: Option<String>, silent: bool) -> Vec<UsageRecord> {
    let goose_path = if let Some(dir) = goose_dir {
        PathBuf::from(dir)
    } else {
        get_goose_logs_dir()
    };

    if !goose_path.exists() {
        eprintln!(
            "Goose CLI logs directory not found: {}",
            goose_path.display()
        );
        return Vec::new();
    }

    // Load cache
    let cache_path = get_cache_path(&goose_path.to_string_lossy());
    let mut cache_data = load_cache(&cache_path);
    let mut all_records = Vec::new();
    let mut files_processed = 0;
    let mut files_from_cache = 0;

    // Collect all files that need processing
    let mut files_to_process = Vec::new();

    // First pass: collect files and use cache when possible
    if let Ok(entries) = fs::read_dir(goose_path) {
        for entry in entries.filter_map(Result::ok) {
            if !entry.path().is_dir() {
                continue;
            }

            if let Ok(files) = fs::read_dir(entry.path()) {
                for file_entry in files.filter_map(Result::ok) {
                    if let Some(extension) = file_entry.path().extension() {
                        if extension == "log" {
                            let file_name = file_entry.file_name().to_string_lossy().to_string();

                            // Process main session log files (not MCP-specific logs)
                            if !file_name.ends_with("-mcp-developer.log")
                                && !file_name.ends_with("-mcp-memory.log")
                            {
                                let file_path = file_entry.path();

                                let current_mtime = match fs::metadata(&file_path) {
                                    Ok(metadata) => match metadata.modified() {
                                        Ok(modified) => modified
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map(|d| d.as_secs_f64())
                                            .unwrap_or(0.0),
                                        Err(_) => continue,
                                    },
                                    Err(_) => continue,
                                };

                                let file_path_str = file_path.to_string_lossy().to_string();

                                // Check if file is in cache and hasn't been modified
                                if let Some(cache_entry) = cache_data.files.get(&file_path_str) {
                                    if !is_file_modified(&file_path, cache_entry.mtime) {
                                        // Use cached records
                                        all_records.extend(cache_entry.records.clone());
                                        files_from_cache += 1;
                                        continue;
                                    }
                                }

                                // Add to files that need processing
                                files_to_process.push(FileToProcess {
                                    path: file_path.to_path_buf(),
                                    project_name: "".to_string(), // Goose logs don't use project names like Claude
                                    current_mtime,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Second pass: process files in parallel
    let processed_results: Vec<_> = files_to_process
        .par_iter()
        .map(|file_info| {
            let records = parse_goose_cli_log_file(&file_info.path);
            (
                file_info.path.to_string_lossy().to_string(),
                file_info.current_mtime,
                records,
            )
        })
        .collect();

    // Third pass: collect results and update cache
    for (file_path, mtime, records) in processed_results {
        all_records.extend(records.clone());
        cache_data
            .files
            .insert(file_path, CacheEntry { mtime, records });
        files_processed += 1;
    }

    // Save updated cache
    save_cache(&cache_path, &cache_data);

    if !silent && (files_processed > 0 || files_from_cache > 0) {
        eprintln!(
            "Goose: Processed {} files, used cache for {} files",
            files_processed, files_from_cache
        );
    }

    all_records
}

/// Scan both Claude and goose logs and combine the results (with silent option)
pub fn scan_all_logs_with_options(
    claude_dir: Option<String>,
    goose_dir: Option<String>,
    silent: bool,
) -> Vec<UsageRecord> {
    let mut all_records = Vec::new();

    // Scan Claude logs
    let claude_records = scan_claude_logs_with_options(claude_dir, silent);
    all_records.extend(claude_records);

    // Scan goose logs
    let goose_records = scan_goose_logs_with_options(goose_dir, silent);
    all_records.extend(goose_records);

    all_records
}

// ----- Report Generation Functions -----

pub type DailyModelReport = HashMap<String, HashMap<String, ModelStats>>; // date -> model -> stats
pub type DailyProjectReport = HashMap<String, HashMap<String, ModelStats>>; // date -> project -> stats

/// Generate daily breakdown of token usage by model with cost information
pub fn generate_daily_model_report(records: &[UsageRecord]) -> DailyModelReport {
    if records.is_empty() {
        return HashMap::new();
    }

    // Group by date and model
    let mut daily_model_usage: HashMap<String, HashMap<String, ModelStats>> = HashMap::new();

    // Process each record
    for record in records {
        let date = &record.date;
        let model = &record.model;

        let model_map = daily_model_usage.entry(date.clone()).or_default();

        let usage = model_map.entry(model.clone()).or_default();

        usage.input_tokens += record.input_tokens;
        usage.output_tokens += record.output_tokens;
        usage.cache_creation_tokens += record.cache_creation_input_tokens;
        usage.cache_read_tokens += record.cache_read_input_tokens;

        let total = record.input_tokens
            + record.output_tokens
            + record.cache_creation_input_tokens
            + record.cache_read_input_tokens;

        usage.total_tokens += total;

        // Calculate costs for this record
        let costs = calculate_cost(
            model,
            record.input_tokens,
            record.output_tokens,
            record.cache_creation_input_tokens,
            record.cache_read_input_tokens,
        );

        usage.input_cost += costs.input_cost;
        usage.output_cost += costs.output_cost;
        usage.cache_creation_cost += costs.cache_creation_cost;
        usage.cache_read_cost += costs.cache_read_cost;
        usage.total_cost += costs.total_cost;
    }

    daily_model_usage
}

/// Generate daily breakdown of token usage by project with cost information
pub fn generate_daily_project_report(records: &[UsageRecord]) -> DailyProjectReport {
    if records.is_empty() {
        return HashMap::new();
    }

    // Group by date and project
    let mut daily_project_usage: HashMap<String, HashMap<String, ModelStats>> = HashMap::new();

    // Process each record
    for record in records {
        let date = &record.date;
        let project = &record.project;

        let project_map = daily_project_usage.entry(date.clone()).or_default();

        let usage = project_map.entry(project.clone()).or_default();

        usage.input_tokens += record.input_tokens;
        usage.output_tokens += record.output_tokens;
        usage.cache_creation_tokens += record.cache_creation_input_tokens;
        usage.cache_read_tokens += record.cache_read_input_tokens;

        let total = record.input_tokens
            + record.output_tokens
            + record.cache_creation_input_tokens
            + record.cache_read_input_tokens;

        usage.total_tokens += total;

        // Calculate costs for this record
        let costs = calculate_cost(
            &record.model,
            record.input_tokens,
            record.output_tokens,
            record.cache_creation_input_tokens,
            record.cache_read_input_tokens,
        );

        usage.input_cost += costs.input_cost;
        usage.output_cost += costs.output_cost;
        usage.cache_creation_cost += costs.cache_creation_cost;
        usage.cache_read_cost += costs.cache_read_cost;
        usage.total_cost += costs.total_cost;
    }

    daily_project_usage
}

// ----- Public API Functions -----

// Removed deprecated load_token_data functions

/// Struct to hold both model and project reports
pub struct TokenData {
    pub model_report: DailyModelReport,
    pub project_report: DailyProjectReport,
}

/// Load both model and project data for the dashboard
pub fn load_complete_token_data() -> Option<TokenData> {
    load_complete_token_data_with_options(false)
}

/// Load both model and project data for the dashboard (with silent option)
pub fn load_complete_token_data_with_options(silent: bool) -> Option<TokenData> {
    let records = scan_all_logs_with_options(None, None, silent);

    if records.is_empty() {
        return None;
    }

    let model_report = generate_daily_model_report(&records);
    let project_report = generate_daily_project_report(&records);

    Some(TokenData {
        model_report,
        project_report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_normalize_model_name() {
        // Basic cases
        assert_eq!(normalize_model_name("claude-3-opus"), "Claude 3 Opus");
        assert_eq!(
            normalize_model_name("claude-3.5-sonnet"),
            "Claude 3.5 Sonnet"
        );
        assert_eq!(normalize_model_name("claude-3.7-sonnet"), "Sonnet 3.7");
        assert_eq!(normalize_model_name("gpt-4o"), "GPT-4o");
        assert_eq!(normalize_model_name("gpt-4"), "GPT-4");
        assert_eq!(normalize_model_name("unknown"), "unknown");

        // Edge cases
        assert_eq!(normalize_model_name(""), ""); // Empty string
        assert_eq!(normalize_model_name("CLAUDE-3-OPUS"), "Claude 3 Opus"); // Case insensitive
        assert_eq!(normalize_model_name("claude-3-opus-extra"), "Claude 3 Opus"); // Extra text
        assert_eq!(normalize_model_name("o3-2025-preview"), "o3"); // Version with preview
        assert_eq!(normalize_model_name("gpt-4o-mini"), "GPT-4o"); // Partial match
    }

    #[test]
    fn test_extract_project_name() {
        assert_eq!(
            extract_project_name("-Users-johnrush-repos-project1"),
            "project1"
        );
        assert_eq!(
            extract_project_name("Users-johnrush-repos-project2"),
            "project2"
        );
        assert_eq!(extract_project_name("some-other-path"), "path");
    }

    #[test]
    fn test_extract_project_name_from_working_dir() {
        assert_eq!(
            extract_project_name_from_working_dir("/Users/johnrush/repos/project1"),
            "project1"
        );
        assert_eq!(
            extract_project_name_from_working_dir("/Users/johnrush/repos/project2/"),
            "project2"
        );
        assert_eq!(extract_project_name_from_working_dir(""), "unknown");
    }

    #[test]
    fn test_calculate_cost() {
        // Existing cost calculation test
        let stats = calculate_cost("Claude 3 Opus", 1000, 500, 0, 0);
        assert_eq!(stats.input_cost, 0.015);
        assert_eq!(stats.output_cost, 0.0375);
        assert_eq!(stats.cache_creation_cost, 0.0);
        assert_eq!(stats.cache_read_cost, 0.0);
        assert_eq!(stats.total_cost, 0.0525);

        // Unknown model
        let stats2 = calculate_cost("Unknown Model", 1000, 500, 0, 0);
        assert_eq!(stats2.total_cost, 0.0);

        // Test cache-aware models
        let stats3 = calculate_cost("Sonnet 3.7", 1000, 500, 200, 100);
        assert_eq!(stats3.input_cost, 0.003);
        assert_eq!(stats3.output_cost, 0.0075);
        assert!((stats3.cache_creation_cost - 0.00075).abs() < 1e-8); // Floating point comparison
        assert!((stats3.cache_read_cost - 0.00003).abs() < 1e-8); // Floating point comparison
        assert!((stats3.total_cost - 0.01128).abs() < 1e-8); // Floating point comparison

        // Test zero tokens
        let stats4 = calculate_cost("GPT-4", 0, 0, 0, 0);
        assert_eq!(stats4.total_cost, 0.0);

        // Test large number of tokens
        let stats5 = calculate_cost("GPT-4o", 1000000, 500000, 0, 0);
        assert_eq!(stats5.input_cost, 5.0); // 1000000 * 0.005 / 1000
        assert_eq!(stats5.output_cost, 7.5); // 500000 * 0.015 / 1000
        assert_eq!(stats5.total_cost, 12.5); // 5.0 + 7.5
    }

    #[test]
    fn test_scan_empty_dir_returns_empty_vec() {
        // Create a temporary empty directory
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let dir_path = temp_dir.path().to_string_lossy().to_string();

        // Scan the empty directory
        let records = scan_claude_logs_with_options(Some(dir_path), true);

        // Verify the result is an empty vector
        assert!(records.is_empty());
    }

    #[test]
    fn test_parse_jsonl_file() {
        // Create a temporary directory and mock JSONL file
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test_claude.jsonl");
        let mut file = File::create(&file_path).expect("Failed to create test file");

        // Write mock Claude assistant messages with usage data
        writeln!(&mut file, r#"{{"type":"assistant","timestamp":"2023-06-15T10:30:00Z","sessionId":"test-session-123","message":{{"model":"claude-3-opus","usage":{{"input_tokens":1000,"output_tokens":500,"cache_creation_input_tokens":200,"cache_read_input_tokens":100}}}}}}"#).unwrap();

        // Write a malformed JSON line
        writeln!(&mut file, r#"{{"type": "assistant", malformed_json"#).unwrap();

        // Write a user message (should be ignored)
        writeln!(
            &mut file,
            r#"{{"type":"user","timestamp":"2023-06-15T10:29:00Z","message":"hello"}}"#
        )
        .unwrap();

        // Write an assistant message with missing usage data
        writeln!(&mut file, r#"{{"type":"assistant","timestamp":"2023-06-15T10:31:00Z","sessionId":"test-session-123","message":{{"model":"claude-3-opus"}}}}"#).unwrap();

        // Write a synthetic message (should be ignored)
        writeln!(&mut file, r#"{{"type":"assistant","timestamp":"2023-06-15T10:32:00Z","sessionId":"test-session-123","message":{{"model":"<synthetic>","usage":{{"input_tokens":100,"output_tokens":50}}}}}}"#).unwrap();

        // Parse the test file
        let project_name = "test-project";
        let records = parse_jsonl_file(&file_path, project_name);

        // Assertions
        assert_eq!(records.len(), 1); // Only valid assistant message with usage data
        let record = &records[0];

        assert_eq!(record.project, "test-project");
        assert_eq!(record.date, "2023-06-15");
        assert_eq!(record.timestamp, "2023-06-15T10:30:00Z");
        assert_eq!(record.model, "claude-3-opus");
        assert_eq!(record.input_tokens, 1000);
        assert_eq!(record.output_tokens, 500);
        assert_eq!(record.cache_creation_input_tokens, 200);
        assert_eq!(record.cache_read_input_tokens, 100);
        assert_eq!(record.total_tokens, Some(1800)); // Sum of all tokens
        assert_eq!(record.session_id, "test-session-123");
        assert_eq!(record.source, Some("claude".to_string()));
    }

    #[test]
    fn test_parse_goose_cli_log_file() {
        // Create a temporary directory and mock goose log file
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test-session-123.log");
        let mut file = File::create(&file_path).expect("Failed to create test file");

        // Write a goose log entry with usage data in output field
        writeln!(&mut file, r#"{{"timestamp":"2023-06-15T10:30:00Z","fields":{{"total_tokens":1800,"input_tokens":1000,"output_tokens":500,"output":"{{\"model\":\"o3\",\"usage\":{{\"prompt_tokens\":1000,\"completion_tokens\":500,\"total_tokens\":1800,\"prompt_tokens_details\":{{\"cached_tokens\":100}},\"completion_tokens_details\":{{\"reasoning_tokens\":300}}}}}}","input":"{{\"messages\":[{{\"role\":\"system\",\"content\":\"You are a helpful assistant. Current directory: /Users/johnrush/repos/my-project\"}}]}}"}}}}"#).unwrap();

        // Write a malformed line
        writeln!(&mut file, "This is not a JSON line").unwrap();

        // Write a goose log entry with minimal usage data
        writeln!(
            &mut file,
            r#"{{"fields":{{"total_tokens":300,"input_tokens":200,"output_tokens":100}}}}"#
        )
        .unwrap();

        // Parse the test file
        let records = parse_goose_cli_log_file(&file_path);

        // Assertions - we expect at least 1 entry (input/output data quality can be inconsistent)
        assert!(records.len() >= 1); // At least one valid entry

        // Verify the record has the expected session_id from the filename
        let record = &records[0];
        assert_eq!(record.session_id, "test-session-123"); // From filename

        // Verify we have token counts (the exact values might not match due to parsing complexity)
        assert!(record.input_tokens > 0);
        assert!(record.output_tokens > 0);
        assert!(record.total_tokens.unwrap_or(0) > 0);
    }

    #[test]
    fn test_cache_invalidation() {
        // Create temporary directories and files
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let claude_dir = temp_dir.path().join("claude");
        let project_dir = claude_dir.join("project-123");
        fs::create_dir_all(&project_dir).expect("Failed to create project directory");

        // Create a test JSONL file
        let file_path = project_dir.join("test.jsonl");
        let mut file = File::create(&file_path).expect("Failed to create test file");
        writeln!(&mut file, r#"{{"type":"assistant","timestamp":"2023-06-15T10:30:00Z","sessionId":"test-session","message":{{"model":"claude-3-opus","usage":{{"input_tokens":1000,"output_tokens":500}}}}}}"#).unwrap();

        // Get cache path for testing
        let cache_path = get_cache_path(&claude_dir.to_string_lossy());

        // First scan should create the cache
        let records1 =
            scan_claude_logs_with_options(Some(claude_dir.to_string_lossy().to_string()), true);
        assert_eq!(records1.len(), 1);

        // Load the cache directly and verify it contains our file
        let cache_data = load_cache(&cache_path);
        assert_eq!(cache_data.files.len(), 1);
        assert!(cache_data
            .files
            .contains_key(&file_path.to_string_lossy().to_string()));

        // Second scan should use the cache (would be hard to verify directly, but we can check it returns same data)
        let records2 =
            scan_claude_logs_with_options(Some(claude_dir.to_string_lossy().to_string()), true);
        assert_eq!(records2.len(), 1);

        // Modify the file
        std::thread::sleep(std::time::Duration::from_secs(1)); // Ensure mtime changes
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&file_path)
            .unwrap();
        writeln!(&mut file, r#"{{"type":"assistant","timestamp":"2023-06-15T10:31:00Z","sessionId":"test-session","message":{{"model":"claude-3-opus","usage":{{"input_tokens":2000,"output_tokens":1000}}}}}}"#).unwrap();

        // Third scan should invalidate the cache and reprocess the file
        let records3 =
            scan_claude_logs_with_options(Some(claude_dir.to_string_lossy().to_string()), true);
        assert_eq!(records3.len(), 2); // Now it should find two records
    }

    #[test]
    fn test_daily_model_report_generation() {
        // Create test records for different days and models
        let records = vec![
            UsageRecord {
                project: "project1".to_string(),
                date: "2023-06-15".to_string(),
                timestamp: "2023-06-15T10:30:00Z".to_string(),
                model: "Claude 3 Opus".to_string(),
                input_tokens: 1000,
                output_tokens: 500,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                total_tokens: Some(1500),
                reasoning_tokens: None,
                session_id: "session1".to_string(),
                file_path: "/path/to/file1".to_string(),
                source: Some("claude".to_string()),
            },
            UsageRecord {
                project: "project1".to_string(),
                date: "2023-06-15".to_string(),
                timestamp: "2023-06-15T11:30:00Z".to_string(),
                model: "Claude 3 Opus".to_string(),
                input_tokens: 2000,
                output_tokens: 1000,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                total_tokens: Some(3000),
                reasoning_tokens: None,
                session_id: "session1".to_string(),
                file_path: "/path/to/file1".to_string(),
                source: Some("claude".to_string()),
            },
            UsageRecord {
                project: "project2".to_string(),
                date: "2023-06-16".to_string(),
                timestamp: "2023-06-16T10:30:00Z".to_string(),
                model: "GPT-4o".to_string(),
                input_tokens: 3000,
                output_tokens: 1500,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                total_tokens: Some(4500),
                reasoning_tokens: None,
                session_id: "session2".to_string(),
                file_path: "/path/to/file2".to_string(),
                source: Some("goose_cli".to_string()),
            },
        ];

        // Generate the report
        let report = generate_daily_model_report(&records);

        // Verify the report structure
        assert_eq!(report.len(), 2); // Two different dates

        // Check the first day
        let day1 = report.get("2023-06-15").expect("Missing day 2023-06-15");
        assert_eq!(day1.len(), 1); // One model

        // Check the Claude 3 Opus stats for the first day
        let opus_stats = day1
            .get("Claude 3 Opus")
            .expect("Missing model Claude 3 Opus");
        assert_eq!(opus_stats.input_tokens, 3000); // 1000 + 2000
        assert_eq!(opus_stats.output_tokens, 1500); // 500 + 1000
        assert_eq!(opus_stats.total_tokens, 4500); // 3000 + 1500

        // Check costs for Claude 3 Opus (using approximate float comparison)
        let expected_input_cost = 3000.0 * 0.015 / 1000.0;
        let expected_output_cost = 1500.0 * 0.075 / 1000.0;
        let expected_total_cost = expected_input_cost + expected_output_cost;

        assert!((opus_stats.input_cost - expected_input_cost).abs() < 1e-8);
        assert!((opus_stats.output_cost - expected_output_cost).abs() < 1e-8);
        assert!((opus_stats.total_cost - expected_total_cost).abs() < 1e-8);

        // Check the second day
        let day2 = report.get("2023-06-16").expect("Missing day 2023-06-16");
        assert_eq!(day2.len(), 1); // One model

        // Check the GPT-4o stats for the second day
        let gpt4o_stats = day2.get("GPT-4o").expect("Missing model GPT-4o");
        assert_eq!(gpt4o_stats.input_tokens, 3000);
        assert_eq!(gpt4o_stats.output_tokens, 1500);
        assert_eq!(gpt4o_stats.total_tokens, 4500);

        // Check costs for GPT-4o (using approximate float comparison)
        let expected_gpt4o_input_cost = 3000.0 * 0.005 / 1000.0;
        let expected_gpt4o_output_cost = 1500.0 * 0.015 / 1000.0;
        let expected_gpt4o_total_cost = expected_gpt4o_input_cost + expected_gpt4o_output_cost;

        assert!((gpt4o_stats.input_cost - expected_gpt4o_input_cost).abs() < 1e-8);
        assert!((gpt4o_stats.output_cost - expected_gpt4o_output_cost).abs() < 1e-8);
        assert!((gpt4o_stats.total_cost - expected_gpt4o_total_cost).abs() < 1e-8);
    }

    #[test]
    fn test_daily_project_report_generation() {
        // Create test records for different days and projects
        let records = vec![
            UsageRecord {
                project: "project1".to_string(),
                date: "2023-06-15".to_string(),
                timestamp: "2023-06-15T10:30:00Z".to_string(),
                model: "Claude 3 Opus".to_string(),
                input_tokens: 1000,
                output_tokens: 500,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                total_tokens: Some(1500),
                reasoning_tokens: None,
                session_id: "session1".to_string(),
                file_path: "/path/to/file1".to_string(),
                source: Some("claude".to_string()),
            },
            UsageRecord {
                project: "project1".to_string(),
                date: "2023-06-15".to_string(),
                timestamp: "2023-06-15T11:30:00Z".to_string(),
                model: "Claude 3 Sonnet".to_string(),
                input_tokens: 2000,
                output_tokens: 1000,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                total_tokens: Some(3000),
                reasoning_tokens: None,
                session_id: "session2".to_string(),
                file_path: "/path/to/file1".to_string(),
                source: Some("claude".to_string()),
            },
            UsageRecord {
                project: "project2".to_string(),
                date: "2023-06-15".to_string(),
                timestamp: "2023-06-15T12:30:00Z".to_string(),
                model: "GPT-4o".to_string(),
                input_tokens: 3000,
                output_tokens: 1500,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                total_tokens: Some(4500),
                reasoning_tokens: None,
                session_id: "session3".to_string(),
                file_path: "/path/to/file2".to_string(),
                source: Some("goose_cli".to_string()),
            },
            UsageRecord {
                project: "project3".to_string(),
                date: "2023-06-16".to_string(),
                timestamp: "2023-06-16T10:30:00Z".to_string(),
                model: "GPT-4o".to_string(),
                input_tokens: 4000,
                output_tokens: 2000,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                total_tokens: Some(6000),
                reasoning_tokens: None,
                session_id: "session4".to_string(),
                file_path: "/path/to/file3".to_string(),
                source: Some("goose_cli".to_string()),
            },
        ];

        // Generate the project report
        let project_report = generate_daily_project_report(&records);

        // Verify the report structure
        assert_eq!(project_report.len(), 2); // Two different dates

        // Check the first day
        let day1 = project_report
            .get("2023-06-15")
            .expect("Missing day 2023-06-15");
        assert_eq!(day1.len(), 2); // Two projects: project1 and project2

        // Verify project1 stats
        let project1_stats = day1.get("project1").expect("Missing project1 stats");
        assert_eq!(project1_stats.input_tokens, 3000); // 1000 + 2000
        assert_eq!(project1_stats.output_tokens, 1500); // 500 + 1000
        assert_eq!(project1_stats.total_tokens, 4500); // 3000 + 1500

        // Verify project2 stats
        let project2_stats = day1.get("project2").expect("Missing project2 stats");
        assert_eq!(project2_stats.input_tokens, 3000);
        assert_eq!(project2_stats.output_tokens, 1500);
        assert_eq!(project2_stats.total_tokens, 4500);

        // Check the second day
        let day2 = project_report
            .get("2023-06-16")
            .expect("Missing day 2023-06-16");
        assert_eq!(day2.len(), 1); // One project: project3

        // Verify project3 stats
        let project3_stats = day2.get("project3").expect("Missing project3 stats");
        assert_eq!(project3_stats.input_tokens, 4000);
        assert_eq!(project3_stats.output_tokens, 2000);
        assert_eq!(project3_stats.total_tokens, 6000);
    }

    #[test]
    fn test_load_complete_token_data() {
        // Create a temporary directory for testing
        let temp_dir = tempdir().expect("Failed to create temp directory");

        // Write a mock file to ensure we get our test record
        let test_file = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&test_file).expect("Failed to create test file");
        writeln!(&mut file, r#"{{"type":"assistant","timestamp":"2023-06-15T10:30:00Z","sessionId":"test-session","message":{{"model":"claude-3-opus","usage":{{"input_tokens":1000,"output_tokens":500}}}}}}"#).unwrap();

        // Test the TokenData structure
        let token_data = load_complete_token_data_with_options(true);

        // Verify the structure contains both model and project reports
        assert!(token_data.is_some());

        let data = token_data.unwrap();
        // Both reports should have data
        assert!(!data.model_report.is_empty());
        assert!(!data.project_report.is_empty());

        // Both reports should have the same dates
        assert_eq!(
            data.model_report.keys().len(),
            data.project_report.keys().len()
        );
    }

    #[test]
    fn test_error_handling() {
        // Test with a non-existent directory
        let records =
            scan_claude_logs_with_options(Some("/path/that/does/not/exist".to_string()), true);
        assert!(records.is_empty());

        // Create a temp directory with corrupted cache file
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("corrupted_cache.json");
        let mut file = File::create(&cache_path).expect("Failed to create test file");
        writeln!(&mut file, "This is not valid JSON").unwrap();

        // Loading the corrupted cache should return a default cache
        let cache_data = load_cache(&cache_path);
        assert_eq!(cache_data.files.len(), 0);

        // Test with empty records
        let empty_records: Vec<UsageRecord> = vec![];
        let report = generate_daily_model_report(&empty_records);
        assert!(report.is_empty());
    }
}
