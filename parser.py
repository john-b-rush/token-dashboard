#!/usr/bin/env python3
"""
Claude Token Usage Parser

Parses Claude conversation logs from ~/.claude/projects/ and extracts token usage statistics.
Generates reports by date, project, and model.
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

# Pricing table (per 1000 tokens)
MODEL_PRICING = {
    # Anthropic Claude models
    "Opus 4": {
        "input_tokens": 0.015,
        "output_tokens": 0.075,
        "input_tokens_batch": None,
        "output_tokens_batch": None,
        "input_tokens_cache_write": 0.01875,
        "input_tokens_cache_read": 0.0015,
    },
    "Sonnet 4": {
        "input_tokens": 0.003,
        "output_tokens": 0.015,
        "input_tokens_batch": None,
        "output_tokens_batch": None,
        "input_tokens_cache_write": 0.00375,
        "input_tokens_cache_read": 0.0003,
    },
    "Sonnet 3.7": {
        "input_tokens": 0.003,
        "output_tokens": 0.015,
        "input_tokens_batch": None,
        "output_tokens_batch": None,
        "input_tokens_cache_write": 0.00375,
        "input_tokens_cache_read": 0.0003,
    },
    "Claude 3.5 Sonnet": {
        "input_tokens": 0.003,
        "output_tokens": 0.015,
        "input_tokens_batch": 0.0015,
        "output_tokens_batch": 0.0075,
        "input_tokens_cache_write": 0.00375,
        "input_tokens_cache_read": 0.0003,
    },
    "Claude 3.5 Haiku": {
        "input_tokens": 0.0008,
        "output_tokens": 0.004,
        "input_tokens_batch": 0.0005,
        "output_tokens_batch": 0.0025,
        "input_tokens_cache_write": 0.001,
        "input_tokens_cache_read": 0.00008,
    },
    "Claude 3 Opus": {
        "input_tokens": 0.015,
        "output_tokens": 0.075,
        "input_tokens_batch": 0.0075,
        "output_tokens_batch": 0.0375,
        "input_tokens_cache_write": None,
        "input_tokens_cache_read": None,
    },
    # OpenAI models (common ones used in goose)
    "GPT-4o": {
        "input_tokens": 0.0025,
        "output_tokens": 0.01,
        "input_tokens_batch": None,
        "output_tokens_batch": None,
        "input_tokens_cache_write": None,
        "input_tokens_cache_read": None,
    },
    "GPT-4": {
        "input_tokens": 0.03,
        "output_tokens": 0.06,
        "input_tokens_batch": None,
        "output_tokens_batch": None,
        "input_tokens_cache_write": None,
        "input_tokens_cache_read": None,
    },
    "o3": {
        "input_tokens": 0.06,  # Estimated pricing for o3
        "output_tokens": 0.24,
        "input_tokens_batch": None,
        "output_tokens_batch": None,
        "input_tokens_cache_write": None,
        "input_tokens_cache_read": None,
    },
}


def extract_project_name(folder_path: str) -> str:
    """Extract project name from Claude folder path like '-Users-johnrush-repos-chuck-data2'"""
    folder_name = os.path.basename(folder_path)
    # Remove leading dash if present
    if folder_name.startswith("-"):
        folder_name = folder_name[1:]

    # Remove the '-Users-johnrush-repos-' prefix and return the project name
    prefix = "Users-johnrush-repos-"
    if folder_name.startswith(prefix):
        return folder_name[len(prefix) :]

    # Fallback: convert dashes back to slashes and extract the last part
    path_parts = folder_name.replace("-", "/")
    return os.path.basename(path_parts)


def extract_project_name_from_working_dir(working_dir: str) -> str:
    """Extract project name from goose working directory path like '/Users/johnrush/repos/chuck-data4'"""
    if not working_dir:
        return "unknown"

    # Extract the last directory name from the path
    project_name = os.path.basename(working_dir.rstrip("/"))

    # Handle special cases
    if project_name in ("", "."):
        return "root"
    elif working_dir == os.path.expanduser("~"):
        return "home"

    return project_name


def parse_jsonl_file(file_path: str, project_name: str) -> List[Dict[str, Any]]:
    """Parse a JSONL file and extract token usage records"""
    records = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Only process assistant messages with usage data
                    if (
                        data.get("type") == "assistant"
                        and "message" in data
                        and "usage" in data["message"]
                    ):

                        usage = data["message"]["usage"]
                        timestamp = data.get("timestamp")
                        model = data["message"].get("model", "unknown")

                        if timestamp and model != "<synthetic>":
                            # Parse timestamp and extract date
                            dt = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            date = dt.date().isoformat()

                            record = {
                                "project": project_name,
                                "date": date,
                                "timestamp": timestamp,
                                "model": model,
                                "input_tokens": usage.get("input_tokens", 0),
                                "output_tokens": usage.get("output_tokens", 0),
                                "cache_creation_input_tokens": usage.get(
                                    "cache_creation_input_tokens", 0
                                ),
                                "cache_read_input_tokens": usage.get(
                                    "cache_read_input_tokens", 0
                                ),
                                "session_id": data.get("sessionId", ""),
                                "file_path": file_path,
                            }

                            # Add server tool use if present
                            if "server_tool_use" in usage:
                                record["server_tool_use"] = usage["server_tool_use"]

                            records.append(record)

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Failed to parse JSON at {file_path}:{line_num} - {e}"
                    )
                    continue

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return records


def parse_goose_session_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a goose session file and extract token usage records"""
    records = []
    project_name = None
    session_id = os.path.basename(file_path).replace(".jsonl", "")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # First line contains session summary with usage statistics
                    if line_num == 1 and "working_dir" in data:
                        # Extract project name from working directory
                        working_dir = data.get("working_dir", "")
                        project_name = extract_project_name_from_working_dir(
                            working_dir
                        )

                        # Extract accumulated usage statistics
                        total_tokens = data.get("accumulated_total_tokens") or 0
                        input_tokens = data.get("accumulated_input_tokens") or 0
                        output_tokens = data.get("accumulated_output_tokens") or 0

                        if total_tokens > 0:
                            # Use file modification time as timestamp for session summary
                            file_mtime = os.path.getmtime(file_path)
                            dt = datetime.fromtimestamp(file_mtime)
                            date = dt.date().isoformat()
                            iso_timestamp = dt.isoformat()

                            # Create a record for the accumulated session usage
                            record = {
                                "project": project_name or "unknown",
                                "date": date,
                                "timestamp": iso_timestamp,
                                "model": "unknown",  # Session summary doesn't specify model
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total_tokens,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "reasoning_tokens": 0,
                                "session_id": session_id,
                                "file_path": file_path,
                                "source": "goose",
                            }

                            records.append(record)

                    # Skip individual message records for now to avoid double counting
                    # We're using the session summary which has accumulated totals

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Failed to parse JSON at {file_path}:{line_num} - {e}"
                    )
                    continue

    except Exception as e:
        print(f"Error reading goose session file {file_path}: {e}")

    return records


def parse_goose_cli_log_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a goose CLI log file and extract token usage records"""
    records = []
    session_id = os.path.basename(file_path).replace(".log", "")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Look for lines with token usage in the fields
                    if (
                        data.get("fields")
                        and "total_tokens" in data.get("fields", {})
                        and "output" in data.get("fields", {})
                    ):

                        fields = data["fields"]

                        # Parse the output JSON to get detailed usage and model info
                        try:
                            output_data = json.loads(fields["output"])
                            usage = output_data.get("usage", {})
                            model = output_data.get("model", "unknown")
                        except (json.JSONDecodeError, KeyError):
                            # Fallback to fields data if output parsing fails
                            usage = {
                                "prompt_tokens": int(fields.get("input_tokens", 0)),
                                "completion_tokens": int(
                                    fields.get("output_tokens", 0)
                                ),
                                "total_tokens": int(fields.get("total_tokens", 0)),
                            }
                            model = "unknown"

                        # Parse input messages to extract working directory/project info
                        project_name = "unknown"
                        try:
                            input_data = json.loads(fields["input"])
                            messages = input_data.get("messages", [])

                            # Look for working directory in system messages
                            for message in messages:
                                content = message.get("content", "")
                                if "current directory:" in content:
                                    # Extract working directory path
                                    import re

                                    match = re.search(
                                        r"current directory: (/[^\s\n]+)", content
                                    )
                                    if match:
                                        working_dir = match.group(1)
                                        project_name = (
                                            extract_project_name_from_working_dir(
                                                working_dir
                                            )
                                        )
                                        break
                        except (json.JSONDecodeError, KeyError):
                            pass

                        # Parse timestamp
                        timestamp = data.get("timestamp")
                        if timestamp:
                            dt = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            date = dt.date().isoformat()
                        else:
                            # Fallback to file modification time
                            file_mtime = os.path.getmtime(file_path)
                            dt = datetime.fromtimestamp(file_mtime)
                            date = dt.date().isoformat()
                            timestamp = dt.isoformat()

                        # Extract token counts
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get(
                            "total_tokens", input_tokens + output_tokens
                        )

                        # Extract detailed token info if available
                        reasoning_tokens = 0
                        if "completion_tokens_details" in usage:
                            reasoning_tokens = usage["completion_tokens_details"].get(
                                "reasoning_tokens", 0
                            )

                        cache_tokens = 0
                        if "prompt_tokens_details" in usage:
                            cache_tokens = usage["prompt_tokens_details"].get(
                                "cached_tokens", 0
                            )

                        record = {
                            "project": project_name,
                            "date": date,
                            "timestamp": timestamp,
                            "model": model,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "cache_creation_input_tokens": 0,  # Not typically in goose logs
                            "cache_read_input_tokens": cache_tokens,
                            "reasoning_tokens": reasoning_tokens,
                            "session_id": session_id,
                            "file_path": file_path,
                            "source": "goose_cli",
                        }

                        records.append(record)

                except json.JSONDecodeError as e:
                    # Skip non-JSON lines (some logs may have mixed content)
                    continue

    except Exception as e:
        print(f"Error reading goose CLI log file {file_path}: {e}")

    return records


def get_cache_path(claude_dir: str) -> str:
    """Get the cache file path for the given Claude directory"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "claude-token-burn")
    os.makedirs(cache_dir, exist_ok=True)
    # Create a safe filename from the claude_dir path
    safe_name = claude_dir.replace("/", "_").replace("\\", "_").replace(":", "_")
    return os.path.join(cache_dir, f"cache_{safe_name}.json")


def load_cache(cache_path: str) -> Dict[str, Any]:
    """Load cache from JSON file"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {"files": {}, "records": []}


def save_cache(cache_path: str, cache_data: Dict[str, Any]):
    """Save cache to JSON file"""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")


def is_file_modified(file_path: str, cached_mtime: float) -> bool:
    """Check if file has been modified since the cached modification time"""
    try:
        current_mtime = os.path.getmtime(file_path)
        return current_mtime > cached_mtime
    except OSError:
        return True


def scan_claude_logs(claude_dir: str = None) -> List[Dict[str, Any]]:
    """Scan all Claude log files and extract token usage data with caching"""
    if claude_dir is None:
        claude_dir = os.path.expanduser("~/.claude/projects")

    if not os.path.exists(claude_dir):
        raise FileNotFoundError(f"Claude projects directory not found: {claude_dir}")

    # Load cache
    cache_path = get_cache_path(claude_dir)
    cache_data = load_cache(cache_path)
    cached_files = cache_data["files"]
    all_records = []
    files_processed = 0
    files_from_cache = 0

    # Iterate through project folders
    for project_folder in os.listdir(claude_dir):
        project_path = os.path.join(claude_dir, project_folder)

        if not os.path.isdir(project_path):
            continue

        project_name = extract_project_name(project_folder)

        # Process all .jsonl files in the project folder
        for file_name in os.listdir(project_path):
            if file_name.endswith(".jsonl"):
                file_path = os.path.join(project_path, file_name)

                try:
                    current_mtime = os.path.getmtime(file_path)
                except OSError:
                    continue

                # Check if file is in cache and hasn't been modified
                if file_path in cached_files and not is_file_modified(
                    file_path, cached_files[file_path]["mtime"]
                ):
                    # Use cached records
                    cached_records = cached_files[file_path]["records"]
                    all_records.extend(cached_records)
                    files_from_cache += 1
                else:
                    # Parse file and update cache
                    records = parse_jsonl_file(file_path, project_name)
                    all_records.extend(records)

                    # Update cache entry
                    cached_files[file_path] = {
                        "mtime": current_mtime,
                        "records": records,
                    }
                    files_processed += 1

    # Save updated cache
    cache_data["files"] = cached_files
    save_cache(cache_path, cache_data)

    if files_processed > 0 or files_from_cache > 0:
        print(
            f"Processed {files_processed} files, used cache for {files_from_cache} files"
        )

    return all_records


def scan_goose_logs(goose_dir: str = None) -> List[Dict[str, Any]]:
    """Scan all goose CLI log files and extract token usage data with caching"""
    if goose_dir is None:
        goose_dir = os.path.expanduser("~/.local/state/goose/logs/cli")

    if not os.path.exists(goose_dir):
        print(f"Goose CLI logs directory not found: {goose_dir}")
        return []

    # Load cache
    cache_path = get_cache_path(goose_dir)
    cache_data = load_cache(cache_path)
    cached_files = cache_data["files"]
    all_records = []
    files_processed = 0
    files_from_cache = 0

    # Process all .log files in date-organized subdirectories
    for date_folder in os.listdir(goose_dir):
        date_path = os.path.join(goose_dir, date_folder)
        if not os.path.isdir(date_path):
            continue

        for file_name in os.listdir(date_path):
            # Process main session log files (not MCP-specific logs)
            if (
                file_name.endswith(".log")
                and not file_name.endswith("-mcp-developer.log")
                and not file_name.endswith("-mcp-memory.log")
            ):
                file_path = os.path.join(date_path, file_name)

                try:
                    current_mtime = os.path.getmtime(file_path)
                except OSError:
                    continue

                # Check if file is in cache and hasn't been modified
                if file_path in cached_files and not is_file_modified(
                    file_path, cached_files[file_path]["mtime"]
                ):
                    # Use cached records
                    cached_records = cached_files[file_path]["records"]
                    all_records.extend(cached_records)
                    files_from_cache += 1
                else:
                    # Parse file and update cache
                    records = parse_goose_cli_log_file(file_path)
                    all_records.extend(records)

                    # Update cache entry
                    cached_files[file_path] = {
                        "mtime": current_mtime,
                        "records": records,
                    }
                    files_processed += 1

    # Save updated cache
    cache_data["files"] = cached_files
    save_cache(cache_path, cache_data)

    if files_processed > 0 or files_from_cache > 0:
        print(
            f"Goose: Processed {files_processed} files, used cache for {files_from_cache} files"
        )

    return all_records


def normalize_model_name(model: str) -> str:
    """Normalize model name to match pricing table keys"""
    model_lower = model.lower()

    # Map Claude model names to pricing table keys
    if "opus-4" in model_lower:
        return "Opus 4"
    elif "sonnet-4" in model_lower:
        return "Sonnet 4"
    elif "claude-3.7" in model_lower and "sonnet" in model_lower:
        return "Sonnet 3.7"
    elif "3-7-sonnet" in model_lower:
        return "Sonnet 3.7"
    elif "claude-3-5-sonnet" in model_lower or "claude-3.5-sonnet" in model_lower:
        return "Claude 3.5 Sonnet"
    elif "claude-3-5-haiku" in model_lower or "claude-3.5-haiku" in model_lower:
        return "Claude 3.5 Haiku"
    elif "claude-3-opus" in model_lower:
        return "Claude 3 Opus"

    # Map OpenAI/goose model names to pricing table keys
    elif model_lower == "o3" or "o3-2025" in model_lower:
        return "o3"
    elif "o1" in model_lower:
        return "o1"
    elif "gpt-4o" in model_lower:
        return "GPT-4o"
    elif "gpt-4" in model_lower and "gpt-4o" not in model_lower:
        return "GPT-4"

    return model  # Return original if no match


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> Dict[str, float]:
    """Calculate cost breakdown for token usage"""
    normalized_model = normalize_model_name(model)

    if normalized_model not in MODEL_PRICING:
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cache_creation_cost": 0.0,
            "cache_read_cost": 0.0,
            "total_cost": 0.0,
        }

    pricing = MODEL_PRICING[normalized_model]

    input_cost = (
        (input_tokens / 1000) * pricing["input_tokens"]
        if pricing["input_tokens"]
        else 0.0
    )
    output_cost = (
        (output_tokens / 1000) * pricing["output_tokens"]
        if pricing["output_tokens"]
        else 0.0
    )
    cache_creation_cost = (
        (cache_creation_tokens / 1000) * pricing["input_tokens_cache_write"]
        if pricing["input_tokens_cache_write"]
        else 0.0
    )
    cache_read_cost = (
        (cache_read_tokens / 1000) * pricing["input_tokens_cache_read"]
        if pricing["input_tokens_cache_read"]
        else 0.0
    )

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_creation_cost": cache_creation_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": input_cost + output_cost + cache_creation_cost + cache_read_cost,
    }


def generate_daily_model_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate daily breakdown of token usage by model with cost information"""
    if not records:
        return {}

    # Group by date and model
    daily_model_usage = defaultdict(
        lambda: defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "cache_creation_cost": 0.0,
                "cache_read_cost": 0.0,
                "total_cost": 0.0,
            }
        )
    )

    # Process each record
    for record in records:
        date = record["date"]
        model = record["model"]

        usage = daily_model_usage[date][model]
        usage["input_tokens"] += record["input_tokens"]
        usage["output_tokens"] += record["output_tokens"]
        usage["cache_creation_tokens"] += record["cache_creation_input_tokens"]
        usage["cache_read_tokens"] += record["cache_read_input_tokens"]
        usage["total_tokens"] += (
            record["input_tokens"]
            + record["output_tokens"]
            + record["cache_creation_input_tokens"]
            + record["cache_read_input_tokens"]
        )

        # Calculate costs for this record
        costs = calculate_cost(
            model,
            record["input_tokens"],
            record["output_tokens"],
            record["cache_creation_input_tokens"],
            record["cache_read_input_tokens"],
        )

        usage["input_cost"] += costs["input_cost"]
        usage["output_cost"] += costs["output_cost"]
        usage["cache_creation_cost"] += costs["cache_creation_cost"]
        usage["cache_read_cost"] += costs["cache_read_cost"]
        usage["total_cost"] += costs["total_cost"]

    # Convert to regular dict and sort dates
    result = {}
    for date in sorted(daily_model_usage.keys()):
        result[date] = dict(daily_model_usage[date])

    return result


def scan_all_logs(
    claude_dir: str = None, goose_dir: str = None
) -> List[Dict[str, Any]]:
    """Scan both Claude and goose logs and combine the results"""
    all_records = []

    # Scan Claude logs
    claude_records = scan_claude_logs(claude_dir)
    all_records.extend(claude_records)

    # Scan goose logs
    goose_records = scan_goose_logs(goose_dir)
    all_records.extend(goose_records)

    return all_records


def main():
    """Main function to run the parser"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse Claude and Goose token usage from conversation logs"
    )
    parser.add_argument(
        "--claude-dir",
        default=os.path.expanduser("~/.claude/projects"),
        help="Path to Claude projects directory (default: ~/.claude/projects)",
    )
    parser.add_argument(
        "--goose-dir",
        default=os.path.expanduser("~/.local/state/goose/logs/cli"),
        help="Path to Goose CLI logs directory (default: ~/.local/state/goose/logs/cli)",
    )
    parser.add_argument(
        "--claude-only", action="store_true", help="Only scan Claude logs"
    )
    parser.add_argument(
        "--goose-only", action="store_true", help="Only scan Goose logs"
    )
    parser.add_argument(
        "--output", "-o", help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "daily"],
        default="daily",
        help="Output format (default: daily)",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Output raw records instead of summary"
    )

    args = parser.parse_args()

    try:
        if args.claude_only:
            records = scan_claude_logs(args.claude_dir)
        elif args.goose_only:
            records = scan_goose_logs(args.goose_dir)
        else:
            records = scan_all_logs(args.claude_dir, args.goose_dir)

        if not records:
            return


        if args.raw:
            output_data = records
        else:
            output_data = generate_daily_model_report(records)

        # Format output
        if args.format == "json":
            output_json = json.dumps(output_data, indent=2, default=str)
        else:
            # Daily format - show token usage by model for each day
            if args.raw:
                output_json = f"Total records: {len(records)}"
            else:
                output_lines = ["Daily Token Usage by Model", "=" * 30, ""]
                daily_total = 0.0
                for date in sorted(output_data.keys()):
                    date_total = 0.0
                    models = output_data[date]
                    date_lines = []

                    for model in sorted(models.keys()):
                        usage = models[model]

                        # Skip models with no token usage
                        if usage["total_tokens"] == 0:
                            continue
                        cache_info = ""
                        cache_cost_info = ""
                        if (
                            usage["cache_creation_tokens"] > 0
                            or usage["cache_read_tokens"] > 0
                        ):
                            cache_info = f", cache_create: {usage['cache_creation_tokens']:,}, cache_read: {usage['cache_read_tokens']:,}"
                            cache_cost_info = f", cache_create: ${usage['cache_creation_cost']:.4f}, cache_read: ${usage['cache_read_cost']:.4f}"

                        # Debug: show normalized model name
                        normalized = normalize_model_name(model)
                        debug_info = (
                            f" [normalized: {normalized}]"
                            if normalized != model
                            else ""
                        )

                        date_lines.append(
                            f"  {model}: {usage['total_tokens']:,} tokens (input: {usage['input_tokens']:,}, output: {usage['output_tokens']:,}{cache_info}){debug_info}"
                        )
                        date_lines.append(
                            f"    Cost: ${usage['total_cost']:.4f} (input: ${usage['input_cost']:.4f}, output: ${usage['output_cost']:.4f}{cache_cost_info})"
                        )
                        date_total += usage["total_cost"]

                    # Only add date section if there are models with usage
                    if date_lines:
                        output_lines.append(f"Date: {date}")
                        output_lines.extend(date_lines)
                        output_lines.append(f"  Daily Total: ${date_total:.4f}")
                        output_lines.append("")
                        daily_total += date_total
                output_lines.append(f"Grand Total: ${daily_total:.4f}")
                output_json = "\n".join(output_lines)

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
        else:
            print("\n" + output_json)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
