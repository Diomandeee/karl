#!/usr/bin/env python3
"""KARL V7 Full Factory — Deep training data generation.

Run by OpenCode. Orchestrates 20+ project sessions across all available
panes, uses GPT-OSS 120B on Together AI for multi-chain roleplay, curates automatically.

Usage:
    TOGETHER_API_KEY=8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84 \
    python3 -m karl.v7.full_factory --turns 30 --batches 4

    python3 -m karl.v7.full_factory --dry-run
    python3 -m karl.v7.full_factory --turns 40 --batches 6 --interval 45

What it does:
  1. Discovers available tmux panes (local + remote)
  2. Assigns projects from goal library to panes (round-robin per batch)
  3. For each batch:
     a. Kills existing sessions in target panes
     b. Launches Claude Code with --dangerously-skip-permissions
     c. Injects rich prompt (goal + constraints)
     d. Starts V7.2 driver (GPT-5.4-mini multi-chain roleplay)
     e. Waits for all sessions to complete
  4. Curates all session logs into training data
  5. Exports v7-factory-full.jsonl

Output: ~/Desktop/karl/v7-factory-full.jsonl
Logs:   ~/Desktop/karl/v7-factory-logs/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Project Goal Library (60 diverse projects) ──────────────────────────

GOAL_LIBRARY = [
    # Rust projects
    {
        "goal": "Build a Rust CLI that reads NATS JetStream events from MESH_EVENTS and renders them in a colored terminal table with filtering by machine name and event type. Use clap for args, colored for output, and async-nats for the NATS client.",
        "project_dir": "~/Desktop/factory/nats-event-viewer",
        "constraints": ["Keep in mind NATS runs on localhost:4222.", "Make sure it handles NATS being offline gracefully."],
    },
    {
        "goal": "Build a Rust HTTP service using axum that classifies prompts into 10 inscription categories based on keyword analysis. Include /classify, /route, /stats endpoints with serde JSON handling.",
        "project_dir": "~/Desktop/factory/inscription-router",
        "constraints": ["Make sure cargo build passes with no warnings.", "Don't add unnecessary dependencies."],
    },
    {
        "goal": "Build a Rust binary that watches a JSONL file for new lines, computes reward scores using 5-signal composite, and publishes scored entries to NATS subject karl.scored. Include --backfill flag.",
        "project_dir": "~/Desktop/factory/karl-scorer-rs",
        "constraints": ["Keep in mind the reward engine has 5 signals.", "Don't recompute already-scored entries."],
    },
    {
        "goal": "Build a Rust daemon that manages tmux pane slots across 5 machines via SSH. HTTP API at :9451 with /inject, /read, /status endpoints. Use tokio + axum.",
        "project_dir": "~/Desktop/factory/meshd-lite",
        "constraints": ["Keep in mind machines are on Tailscale IPs.", "Make sure SSH connections have 5s timeout."],
    },
    {
        "goal": "Build a Rust CLI tool that takes a directory of JSONL files, deduplicates entries by content hash, computes TF-IDF similarity between all pairs, and outputs a cleaned JSONL with a similarity_score field.",
        "project_dir": "~/Desktop/factory/dedup-tool",
        "constraints": ["Don't load everything into memory at once.", "Make sure it handles malformed JSON lines gracefully."],
    },
    # Python projects
    {
        "goal": "Build a Python FastAPI service that wraps the MLX cognitive twin at http://100.109.94.124:8100 as a REST API. Endpoints: POST /chat, POST /opinion, POST /drive, GET /health. Use httpx async and Pydantic models.",
        "project_dir": "~/Desktop/factory/twin-api-v2",
        "constraints": ["Keep in mind MLX cold starts take 30s.", "Don't block on MLX calls, use async."],
    },
    {
        "goal": "Build a Python script that connects to Supabase, pulls rows from memory_turns with pagination (limit 1000), computes 4 anticipation scalars, classifies inscriptions, and exports curated training examples as JSONL.",
        "project_dir": "~/Desktop/factory/data-scaler",
        "constraints": ["Keep in mind Supabase URL and key come from env vars.", "Don't pull all rows at once, paginate."],
    },
    {
        "goal": "Build a Python Prefect flow that runs every 6 hours: pulls new prompt logs from Claude Code sessions, extracts correction pairs using signature phrase detection, generates rich prompts, and writes them to an output directory.",
        "project_dir": "~/Desktop/factory/correction-flow",
        "constraints": ["Make sure the flow is idempotent.", "Don't re-process already-seen sessions."],
    },
    {
        "goal": "Build a Python CLI that takes a directory of .wav files, transcribes them using faster-whisper, diarizes speakers, and outputs a structured JSONL with speaker labels and timestamps.",
        "project_dir": "~/Desktop/factory/audio-pipeline",
        "constraints": ["Keep in mind faster-whisper needs the model downloaded first.", "Make sure it handles stereo and mono."],
    },
    {
        "goal": "Build a Python service that monitors 5 Mac machines via SSH, collects CPU/memory/disk metrics every 30 seconds, stores in SQLite, and exposes a /metrics endpoint for Grafana scraping.",
        "project_dir": "~/Desktop/factory/mesh-metrics",
        "constraints": ["Keep in mind machines are on Tailscale.", "Make sure SSH has ConnectTimeout=5."],
    },
    {
        "goal": "Build a Python script that reads KARL trajectories.jsonl, groups by skill domain, computes per-domain reward statistics, generates a markdown report with tables and distribution plots using matplotlib, and saves to report.md.",
        "project_dir": "~/Desktop/factory/karl-report",
        "constraints": ["Make sure matplotlib uses Agg backend for headless.", "Don't require any external databases."],
    },
    {
        "goal": "Build a Python WebSocket server that receives real-time body pose data (27 joints), classifies gestures (reach, drop, flow, pulse, freeze), and broadcasts gesture events to connected clients.",
        "project_dir": "~/Desktop/factory/gesture-ws",
        "constraints": ["Keep in mind pose data arrives at 30fps.", "Don't buffer more than 1 second of frames."],
    },
    # Next.js / TypeScript projects
    {
        "goal": "Build a Next.js dashboard that polls meshd health endpoints across 5 Macs every 10 seconds and displays real-time status cards with colored dots per service. Dark theme, shadcn/ui components. Server-side polling to avoid CORS.",
        "project_dir": "~/Desktop/factory/mesh-pulse-v2",
        "constraints": ["Keep in mind machines are on Tailscale IPs.", "Make sure it works when some machines are offline."],
    },
    {
        "goal": "Build a Next.js app that displays a searchable, filterable table of KARL trajectories loaded from a JSONL API endpoint. Include reward distribution chart, skill domain filters, and trajectory detail modal.",
        "project_dir": "~/Desktop/factory/karl-explorer",
        "constraints": ["Make sure the table handles 3000+ rows without lag.", "Use server components for data loading."],
    },
    {
        "goal": "Build a Next.js landing page for a consulting brand called Grand Diomande. Include hero section with animated gradient, services grid, testimonials carousel, and a Cal.com booking embed. Dark theme, custom typography.",
        "project_dir": "~/Desktop/factory/consulting-site",
        "constraints": ["Make sure Tailwind classes are used, not inline styles.", "Don't use generic fonts like Arial or Inter."],
    },
    # SwiftUI / iOS projects
    {
        "goal": "Build a SwiftUI iOS app called MeshMonitor that displays the health of 5 Mac mesh nodes. Each node shows name, IP, uptime, and service status. Pull data from meshd :9451/health. Add pull-to-refresh and a Settings screen.",
        "project_dir": "~/Desktop/factory/MeshMonitorApp",
        "constraints": ["Keep in mind this is SwiftUI, not UIKit.", "Don't force-unwrap optionals."],
    },
    {
        "goal": "Build a SwiftUI iOS app that records voice memos, transcribes them locally using Apple Speech framework, and displays a searchable list of transcriptions with timestamps.",
        "project_dir": "~/Desktop/factory/VoiceMemoApp",
        "constraints": ["Make sure to request microphone permission.", "Don't block the main thread during transcription."],
    },
    # Mixed / infrastructure projects
    {
        "goal": "Build a Docker Compose stack with: a Python FastAPI backend, a Next.js frontend, a PostgreSQL database, and an nginx reverse proxy. The backend serves /api/health and /api/data. The frontend calls the backend via nginx.",
        "project_dir": "~/Desktop/factory/fullstack-compose",
        "constraints": ["Make sure all services have health checks.", "Don't hardcode ports, use environment variables."],
    },
    {
        "goal": "Build a GitHub Actions CI/CD pipeline for a Rust project: lint with clippy, test with cargo test, build release binary, upload as artifact, and deploy to a server via SSH on push to main.",
        "project_dir": "~/Desktop/factory/rust-cicd",
        "constraints": ["Make sure secrets are never logged.", "Don't use deprecated actions."],
    },
    {
        "goal": "Build a Python CLI tool that takes a GitHub repository URL, clones it, analyzes the codebase structure (file types, LOC, dependencies), and outputs a structured JSON report with architecture insights.",
        "project_dir": "~/Desktop/factory/repo-analyzer",
        "constraints": ["Make sure it handles private repos with SSH keys.", "Don't clone into the current directory."],
    },
    # ── Batch 5: Mesh & Infrastructure ──
    {
        "goal": "Build a Rust service that bridges NATS JetStream events to Supabase Realtime. Subscribe to MESH_EVENTS on :4222, transform events into Supabase inserts on the mesh_events table. Include backpressure handling.",
        "project_dir": "~/Desktop/factory/nats-supabase-bridge",
        "constraints": ["Keep in mind NATS is on localhost:4222.", "Supabase URL and key come from env vars."],
    },
    {
        "goal": "Build a Python script that reads all LaunchAgent plist files on Mac1, checks their status via launchctl, and outputs a health dashboard showing which agents are running, failed, or disabled.",
        "project_dir": "~/Desktop/factory/launchagent-health",
        "constraints": ["Keep in mind plists are in ~/Library/LaunchAgents/.", "Make sure it handles permission errors gracefully."],
    },
    {
        "goal": "Build a Rust CLI that SSH-tunnels to Mac5 and queries the MLX server at :8100 with a prompt, returning the response. Include --model flag to select adapter and --temperature flag.",
        "project_dir": "~/Desktop/factory/mlx-remote-query",
        "constraints": ["Keep in mind Mac5 is at 100.109.94.124.", "Use ssh alias mac5, not raw IP."],
    },
    {
        "goal": "Build a Python Prefect flow that runs every hour: checks disk usage on all 5 Macs via SSH, alerts if any disk is above 85%, and logs metrics to Supabase.",
        "project_dir": "~/Desktop/factory/disk-monitor-flow",
        "constraints": ["Keep in mind machines are on Tailscale.", "Don't store SSH keys in code."],
    },
    {
        "goal": "Build a Next.js app that visualizes KARL trajectory data as a force-directed graph. Nodes are sessions, edges are skill domain connections. Color by reward score. Use d3-force for layout.",
        "project_dir": "~/Desktop/factory/trajectory-graph",
        "constraints": ["Make sure it handles 3000+ trajectories without lag.", "Use server components for data loading."],
    },
    # ── Batch 6: ML & Training ──
    {
        "goal": "Build a Python script that takes a JSONL file of SFT training examples, computes token length distribution, filters examples exceeding max_seq_len, deduplicates by prompt hash, and exports cleaned train/eval splits.",
        "project_dir": "~/Desktop/factory/sft-data-cleaner",
        "constraints": ["Don't load everything into memory.", "Make sure the splits are deterministic with a seed."],
    },
    {
        "goal": "Build a Python CLI that evaluates a LoRA adapter by loading it with MLX, generating responses to 20 test prompts, and scoring them with a style validator. Output a markdown report.",
        "project_dir": "~/Desktop/factory/adapter-eval",
        "constraints": ["Keep in mind MLX adapters need adapter_config.json + adapters.safetensors.", "Don't run on Mac1."],
    },
    {
        "goal": "Build a Rust binary that implements a 5-signal reward engine for scoring agent trajectories. Signals: outcome, process, efficiency, verification, consistency. Output composite score to stdout as JSON.",
        "project_dir": "~/Desktop/factory/reward-engine-rs",
        "constraints": ["Make sure all scores normalize to [0, 1].", "Use serde for JSON serialization."],
    },
    {
        "goal": "Build a Python script that reads correction pairs from a JSONL file, clusters them by failure mode using TF-IDF + KMeans, and generates a report showing the top 10 correction patterns with examples.",
        "project_dir": "~/Desktop/factory/correction-analyzer",
        "constraints": ["Don't use heavy ML frameworks, just scikit-learn.", "Make sure it handles empty clusters."],
    },
    {
        "goal": "Build a Python FastAPI service that serves as a prompt router. Given a prompt, it classifies the domain (rust, python, swift, infra, ml) using keyword analysis, and returns the recommended agent + machine to handle it.",
        "project_dir": "~/Desktop/factory/prompt-router",
        "constraints": ["Keep in mind Mac1=orchestrator, Mac4=compute, Mac5=ML.", "Return JSON with confidence scores."],
    },
    # ── Batch 7: iOS & SwiftUI ──
    {
        "goal": "Build a SwiftUI iOS app that connects to meshd at :9451 via WebSocket and displays real-time agent slot status. Each slot shows: agent name, status (idle/active/error), and last activity timestamp.",
        "project_dir": "~/Desktop/factory/MeshSlotViewer",
        "constraints": ["Use URLSessionWebSocketTask, not a third-party library.", "Handle reconnection gracefully."],
    },
    {
        "goal": "Build a SwiftUI iOS app called TrajectoryViewer that loads KARL trajectories from a REST API, displays them in a list with reward score badges, and shows detail view with tool call timeline.",
        "project_dir": "~/Desktop/factory/TrajectoryViewer",
        "constraints": ["Use async/await for network calls.", "Make sure the list is lazy-loaded for performance."],
    },
    {
        "goal": "Build a SwiftUI widget that shows the current mesh health status — green/yellow/red dot per machine, refreshed every 15 minutes via WidgetKit timeline.",
        "project_dir": "~/Desktop/factory/MeshHealthWidget",
        "constraints": ["Keep in mind WidgetKit has strict memory limits.", "Use App Groups for shared data."],
    },
    {
        "goal": "Build a SwiftUI iOS app that records audio, sends it to a local Whisper endpoint for transcription, and displays the result with timestamps. Include a history list persisted in SwiftData.",
        "project_dir": "~/Desktop/factory/WhisperRecorder",
        "constraints": ["Request microphone permission properly.", "Don't block the main thread."],
    },
    {
        "goal": "Build a SwiftUI iOS app that displays a kanban board for KARL training tasks. Columns: todo, in-progress, done. Cards show task name, assigned machine, and ETA. Pull data from Supabase.",
        "project_dir": "~/Desktop/factory/KARLKanban",
        "constraints": ["Use drag-and-drop for moving cards.", "Sync changes back to Supabase in real-time."],
    },
    # ── Batch 8: Creative & Content ──
    {
        "goal": "Build a Next.js app that generates social media post variants. Input: a topic and tone. Output: 5 variants for Twitter, LinkedIn, Instagram. Use a local LLM endpoint at :8100.",
        "project_dir": "~/Desktop/factory/content-generator",
        "constraints": ["Keep in mind MLX on Mac5:8100.", "Don't hardcode the model name."],
    },
    {
        "goal": "Build a Python script that takes a video file, extracts frames at 1fps, describes each frame using a vision model API, and generates a shot-by-shot breakdown as markdown.",
        "project_dir": "~/Desktop/factory/video-breakdown",
        "constraints": ["Use ffmpeg for frame extraction.", "Don't load all frames into memory at once."],
    },
    {
        "goal": "Build a Next.js portfolio site with: animated hero, project grid with hover effects, about section with timeline, and contact form. Dark theme with custom typography and gradient mesh backgrounds.",
        "project_dir": "~/Desktop/factory/portfolio-site",
        "constraints": ["Don't use Arial or Inter.", "Make sure animations are CSS-only, not JS."],
    },
    {
        "goal": "Build a Python script that downloads the latest 10 posts from an RSS feed, summarizes each using a local LLM, and generates a newsletter draft as HTML with inline styles.",
        "project_dir": "~/Desktop/factory/newsletter-gen",
        "constraints": ["Use feedparser for RSS.", "Make the HTML email-client compatible."],
    },
    {
        "goal": "Build a Rust CLI that takes a markdown file and converts it to a beautifully formatted PDF using printpdf crate. Support headings, code blocks, lists, and images.",
        "project_dir": "~/Desktop/factory/md-to-pdf",
        "constraints": ["Don't use external binaries like wkhtmltopdf.", "Make sure code blocks have syntax highlighting."],
    },
    # ── Batch 9: DevOps & Automation ──
    {
        "goal": "Build a Python script that automates Xcode project creation: creates a new SwiftUI project from a template, sets bundle ID, adds entitlements, and runs initial build to verify.",
        "project_dir": "~/Desktop/factory/xcode-bootstrap",
        "constraints": ["Use xcodebuild CLI, not Xcode GUI.", "Make sure code signing is set to automatic."],
    },
    {
        "goal": "Build a Rust daemon that watches a directory for new JSONL files, validates each line against a schema, and moves valid files to a processed/ directory. Invalid files go to errors/.",
        "project_dir": "~/Desktop/factory/jsonl-validator",
        "constraints": ["Use notify crate for file watching.", "Don't hold file locks longer than necessary."],
    },
    {
        "goal": "Build a Docker Compose stack with Prometheus, Grafana, and a custom Python exporter that collects metrics from meshd :9451 on all 5 Macs. Pre-configured dashboards for agent slot utilization.",
        "project_dir": "~/Desktop/factory/mesh-monitoring",
        "constraints": ["Make sure Grafana provisions dashboards from JSON on startup.", "Use bridge network mode."],
    },
    {
        "goal": "Build a Python script that reads git log from all projects in ~/Desktop/, computes daily commit frequency per project, and generates a heatmap visualization saved as PNG.",
        "project_dir": "~/Desktop/factory/commit-heatmap",
        "constraints": ["Use matplotlib with Agg backend.", "Handle repos with no commits gracefully."],
    },
    {
        "goal": "Build a Rust CLI that manages tmux sessions declaratively from a YAML config file. Define sessions, windows, panes, and startup commands. Apply creates/destroys sessions to match the config.",
        "project_dir": "~/Desktop/factory/tmux-declarative",
        "constraints": ["Don't kill sessions not in the config.", "Make sure it handles already-running sessions."],
    },
    # ── Batch 10: Data & Analytics ──
    {
        "goal": "Build a Python script that connects to Supabase, reads the claude_prompts table (100K+ rows), computes prompt length distribution, categorizes by project, and exports a summary CSV.",
        "project_dir": "~/Desktop/factory/prompt-analytics",
        "constraints": ["Paginate with limit/offset, don't pull all at once.", "Keep in mind Supabase URL comes from env."],
    },
    {
        "goal": "Build a Python CLI that takes two JSONL files (train.jsonl and baseline.jsonl), computes perplexity on each, and reports the delta with confidence intervals.",
        "project_dir": "~/Desktop/factory/perplexity-compare",
        "constraints": ["Use numpy for statistics, not pandas.", "Make sure it handles empty files."],
    },
    {
        "goal": "Build a Rust binary that reads NATS JetStream messages, computes a sliding window average of message rate per subject, and exposes a Prometheus-compatible /metrics endpoint.",
        "project_dir": "~/Desktop/factory/nats-metrics-exporter",
        "constraints": ["Keep in mind NATS is on :4222.", "Use a 60-second sliding window."],
    },
    {
        "goal": "Build a Python script that reads all memory files from ~/.claude/projects/-Users-mohameddiomande/memory/, extracts key facts as structured JSON, and builds a knowledge graph stored in SQLite.",
        "project_dir": "~/Desktop/factory/memory-graph",
        "constraints": ["Parse markdown frontmatter with yaml.", "Don't overwrite existing graph, append new facts."],
    },
    {
        "goal": "Build a Next.js dashboard that shows Supabase table sizes, row counts, and RLS status for all 141 tables. Include search, sort by size, and a one-click RLS toggle.",
        "project_dir": "~/Desktop/factory/supabase-inspector",
        "constraints": ["Use Supabase management API, not direct SQL.", "Make sure RLS toggle confirms before acting."],
    },
    # ── Batch 11: Security & Testing ──
    {
        "goal": "Build a Python script that scans all .env files in ~/Desktop/ recursively, identifies exposed API keys and secrets by pattern matching, and generates a security report listing each finding with file path and line number.",
        "project_dir": "~/Desktop/factory/secret-scanner",
        "constraints": ["Don't print the actual secret values.", "Check for at least 10 key patterns (AWS, Supabase, OpenAI, etc.)."],
    },
    {
        "goal": "Build a Rust CLI that generates load test traffic against an HTTP endpoint. Configurable: concurrency, request count, method, headers, body. Output: p50/p95/p99 latency, errors, throughput.",
        "project_dir": "~/Desktop/factory/http-loadtest",
        "constraints": ["Use tokio for async requests.", "Don't panic on connection errors."],
    },
    {
        "goal": "Build a Python script that reads a pytest test suite, generates a test coverage matrix showing which functions are tested, identifies untested public functions, and outputs a gap analysis.",
        "project_dir": "~/Desktop/factory/coverage-analyzer",
        "constraints": ["Use ast module to parse Python files.", "Don't actually run the tests."],
    },
    {
        "goal": "Build a Python CLI that fuzzes a REST API endpoint by generating random but schema-valid payloads from an OpenAPI spec. Report crashes, 500 errors, and unexpected responses.",
        "project_dir": "~/Desktop/factory/api-fuzzer",
        "constraints": ["Use hypothesis for property-based generation.", "Respect rate limits with a --delay flag."],
    },
    {
        "goal": "Build a Rust binary that validates all YAML and JSON config files in a directory tree, reports syntax errors with file path and line number, and optionally auto-formats them.",
        "project_dir": "~/Desktop/factory/config-validator",
        "constraints": ["Support .yaml, .yml, .json, .toml.", "Don't modify files unless --fix flag is passed."],
    },
    # ── Batch 12: Cross-Domain ──
    {
        "goal": "Build a Python script that implements a simple RAG pipeline: read a directory of markdown files, chunk them, compute embeddings using sentence-transformers, store in FAISS, and answer questions via similarity search + LLM.",
        "project_dir": "~/Desktop/factory/mini-rag",
        "constraints": ["Use all-MiniLM-L6-v2 for embeddings.", "Keep chunks under 512 tokens."],
    },
    {
        "goal": "Build a Next.js app that implements a Pomodoro timer with: customizable work/break durations, session history stored in localStorage, daily streaks, and a minimal dark UI with animated countdown ring.",
        "project_dir": "~/Desktop/factory/pomodoro-app",
        "constraints": ["Use CSS for the countdown ring animation.", "Make sure it works offline."],
    },
    {
        "goal": "Build a Rust HTTP server that implements a URL shortener. POST /shorten with a URL returns a short code. GET /:code redirects. Store in an in-memory HashMap with optional SQLite persistence.",
        "project_dir": "~/Desktop/factory/url-shortener",
        "constraints": ["Use axum for the HTTP server.", "Make short codes 6 characters alphanumeric."],
    },
    {
        "goal": "Build a Python script that implements a simple task queue with Redis. Producer pushes JSON tasks, worker consumes and processes them. Include retry logic, dead letter queue, and basic metrics.",
        "project_dir": "~/Desktop/factory/task-queue",
        "constraints": ["Use redis-py, not celery.", "Handle Redis connection failures gracefully."],
    },
    {
        "goal": "Build a SwiftUI macOS app that sits in the menu bar and shows the current git branch + status for a configured project directory. Click to see recent commits. Use NSStatusItem.",
        "project_dir": "~/Desktop/factory/GitMenuBar",
        "constraints": ["Use Process/NSTask to run git commands.", "Refresh every 30 seconds."],
    },
]


# ── Pane Discovery ───────────────────────────────────────────────────────

@dataclass
class Pane:
    machine: str
    session: str
    pane_id: str
    label: str

    @property
    def full_id(self):
        return f"{self.session}:{self.pane_id}"


def discover_panes() -> list[Pane]:
    """Find all available tmux panes (local + SSH)."""
    panes = []

    # Local panes
    try:
        r = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{session_name}:#{window_index}.#{pane_index}"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if not line:
                    continue
                session = line.split(":")[0]
                # Skip the pane running this script (opencode)
                if "opencode" in session.lower():
                    continue
                panes.append(Pane("mac1", session, line.split(":")[-1] if ":" in line else "0.0", session))
    except Exception:
        pass

    # Remote panes (Mac2-5) — check if SSH works
    for machine in ["mac2", "mac4", "mac5"]:
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=3", machine,
                 "tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index}' 2>/dev/null"],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.strip().split("\n"):
                    if line.strip():
                        session = line.split(":")[0]
                        panes.append(Pane(machine, session, line.split(":")[-1], f"{machine}:{session}"))
        except Exception:
            continue

    return panes


# ── Session Management ───────────────────────────────────────────────────

def kill_pane(pane: Pane):
    """Kill whatever's running in a pane."""
    full = pane.full_id
    if pane.machine in ("mac1", "localhost"):
        subprocess.run(["tmux", "send-keys", "-t", full, "C-c", "C-c"], timeout=3, capture_output=True)
        time.sleep(0.5)
        subprocess.run(["tmux", "send-keys", "-t", full, "/exit", "Enter"], timeout=3, capture_output=True)
        time.sleep(1)
        subprocess.run(["tmux", "send-keys", "-t", full, "exit", "Enter"], timeout=3, capture_output=True)
        time.sleep(2)
        subprocess.run(["tmux", "respawn-pane", "-t", full, "-k"], timeout=3, capture_output=True)
        time.sleep(2)
    else:
        ssh = ["ssh", "-o", "ConnectTimeout=5", pane.machine]
        subprocess.run(ssh + ["tmux", "send-keys", "-t", full, "C-c", "C-c"], timeout=10, capture_output=True)
        time.sleep(1)


def kill_driver_processes_for_pane(pane: Pane):
    """Terminate any lingering local driver wrappers for a pane before reuse."""
    pattern = f"karl.v7.driver {pane.machine} {pane.full_id}"
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
    except Exception:
        return

    pids = []
    for line in r.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)

    for pid in pids:
        subprocess.run(["kill", "-TERM", str(pid)], timeout=5, capture_output=True)
    if pids:
        time.sleep(2)
        for pid in pids:
            subprocess.run(["kill", "-KILL", str(pid)], timeout=5, capture_output=True)


def launch_claude(pane: Pane, project_dir: str):
    """Launch Claude Code in the pane."""
    full = pane.full_id
    expanded = os.path.expanduser(project_dir)
    os.makedirs(expanded, exist_ok=True)
    cmd = f"cd {project_dir} && claude --dangerously-skip-permissions"

    if pane.machine in ("mac1", "localhost"):
        subprocess.run(["tmux", "send-keys", "-t", full, cmd, "Enter"], timeout=5, capture_output=True)
    else:
        ssh = ["ssh", "-o", "ConnectTimeout=5", pane.machine]
        subprocess.run(ssh + [f"tmux send-keys -t {full} '{cmd}' Enter"], timeout=10, capture_output=True)
    time.sleep(15)


def inject_prompt(pane: Pane, prompt: str):
    """Inject a prompt into a Claude Code session."""
    full = pane.full_id
    if pane.machine in ("mac1", "localhost"):
        subprocess.run(["tmux", "load-buffer", "-"], input=prompt.encode(), timeout=5, capture_output=True)
        subprocess.run(["tmux", "paste-buffer", "-t", full], timeout=5, capture_output=True)
        time.sleep(1)
        subprocess.run(["tmux", "send-keys", "-t", full, "Enter"], timeout=5, capture_output=True)
    else:
        escaped = prompt.replace("'", "'\\''")
        ssh = ["ssh", "-o", "ConnectTimeout=5", pane.machine]
        cmd = f"printf '%s' '{escaped}' | tmux load-buffer - && tmux paste-buffer -t {full} && tmux send-keys -t {full} Enter"
        subprocess.run(ssh + [cmd], timeout=15, capture_output=True)


def start_driver(pane: Pane, goal: str, project_dir: str, turns: int, interval: int) -> subprocess.Popen:
    """Start V7.2 driver in background with Together AI env vars."""
    full = pane.full_id
    safe_goal = goal[:200].replace('"', '\\"').replace("'", "")
    together_key = os.environ.get("TOGETHER_API_KEY", "8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84")
    slug = project_dir.split("/")[-1]
    driver_cmd = (
        f"sleep 90 && cd ~/Desktop/karl && "
        f"TOGETHER_API_KEY={together_key} KARL_ROLEPLAY_BACKEND=together "
        f"python3 -u -m karl.v7.driver "
        f"{pane.machine} {full} "
        f'--goal "{safe_goal}" '
        f"--project-dir {project_dir} "
        f"--turns {turns} --interval {interval} "
        f"> ~/Desktop/karl/v7-factory-logs/{slug}.log 2>&1"
    )
    return subprocess.Popen(
        ["bash", "-c", driver_cmd],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def clear_session_states():
    """Remove persisted state only after the prior batch is fully done."""
    import glob

    for pattern in ("~/.karl-sessions/*.json", "~/.karl-sessions/*.tmp"):
        for fpath in glob.glob(os.path.expanduser(pattern)):
            try:
                os.remove(fpath)
            except FileNotFoundError:
                continue


# ── Batch Orchestration ──────────────────────────────────────────────────

def build_rich_prompt(goal_entry: dict) -> str:
    """Build a rich prompt with constraints baked in."""
    parts = [goal_entry["goal"]]
    if goal_entry.get("constraints"):
        parts.append(" ".join(goal_entry["constraints"]))
    parts.append("Execute directly. Don't explain unless asked. Make sure you test before saying it's done.")
    return "\n\n".join(parts)


def run_batch(
    panes: list[Pane],
    goals: list[dict],
    turns: int,
    interval: int,
    dry_run: bool,
    batch_num: int,
):
    """Run a single batch: assign goals to panes, spawn sessions, wait."""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}: {len(goals)} projects -> {len(panes)} panes")
    print(f"{'='*60}")

    assignments = []
    driver_procs: list[tuple[Pane, str, subprocess.Popen]] = []
    for i, goal in enumerate(goals):
        pane = panes[i % len(panes)]
        assignments.append((pane, goal))

    for i, (pane, goal_entry) in enumerate(assignments):
        project_dir = goal_entry["project_dir"]
        goal = goal_entry["goal"]
        slug = project_dir.split("/")[-1]

        print(f"\n[{i+1}/{len(assignments)}] {slug} -> {pane.label}")
        print(f"  Goal: {goal[:80]}...")

        if dry_run:
            print(f"  [DRY RUN] Would spawn")
            continue

        # Kill, launch, inject, start driver
        print(f"  Killing lingering drivers...", flush=True)
        kill_driver_processes_for_pane(pane)

        print(f"  Killing pane...", flush=True)
        kill_pane(pane)

        print(f"  Launching Claude...", flush=True)
        launch_claude(pane, project_dir)

        rich_prompt = build_rich_prompt(goal_entry)
        print(f"  Injecting prompt ({len(rich_prompt)} chars)...", flush=True)
        inject_prompt(pane, rich_prompt)

        print(f"  V7.2 driver starts in 90s ({turns} turns, {interval}s interval)...", flush=True)
        driver_proc = start_driver(pane, goal, project_dir, turns, interval)
        driver_procs.append((pane, slug, driver_proc))

        if i < len(assignments) - 1:
            time.sleep(5)

    if dry_run:
        return

    # Wait for the specific drivers from this batch. Their runtime is longer than
    # turns*interval because prompt generation and Claude's own work add overhead.
    max_wait = 90 + (turns * (interval + 25)) + 300
    print(f"\n  Waiting for batch drivers to complete (up to ~{max_wait//60}min)...")

    elapsed = 0
    poll_interval = 30
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        active = [(pane, slug, proc) for pane, slug, proc in driver_procs if proc.poll() is None]
        mins = elapsed // 60
        print(f"  [{mins}min] {len(active)} batch drivers still running", flush=True)

        if not active and elapsed >= 120:
            print("  All batch drivers finished.", flush=True)
            break
    else:
        print("  Batch exceeded max wait. Terminating lingering drivers...", flush=True)
        for pane, slug, proc in driver_procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(5)
        for pane, slug, proc in driver_procs:
            if proc.poll() is None:
                proc.kill()

    time.sleep(10)


# ── Curation ─────────────────────────────────────────────────────────────

def curate_all():
    """Run curation on all factory logs and export combined dataset."""
    import glob
    from .curate_training_data import load_v7_logs, filter_entries, format_training_pairs, export, analyze

    print(f"\n{'='*60}")
    print(f"CURATION")
    print(f"{'='*60}")

    entries = load_v7_logs()
    print(f"Loaded {len(entries)} log entries")

    filtered = filter_entries(entries)
    pairs = format_training_pairs(filtered)
    output = os.path.expanduser("~/Desktop/karl/v7-factory-full.jsonl")
    export(pairs, output)
    analyze(pairs)
    return len(pairs)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KARL V7 Full Factory")
    parser.add_argument("--turns", type=int, default=50, help="Turns per session")
    parser.add_argument("--interval", type=int, default=45, help="Seconds between turns")
    parser.add_argument("--batches", type=int, default=12, help="Number of batches (60 projects / 5 panes = 12)")
    parser.add_argument("--start-project", type=int, default=1,
                        help="1-based project index to start from")
    parser.add_argument("--end-project", type=int, default=0,
                        help="1-based inclusive project index to stop at (0 = through the end)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--projects-per-batch", type=int, default=0,
                        help="Projects per batch (0 = auto from pane count)")
    args = parser.parse_args()

    # Setup
    log_dir = os.path.expanduser("~/Desktop/karl/v7-factory-logs")
    os.makedirs(log_dir, exist_ok=True)

    # Clear old session states
    clear_session_states()

    print("KARL V7 Full Factory")
    print("=" * 60)
    print(f"Goal library: {len(GOAL_LIBRARY)} projects")
    print(f"Turns: {args.turns}, Interval: {args.interval}s")
    print(f"Batches: {args.batches}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Discover panes
    panes = discover_panes()
    print(f"Discovered {len(panes)} panes:")
    for p in panes:
        print(f"  {p.label} ({p.machine})")

    if not panes:
        print("ERROR: No panes found. Create tmux sessions first.")
        sys.exit(1)

    # Filter to usable panes (skip opencode since it's running this script)
    usable = [p for p in panes if "opencode" not in p.session.lower()]
    if not usable:
        print("ERROR: No usable panes (all are opencode).")
        sys.exit(1)

    print(f"\nUsable panes: {len(usable)}")
    per_batch = args.projects_per_batch or len(usable)

    # Assign goals to batches
    goals = list(GOAL_LIBRARY)
    start_idx = max(0, args.start_project - 1)
    end_idx = len(goals) if args.end_project <= 0 else min(len(goals), args.end_project)
    goals = goals[start_idx:end_idx]
    total_projects = min(len(goals), per_batch * args.batches)
    print(f"Project range: {start_idx + 1}-{start_idx + len(goals)}")
    print(f"Total projects to run: {total_projects} ({args.batches} batches x {per_batch}/batch)")
    print()

    t0 = time.time()
    total_pairs = 0

    for batch in range(args.batches):
        start = batch * per_batch
        end = min(start + per_batch, len(goals))
        if start >= len(goals):
            print(f"\nBatch {batch+1}: No more projects. Done.")
            break

        batch_goals = goals[start:end]
        run_batch(usable, batch_goals, args.turns, args.interval, args.dry_run, batch + 1)

        if not args.dry_run:
            clear_session_states()

    if not args.dry_run:
        total_pairs = curate_all()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FACTORY COMPLETE")
    print(f"{'='*60}")
    print(f"Projects run: {total_projects}")
    print(f"Training pairs: {total_pairs}")
    print(f"Wall time: {elapsed/60:.1f} min")
    print(f"Output: ~/Desktop/karl/v7-factory-full.jsonl")


if __name__ == "__main__":
    main()
