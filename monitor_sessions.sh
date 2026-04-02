#!/bin/bash
# Monitor all 3 twin-driven sessions continuously
# Logs to ~/Desktop/karl/session-monitor.log

LOG=~/Desktop/karl/session-monitor.log
echo "=== Session Monitor Started $(date) ===" >> "$LOG"

while true; do
    TS=$(date +"%H:%M:%S")
    echo "" >> "$LOG"
    echo "[$TS] ===========================================" >> "$LOG"

    for pane in agent-codex:1.1 agent-claude2:1.1 agent-opencode:1.1; do
        STATUS=$(tmux capture-pane -t "$pane" -p 2>/dev/null | grep -v "^$" | tail -5)
        # Detect state
        if echo "$STATUS" | grep -q "Running\|Synthesizing\|Crunched\|Elucidating\|Razzle\|Boogie\|Roosting\|Doodling\|Perambulating"; then
            STATE="WORKING"
        elif echo "$STATUS" | grep -qE "^❯\s*$"; then
            STATE="IDLE"
        elif echo "$STATUS" | grep -q "zsh\|bash"; then
            STATE="SHELL (session dead)"
        else
            STATE="UNKNOWN"
        fi

        LAST_LINE=$(tmux capture-pane -t "$pane" -p 2>/dev/null | grep -v "^$" | grep -v "TOKEN STACK\|bypass\|shift+tab\|ctrl+" | tail -1)
        echo "[$TS] $pane | $STATE | $LAST_LINE" >> "$LOG"
    done

    # Also log twin driver activity
    for f in /tmp/twin-mesh-health.log /tmp/twin-twin-api.log /tmp/twin-inscription.log; do
        LAST=$(tail -1 "$f" 2>/dev/null)
        echo "[$TS] DRIVER $(basename $f): $LAST" >> "$LOG"
    done

    sleep 30
done
