import re
import sys
from pathlib import Path

# Fix Windows cp1252 console encoding (crashes on special chars without this)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Path to the pipeline log file
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "pipeline.log"

def analyze_logs():
    print("═" * 60)
    print("                 EDUBOT API LOG ANALYZER")
    print("═" * 60)
    
    if not LOG_FILE.exists():
        print(f"Log file not found at: {LOG_FILE}")
        print("Please start the FastAPI server and make some requests first!")
        print("═" * 60)
        return

    # Regular expression to parse log lines
    log_pattern = re.compile(
        r"Request:\s+(?P<method>\w+)\s+(?P<path>\S+)\s+-\s+Status:\s+(?P<status>\d+)\s+-\s+Duration:\s+(?P<duration>[\d.]+)s"
    )

    total_requests = 0
    status_counts = {}
    path_counts = {}
    durations = []

    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                total_requests += 1
                method = match.group("method")
                path = match.group("path")
                status = match.group("status")
                duration = float(match.group("duration"))

                # Count status codes
                status_counts[status] = status_counts.get(status, 0) + 1

                # Count endpoints
                endpoint = f"{method} {path}"
                path_counts[endpoint] = path_counts.get(endpoint, 0) + 1

                # Collect durations
                durations.append(duration)

    if total_requests == 0:
        print(f"Log file found at {LOG_FILE} but no request log records could be parsed.")
        print("Ensure the middleware has recorded API requests.")
        print("═" * 60)
        return

    avg_duration = sum(durations) / len(durations)
    max_duration = max(durations)
    min_duration = min(durations)

    print(f"📊 Summary Statistics:")
    print(f"  • Total API Requests: {total_requests}")
    print(f"  • Average Latency:    {avg_duration:.4f}s")
    print(f"  • Min / Max Latency:  {min_duration:.4f}s / {max_duration:.4f}s")
    print()

    print("🔑 Status Code Breakdown:")
    for status, count in sorted(status_counts.items()):
        pct = (count / total_requests) * 100
        print(f"  • Status {status}: {count} ({pct:.1f}%)")
    print()

    print("🛣️ Top Endpoint Paths:")
    for endpoint, count in sorted(path_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_requests) * 100
        print(f"  • {endpoint}: {count} ({pct:.1f}%)")
        
    print("═" * 60)

if __name__ == "__main__":
    analyze_logs()
