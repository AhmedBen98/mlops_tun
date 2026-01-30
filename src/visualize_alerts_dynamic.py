"""
Script de visualisation dynamique et interactive des alertes de monitoring.
Génère des dashboards HTML avec graphiques Chart.js pour une analyse visuelle complète.
"""

import os
import json
from pathlib import Path

def load_json_file(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                for key in ['alerts', 'history', 'data', 'entries']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            else:
                return []
    except Exception:
        return []

def aggregate_alerts(alerts_dir):
    alerts = []
    for version in sorted(os.listdir(alerts_dir)):
        version_dir = os.path.join(alerts_dir, version)
        if os.path.isdir(version_dir):
            alert_file = os.path.join(version_dir, 'alerts.json')
            version_alerts = load_json_file(alert_file)
            for a in version_alerts:
                a['version'] = version
            alerts.extend(version_alerts)
    return alerts

def aggregate_drift_history(history_dir):
    history = []
    for version in sorted(os.listdir(history_dir)):
        version_dir = os.path.join(history_dir, version)
        if os.path.isdir(version_dir):
            drift_file = os.path.join(version_dir, 'drift_history.json')
            version_history = load_json_file(drift_file)
            for h in version_history:
                h['version'] = version
            history.extend(version_history)
    return history

def generate_dynamic_html_dashboard(alerts, drift_history, output_path):
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Dynamic Monitoring Dashboard</title>
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f7f7f7; }}
        .container {{ max-width: 1200px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 32px; }}
        h1, h2 {{ color: #1a202c; }}
        .section {{ margin-bottom: 40px; }}
        .alert-table, .drift-table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        .alert-table th, .alert-table td, .drift-table th, .drift-table td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; }}
        .alert-table th, .drift-table th {{ background: #f1f5f9; }}
        .alert-row-critical {{ background: #ffe5e5; }}
        .alert-row-warning {{ background: #fffbe5; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>Dynamic Monitoring Dashboard</h1>
        <div class=\"section\">
            <h2>Drift Alerts ({len(alerts)})</h2>
            <table class=\"alert-table\">
                <tr><th>Timestamp</th><th>Type</th><th>Message</th><th>Level</th><th>Version</th></tr>
                {''.join([
                    f'<tr class="alert-row-{{a.get("level", "")}}"><td>{{a.get("timestamp", "")}}</td><td>{{a.get("type", "")}}</td><td>{{a.get("message", "")}}</td><td>{{a.get("level", "")}}</td><td>{{a.get("version", "")}}</td></tr>'
                    for a in alerts
                ])}
            </table>
        </div>
        <div class=\"section\">
            <h2>Drift History ({len(drift_history)})</h2>
            <table class=\"drift-table\">
                <tr><th>Timestamp</th><th>Drift Score</th><th>Version</th></tr>
                {''.join([
                    f'<tr><td>{{h.get("timestamp", "")}}</td><td>{{h.get("drift_score", "")}}</td><td>{{h.get("version", "")}}</td></tr>'
                    for h in drift_history
                ])}
            </table>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

def main():
    monitoring_dir = Path(__file__).parent.parent / 'monitoring'
    alerts_dir = monitoring_dir / 'alerts'
    history_dir = monitoring_dir / 'history'
    output_path = monitoring_dir / 'dashboard_dynamic.html'

    alerts = aggregate_alerts(str(alerts_dir))
    drift_history = aggregate_drift_history(str(history_dir))

    print(f"Alertes drift: {len(alerts)}")
    print(f"Historique drift: {len(drift_history)} entrées")

    generate_dynamic_html_dashboard(alerts, drift_history, str(output_path))
    print(f"Dashboard généré: {output_path}")

if __name__ == "__main__":
    main()
