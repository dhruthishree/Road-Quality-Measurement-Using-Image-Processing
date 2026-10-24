"""
Automated Civil Engineering Audit Report & Data Export Generator
Generates standalone HTML executive reports, CSV segment tables, and JSON datasets.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class ReportGenerator:
    """Creates engineering audit reports with findings, metrics, and actionable recommendations."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_csv_reports(self, summary_data: Dict, unique_defects: List[Dict], prefix: str = "road_quality") -> Tuple[str, str]:
        """Exports segments summary CSV and defects catalog CSV."""
        # 1. Segments Summary CSV
        segments = summary_data.get("segments", [])
        seg_csv_path = os.path.join(self.output_dir, f"{prefix}_segments_summary.csv")
        if segments:
            df_seg = pd.DataFrame(segments)
            df_seg.to_csv(seg_csv_path, index=False)
        else:
            df_seg = pd.DataFrame()

        # 2. Defects Catalog CSV
        defect_csv_path = os.path.join(self.output_dir, f"{prefix}_defects_catalog.csv")
        if unique_defects:
            df_defects = pd.DataFrame(unique_defects)
            df_defects.to_csv(defect_csv_path, index=False)
        else:
            df_defects = pd.DataFrame()

        # 3. Overall JSON
        json_path = os.path.join(self.output_dir, f"{prefix}_assessment_summary.json")
        with open(json_path, "w") as f:
            json.dump({
                "summary": {k: v for k, v in summary_data.items() if k != "segments"},
                "segments": segments,
                "defects": unique_defects
            }, f, indent=2)

        return seg_csv_path, defect_csv_path

    def generate_html_report(
        self,
        summary_data: Dict,
        unique_defects: List[Dict],
        route_title: str = "Continuous Road Survey",
        filename: str = "road_quality_audit_report.html"
    ) -> str:
        """Generates an executive HTML report with responsive styling, KPI badges, and charts."""
        report_path = os.path.join(self.output_dir, filename)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        overall_rating = summary_data.get("overall_rating", "Unknown")
        avg_rqi = summary_data.get("overall_average_rqi", 0.0)
        total_dist = summary_data.get("total_survey_distance_m", 0.0)
        total_potholes = summary_data.get("total_unique_potholes", 0)
        total_cracks = summary_data.get("total_unique_cracks", 0)
        total_area = summary_data.get("total_defect_area_m2", 0.0)
        segments = summary_data.get("segments", [])

        # Rating color badge
        badge_color = "#10b981" if overall_rating == "Good" else ("#f59e0b" if overall_rating == "Fair" else "#ef4444")

        # Build segments table rows
        table_rows = []
        for s in segments:
            rating = s["quality_rating"]
            row_badge = "#10b981" if rating == "Good" else ("#f59e0b" if rating == "Fair" else "#ef4444")
            table_rows.append(f"""
            <tr>
                <td style="font-weight: 600;">Seg #{s['segment_id'] + 1}</td>
                <td>{s['chainage_start_m']}m - {s['chainage_end_m']}m</td>
                <td><span style="display:inline-block; padding: 3px 10px; border-radius: 999px; background: {row_badge}22; color: {row_badge}; font-weight:700; border: 1px solid {row_badge}44;">{rating}</span></td>
                <td style="font-weight: 700; font-size: 1.05rem;">{s['rqi_score']}</td>
                <td>{s['pothole_count']}</td>
                <td>{s['crack_count']}</td>
                <td>{s['total_defect_area_m2']} m²</td>
                <td>{s['defect_density_pct']}%</td>
                <td>{s['mean_grade_pct']}%</td>
                <td style="font-size: 0.85rem; color: #475569;">{s['action_recommendation']}</td>
            </tr>
            """)
        segments_tbody = "\n".join(table_rows) if table_rows else "<tr><td colspan='10'>No segments recorded</td></tr>"

        # Build defects table rows
        defect_rows = []
        for d in unique_defects[:25]: # Top 25
            d_type = d.get("type", "unknown").upper()
            defect_rows.append(f"""
            <tr>
                <td style="font-weight: 600;">T-{d.get('track_id', 0):03d}</td>
                <td>{d_type}</td>
                <td>{d.get('chainage_distance_m', 0.0)}m</td>
                <td>{d.get('smoothed_area_m2', 0.0)} m²</td>
                <td>{d.get('smoothed_length_m', 0.0)}m × {d.get('smoothed_width_m', 0.0)}m</td>
                <td><span style="font-weight:700; color:{'#ef4444' if d.get('severity', 1)>=4 else '#f59e0b'};">Severity {d.get('severity', 1)}/5</span></td>
                <td>{int(d.get('confidence', 0.8) * 100)}%</td>
            </tr>
            """)
        defects_tbody = "\n".join(defect_rows) if defect_rows else "<tr><td colspan='7'>No physical defects detected</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Road Quality Audit Report - {route_title}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-card: #273549;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --good: #10b981;
            --fair: #f59e0b;
            --bad: #ef4444;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: var(--bg); color: var(--text-main); padding: 30px; line-height: 1.5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 1.9rem; font-weight: 700; color: #fff; }}
        .header p {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 30px; }}
        .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
        .kpi-title {{ font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 1.85rem; font-weight: 700; color: #fff; }}
        .rating-badge {{ display: inline-block; padding: 6px 16px; border-radius: 999px; font-size: 1.1rem; font-weight: 800; background: {badge_color}22; color: {badge_color}; border: 1.5px solid {badge_color}; }}
        .section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 30px; }}
        .section-title {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 16px; color: #fff; display: flex; align-items: center; gap: 8px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; margin-top: 10px; font-size: 0.92rem; }}
        th {{ background: #162032; padding: 12px 14px; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); }}
        td {{ padding: 12px 14px; border-bottom: 1px solid #28374d; color: #e2e8f0; }}
        tr:hover {{ background: #253347; }}
        .callout {{ background: #1a273b; border-left: 4px solid var(--primary); padding: 16px; border-radius: 6px; margin-bottom: 20px; font-size: 0.95rem; }}
        .footer {{ text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Road Quality Measurement & Surface Audit Report</h1>
                <p>Survey Target: <strong>{route_title}</strong> | Generated on {now_str}</p>
            </div>
            <div>
                <span class="rating-badge">{overall_rating.upper()} CONDITION</span>
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Overall Road Quality Index (RQI)</div>
                <div class="kpi-value" style="color: {badge_color};">{avg_rqi} <span style="font-size: 1rem; color: var(--text-muted);">/ 100</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Surveyed Length</div>
                <div class="kpi-value">{total_dist:.1f} <span style="font-size: 1rem; color: var(--text-muted);">m</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Confirmed Potholes</div>
                <div class="kpi-value" style="color: #ef4444;">{total_potholes}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Crack Formations</div>
                <div class="kpi-value" style="color: #f59e0b;">{total_cracks}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Total Defect Surface Area</div>
                <div class="kpi-value">{total_area:.2f} <span style="font-size: 1rem; color: var(--text-muted);">m²</span></div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Civil Engineering Executive Summary & Assessment</div>
            <div class="callout">
                <strong>Standard Alignment:</strong> Methodology adheres to ASTM D6433 (Pavement Condition Index) and IRC:SP:16 guidelines.
                Perspective homography (IPM) eliminates camera foreshortening to compute true metric damage.
            </div>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                The continuous evaluation over <strong>{total_dist:.1f} meters</strong> classifies this pavement section as <strong>{overall_rating}</strong> (Overall RQI Score: <strong>{avg_rqi}/100</strong>).
                A total of <strong>{len(unique_defects)} unique physical surface defects</strong> were verified across <strong>{len(segments)} discrete chainage segments</strong>.
            </p>
        </div>

        <div class="section">
            <div class="section-title">Segment-by-Segment Continuous Chainage Log (50m Intervals)</div>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Segment</th>
                            <th>Chainage Interval</th>
                            <th>Status</th>
                            <th>RQI</th>
                            <th>Potholes</th>
                            <th>Cracks</th>
                            <th>Defect Area</th>
                            <th>Defect Density</th>
                            <th>Road Grade</th>
                            <th>Action Plan</th>
                        </tr>
                    </thead>
                    <tbody>
                        {segments_tbody}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Physical Defect Catalog & Metric Dimensions</div>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Track ID</th>
                            <th>Defect Type</th>
                            <th>Chainage</th>
                            <th>Metric Area (m²)</th>
                            <th>Length × Width</th>
                            <th>Severity</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {defects_tbody}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Engineering Recommendations & Action Protocol</div>
            <ul style="padding-left: 20px; color: #cbd5e1; line-height: 1.8; font-size: 0.95rem;">
                <li><strong>Priority Maintenance:</strong> Segments classified as <em>Bad</em> must be scheduled for cold-milling and asphalt overlay within 14 business days.</li>
                <li><strong>Drainage & Slope Inspection:</strong> Road segments exhibiting dynamic inclination variance (>2.0% grade change) should be audited for localized water ponding and shoulder scour.</li>
                <li><strong>Preventative Seal:</strong> Moderate fatigue/alligator cracks should receive bituminous emulsion crack sealant before seasonal rains expand cavities into severe potholes.</li>
            </ul>
        </div>

        <div class="footer">
            Continuous Road Quality Measurement System | Computer Vision & Image Processing Pipeline v1.0
        </div>
    </div>
</body>
</html>"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path
