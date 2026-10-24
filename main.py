"""
Main CLI Application: Road Quality Measurement using Image Processing
Continuous road defect detection, road inclination estimation, RQI scoring, and report generation.
"""

import argparse
import os
import sys
import glob
from src.pipeline import RoadQualityPipeline
from src.config import CONFIG

def process_single_video(video_path: str, output_dir: str = "output"):
    """Runs end-to-end evaluation on a single road video."""
    print(f"\n=======================================================")
    print(f"[*] Processing Video: {os.path.basename(video_path)}")
    print(f"=======================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    annotated_video_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")
    
    pipeline = RoadQualityPipeline()
    res = pipeline.process_video(video_path, output_annotated_video_path=annotated_video_path)
    
    summary = res["summary"]
    print(f"[OK] Processed {res['processed_frames']} frames")
    print(f"[+] Total Survey Distance: {summary.get('total_survey_distance_m', 0):.1f} meters")
    print(f"[+] Overall Road Quality Index (RQI): {summary.get('overall_average_rqi', 0)} / 100")
    print(f"[+] Overall Road Condition: {summary.get('overall_rating', 'Unknown').upper()}")
    print(f"[+] Unique Potholes Confirmed: {summary.get('total_unique_potholes', 0)}")
    print(f"[+] Unique Crack Networks: {summary.get('total_unique_cracks', 0)}")
    print(f"[+] Total Defect Surface Area: {summary.get('total_defect_area_m2', 0):.2f} m2")
    print(f"[+] Segments Breakdown: {summary.get('good_segments_count', 0)} Good | {summary.get('fair_segments_count', 0)} Fair | {summary.get('bad_segments_count', 0)} Bad")
    print(f"\n[Artifacts Generated]:")
    print(f"  - Annotated Video: {annotated_video_path}")
    print(f"  - Audit HTML Report: {res['html_report_path']}")
    print(f"  - Segments CSV: {res['segments_csv_path']}")
    print(f"  - Defects Catalog CSV: {res['defects_csv_path']}")

def run_benchmark_suite():
    """Runs batch evaluation across all benchmark routes in dataset/routes."""
    routes_dir = os.path.join(os.path.dirname(__file__), "dataset", "routes")
    video_files = glob.glob(os.path.join(routes_dir, "*", "*.mp4"))
    
    if not video_files:
        print("[!] No benchmark videos found. Running dataset generator first...")
        from dataset.generator.dataset_generator import RoadDatasetGenerator
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset"))
        gen = RoadDatasetGenerator(base_dir)
        gen.generate_all_benchmark_routes()
        video_files = glob.glob(os.path.join(routes_dir, "*", "*.mp4"))

    print(f"\n[+] Found {len(video_files)} benchmark route videos.")
    results = []
    for vf in sorted(video_files):
        base_name = os.path.splitext(os.path.basename(vf))[0]
        pipeline = RoadQualityPipeline()
        res = pipeline.process_video(vf)
        s = res["summary"]
        results.append({
            "route": base_name,
            "distance_m": s.get("total_survey_distance_m", 0),
            "rqi": s.get("overall_average_rqi", 0),
            "condition": s.get("overall_rating", "Unknown"),
            "potholes": s.get("total_unique_potholes", 0),
            "cracks": s.get("total_unique_cracks", 0),
            "defect_area_m2": s.get("total_defect_area_m2", 0)
        })

    # Print summary benchmark table
    print("\n" + "=" * 80)
    print("BENCHMARK VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Route Name':<38} | {'Dist(m)':<8} | {'RQI':<6} | {'Status':<6} | {'Potholes':<8} | {'Cracks':<6} | {'Defect m2':<9}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['route']:<38} | {r['distance_m']:<8.1f} | {r['rqi']:<6.1f} | {r['condition']:<6} | {r['potholes']:<8} | {r['cracks']:<6} | {r['defect_area_m2']:<9.2f}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Continuous Road Quality Measurement using Image Processing")
    parser.add_argument("--video", type=str, help="Path to input video file for road survey analysis")
    parser.add_argument("--benchmark", action="store_true", help="Run automated evaluation across all 6 original benchmark routes")
    parser.add_argument("--web", action="store_true", help="Launch interactive web dashboard")
    parser.add_argument("--output", type=str, default="output", help="Directory to save audit outputs and reports")
    args = parser.parse_args()

    if args.web:
        print("[*] Launching Web Dashboard...")
        from web.app import run_app
        run_app()
    elif args.video:
        process_single_video(args.video, output_dir=args.output)
    elif args.benchmark:
        run_benchmark_suite()
    else:
        # Default behavior: run benchmark validation
        print("[*] No arguments provided. Running benchmark validation suite across original routes...")
        run_benchmark_suite()

if __name__ == "__main__":
    main()
