/**
 * AutoRoad Vision AI - Interactive Dashboard Logic
 * Real-time video streaming, Chart.js continuous quality plotting, and civil engineering analytics.
 */

document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const routeSelect = document.getElementById("routeSelect");
    const btnRunSurvey = document.getElementById("btnRunSurvey");
    const videoUploadInput = document.getElementById("videoUploadInput");
    const streamPlayer = document.getElementById("streamPlayer");
    const videoPlaceholder = document.getElementById("videoPlaceholder");
    const btnViewAnnotated = document.getElementById("btnViewAnnotated");
    const btnViewIPM = document.getElementById("btnViewIPM");

    // KPI Elements
    const valRqi = document.getElementById("valRqi");
    const badgeQualityStatus = document.getElementById("badgeQualityStatus");
    const valDistance = document.getElementById("valDistance");
    const txtSegmentsCount = document.getElementById("txtSegmentsCount");
    const valPotholes = document.getElementById("valPotholes");
    const valPotholeArea = document.getElementById("valPotholeArea");
    const valCracks = document.getElementById("valCracks");
    const valCrackArea = document.getElementById("valCrackArea");
    const valDensity = document.getElementById("valDensity");
    const valTotalDamageArea = document.getElementById("valTotalDamageArea");

    // HUD Elements
    const hudSpeed = document.getElementById("hudSpeed");
    const hudGrade = document.getElementById("hudGrade");
    const hudPitch = document.getElementById("hudPitch");
    const hudSegment = document.getElementById("hudSegment");
    const hudHorizon = document.getElementById("hudHorizon");

    // Tables & Links
    const segmentsTable = document.getElementById("segmentsTable").querySelector("tbody");
    const defectsTable = document.getElementById("defectsTable").querySelector("tbody");
    const linkHtmlReport = document.getElementById("linkHtmlReport");
    const linkSegmentsCsv = document.getElementById("linkSegmentsCsv");
    const linkDefectsCsv = document.getElementById("linkDefectsCsv");

    let currentStreamMode = "annotated";
    let profileChart = null;

    // Initialize Chart.js
    function initChart() {
        const ctx = document.getElementById("roadProfileChart").getContext("2d");
        profileChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: ["0m", "25m", "50m", "75m", "100m"],
                datasets: [
                    {
                        label: "Road Quality Index (RQI: 0-100)",
                        data: [100, 100, 100, 100, 100],
                        borderColor: "#10b981",
                        backgroundColor: "rgba(16, 185, 129, 0.1)",
                        borderWidth: 3,
                        fill: true,
                        tension: 0.35,
                        yAxisID: "y"
                    },
                    {
                        label: "Road Inclination Grade (%)",
                        data: [0, 0.5, 1.2, 0.8, 0],
                        borderColor: "#06b6d4",
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.3,
                        yAxisID: "y1"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#9ca3af", font: { family: "Outfit" } }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.08)" },
                        ticks: { color: "#10b981", font: { family: "JetBrains Mono" } },
                        title: { display: true, text: "RQI Score", color: "#10b981" }
                    },
                    y1: {
                        position: "right",
                        min: -15,
                        max: 15,
                        grid: { drawOnChartArea: false },
                        ticks: { color: "#06b6d4", font: { family: "JetBrains Mono" } },
                        title: { display: true, text: "Grade %", color: "#06b6d4" }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: "#e5e7eb", font: { family: "Outfit", size: 11 } }
                    }
                }
            }
        });
    }

    // Tab Switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });

    // Toggle Stream Mode (Perspective vs IPM BEV)
    btnViewAnnotated.addEventListener("click", () => {
        btnViewAnnotated.classList.add("active");
        btnViewIPM.classList.remove("active");
        currentStreamMode = "annotated";
        refreshStream();
    });

    btnViewIPM.addEventListener("click", () => {
        btnViewIPM.classList.add("active");
        btnViewAnnotated.classList.remove("active");
        currentStreamMode = "ipm";
        refreshStream();
    });

    function refreshStream() {
        const routeName = routeSelect.value;
        if (!routeName) return;
        
        videoPlaceholder.style.display = "none";
        streamPlayer.style.display = "block";
        streamPlayer.src = `/api/video_feed/${routeName}?mode=${currentStreamMode}&t=${Date.now()}`;
    }

    // Run Full Survey
    btnRunSurvey.addEventListener("click", async () => {
        const routeName = routeSelect.value;
        if (!routeName) return;

        btnRunSurvey.innerHTML = `<span class="btn-icon">⏳</span> Analyzing...`;
        btnRunSurvey.disabled = true;

        refreshStream();

        try {
            const resp = await fetch(`/api/run_survey/${routeName}`);
            const data = await resp.json();

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            updateDashboardMetrics(data);
        } catch (err) {
            console.error("Survey execution failed:", err);
            alert("Survey execution failed. Check console.");
        } finally {
            btnRunSurvey.innerHTML = `<span class="btn-icon">▶</span> Run Full Survey`;
            btnRunSurvey.disabled = false;
        }
    });

    // Update Dashboard Metrics and Tables
    function updateDashboardMetrics(data) {
        const s = data.summary;
        const defects = data.unique_defects || [];
        const segments = s.segments || [];

        // 1. Top KPIs
        valRqi.textContent = s.overall_average_rqi;
        const rating = s.overall_rating || "Good";
        badgeQualityStatus.textContent = rating.toUpperCase();
        badgeQualityStatus.className = `kpi-pill ${rating.toLowerCase()}`;
        
        valDistance.textContent = s.total_survey_distance_m.toFixed(1);
        txtSegmentsCount.textContent = `${s.total_segments_surveyed} discrete 50m segments`;
        
        valPotholes.textContent = s.total_unique_potholes;
        valCracks.textContent = s.total_unique_cracks;
        
        const potholeArea = segments.reduce((acc, seg) => acc + seg.pothole_area_m2, 0);
        const crackArea = segments.reduce((acc, seg) => acc + seg.crack_area_m2, 0);
        valPotholeArea.textContent = `${potholeArea.toFixed(2)} m² total pothole damage`;
        valCrackArea.textContent = `${crackArea.toFixed(2)} m² crack area`;

        const totalDefectArea = s.total_defect_area_m2;
        const totalRoadArea = s.total_survey_distance_m * 7.0;
        const overallDensity = totalRoadArea > 0 ? ((totalDefectArea / totalRoadArea) * 100).toFixed(2) : "0.0";
        valDensity.textContent = `${overallDensity}%`;
        valTotalDamageArea.textContent = `${totalDefectArea.toFixed(2)} m² damaged / ${totalRoadArea.toFixed(0)} m²`;

        // 2. Update Chart
        if (segments.length > 0) {
            const labels = segments.map(seg => `${seg.chainage_start_m}m-${seg.chainage_end_m}m`);
            const rqiData = segments.map(seg => seg.rqi_score);
            const gradeData = segments.map(seg => seg.mean_grade_pct);

            profileChart.data.labels = labels;
            profileChart.data.datasets[0].data = rqiData;
            profileChart.data.datasets[1].data = gradeData;
            
            // Adjust line color based on average RQI
            if (s.overall_average_rqi >= 80) {
                profileChart.data.datasets[0].borderColor = "#10b981";
                profileChart.data.datasets[0].backgroundColor = "rgba(16, 185, 129, 0.15)";
            } else if (s.overall_average_rqi >= 60) {
                profileChart.data.datasets[0].borderColor = "#f59e0b";
                profileChart.data.datasets[0].backgroundColor = "rgba(245, 158, 11, 0.15)";
            } else {
                profileChart.data.datasets[0].borderColor = "#ef4444";
                profileChart.data.datasets[0].backgroundColor = "rgba(239, 68, 68, 0.15)";
            }
            profileChart.update();
        }

        // 3. Populate Segment Matrix Table
        if (segments.length > 0) {
            segmentsTable.innerHTML = segments.map(seg => {
                const bClass = `badge-${seg.quality_rating.toLowerCase()}`;
                return `
                <tr>
                    <td><strong>Seg #${seg.segment_id + 1}</strong></td>
                    <td>${seg.chainage_start_m}m - ${seg.chainage_end_m}m</td>
                    <td><span class="badge-sm ${bClass}">${seg.quality_rating}</span></td>
                    <td style="font-weight:700;">${seg.rqi_score}</td>
                    <td>${seg.pothole_count}</td>
                    <td>${seg.crack_count}</td>
                    <td>${seg.defect_density_pct}%</td>
                    <td>${seg.mean_grade_pct > 0 ? '+' : ''}${seg.mean_grade_pct}%</td>
                    <td style="font-size:0.78rem; color:#9ca3af;">${seg.action_recommendation}</td>
                </tr>`;
            }).join("");
        } else {
            segmentsTable.innerHTML = `<tr><td colspan="9" class="text-center">No segments logged</td></tr>`;
        }

        // 4. Populate Tracked Defects Table
        if (defects.length > 0) {
            defectsTable.innerHTML = defects.map(d => {
                const sevColor = d.severity >= 4 ? '#ef4444' : (d.severity >= 3 ? '#f59e0b' : '#10b981');
                return `
                <tr>
                    <td><strong>T-${String(d.track_id).padStart(3, '0')}</strong></td>
                    <td>${d.type.toUpperCase()}</td>
                    <td>${d.chainage_distance_m}m</td>
                    <td>${d.smoothed_area_m2} m²</td>
                    <td>${d.smoothed_length_m}m × ${d.smoothed_width_m}m</td>
                    <td><span style="font-weight:700; color:${sevColor};">Sev ${d.severity}/5</span></td>
                    <td>${Math.round(d.confidence * 100)}%</td>
                </tr>`;
            }).join("");
        } else {
            defectsTable.innerHTML = `<tr><td colspan="7" class="text-center">No physical defects detected</td></tr>`;
        }

        // 5. Update Telemetry HUD
        if (segments.length > 0) {
            const lastSeg = segments[segments.length - 1];
            hudGrade.textContent = `${lastSeg.mean_grade_pct > 0 ? '+' : ''}${lastSeg.mean_grade_pct}%`;
            hudPitch.textContent = `${(lastSeg.mean_grade_pct * 0.57).toFixed(1)}°`;
            hudSegment.textContent = `#${lastSeg.segment_id + 1} (${lastSeg.chainage_start_m}-${lastSeg.chainage_end_m}m)`;
        }

        // 6. Enable Export Links
        linkHtmlReport.href = `/reports/${data.html_report}`;
        linkHtmlReport.classList.remove("disabled");

        linkSegmentsCsv.href = `/reports/${data.segments_csv}`;
        linkSegmentsCsv.classList.remove("disabled");

        linkDefectsCsv.href = `/reports/${data.defects_csv}`;
        linkDefectsCsv.classList.remove("disabled");
    }

    // Video Upload Handler
    videoUploadInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();
            if (data.success) {
                // Add to dropdown
                const opt = document.createElement("option");
                opt.value = data.route_name;
                opt.textContent = `${data.filename} (Uploaded)`;
                routeSelect.appendChild(opt);
                routeSelect.value = data.route_name;
                alert(`Video '${data.filename}' uploaded successfully! Click 'Run Full Survey' to analyze.`);
            } else {
                alert("Upload failed: " + data.error);
            }
        } catch (err) {
            console.error("Upload error:", err);
            alert("Upload failed.");
        }
    });

    initChart();
});
