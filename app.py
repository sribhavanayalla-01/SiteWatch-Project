import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SiteWatch AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f4f6f9;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #101827;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* TITLES */

.title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #667085;
    margin-bottom: 25px;
}

/* HERO */

.hero {
    background: linear-gradient(135deg, #111827, #374151);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin: 0 0 10px 0;
}

.hero p {
    color: #d1d5db;
    font-size: 16px;
    margin: 0;
}

/* LOGO */

.logo-box {
    text-align: center;
    padding: 10px 5px 25px 5px;
}

.logo-icon {
    font-size: 48px;
    margin-bottom: 5px;
}

.logo-title {
    font-size: 24px;
    font-weight: 800;
    color: white;
}

.logo-subtitle {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 5px;
}

/* METRIC */

.metric-box {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
    text-align: center;
    min-height: 105px;
}

.metric-number {
    font-size: 32px;
    font-weight: 800;
    color: #111827;
}

.metric-label {
    color: #667085;
    font-size: 14px;
    margin-top: 5px;
}

/* INFO */

.info-box {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    margin: 15px 0;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.04);
}

.info-box h3 {
    color: #111827;
    margin-top: 0;
}

.info-box p {
    color: #667085;
    line-height: 1.6;
}

/* RISK */

.critical {
    background: #fee2e2;
    border-left: 6px solid #dc2626;
    padding: 18px;
    border-radius: 12px;
    margin: 10px 0;
}

.critical h3,
.critical p {
    color: #7f1d1d;
}

.high {
    background: #ffedd5;
    border-left: 6px solid #ea580c;
    padding: 18px;
    border-radius: 12px;
    margin: 10px 0;
}

.high h3,
.high p {
    color: #9a3412;
}

.medium {
    background: #fef3c7;
    border-left: 6px solid #d97706;
    padding: 18px;
    border-radius: 12px;
    margin: 10px 0;
}

.medium h3,
.medium p {
    color: #92400e;
}

.safe {
    background: #dcfce7;
    border-left: 6px solid #16a34a;
    padding: 18px;
    border-radius: 12px;
    margin: 10px 0;
}

.safe h3,
.safe p {
    color: #166534;
}

/* BUTTON */

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 45px;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    padding: 30px 0 10px 0;
    color: #98A2B3;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

DB = "sitewatch.db"


def create_database():

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            danger TEXT,
            risk TEXT,
            confidence REAL,
            recommendation TEXT
        )
    """)

    # Check current columns
    cursor.execute("PRAGMA table_info(incidents)")

    existing_columns = [
        row[1] for row in cursor.fetchall()
    ]

    required_columns = {
        "timestamp": "TEXT",
        "source": "TEXT",
        "danger": "TEXT",
        "risk": "TEXT",
        "confidence": "REAL",
        "recommendation": "TEXT"
    }

    for column, datatype in required_columns.items():

        if column not in existing_columns:

            cursor.execute(
                f"ALTER TABLE incidents ADD COLUMN {column} {datatype}"
            )

    conn.commit()
    conn.close()


create_database()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Custom trained model
    if os.path.exists("best.pt"):
        return YOLO("best.pt")

    # Alternative custom model path
    if os.path.exists(
        "runs/sitewatch_ppe/weights/best.pt"
    ):
        return YOLO(
            "runs/sitewatch_ppe/weights/best.pt"
        )

    # Default YOLO model
    return YOLO("yolo11n.pt")


try:

    model = load_model()

    MODEL_READY = True
    MODEL_ERROR = ""

except Exception as e:

    model = None
    MODEL_READY = False
    MODEL_ERROR = str(e)


# ============================================================
# DANGER RULES
# ============================================================

DANGER_RULES = {

    "fire": {
        "risk": "Critical",
        "name": "Fire Detected",
        "recommendation":
            "Immediately isolate the affected area and follow emergency fire safety procedures."
    },

    "smoke": {
        "risk": "High",
        "name": "Smoke Detected",
        "recommendation":
            "Inspect the area for fire or hazardous fumes and restrict access."
    },

    "no_helmet": {
        "risk": "High",
        "name": "Helmet Violation",
        "recommendation":
            "Worker must wear an approved safety helmet before entering the construction zone."
    },

    "no_vest": {
        "risk": "High",
        "name": "Safety Vest Violation",
        "recommendation":
            "Worker should wear a high-visibility safety vest."
    },

    "no_gloves": {
        "risk": "Medium",
        "name": "Glove Violation",
        "recommendation":
            "Worker should wear suitable protective gloves."
    },

    "no_boots": {
        "risk": "High",
        "name": "Safety Boots Violation",
        "recommendation":
            "Worker should wear approved safety footwear."
    },

    "no_goggle": {
        "risk": "High",
        "name": "Safety Goggles Violation",
        "recommendation":
            "Worker should use suitable eye protection."
    },

    "knife": {
        "risk": "High",
        "name": "Sharp Object Detected",
        "recommendation":
            "Keep sharp tools secured and ensure appropriate protective equipment is used."
    },

    "truck": {
        "risk": "Medium",
        "name": "Heavy Vehicle Detected",
        "recommendation":
            "Maintain a safe distance from heavy construction vehicles."
    },

    "car": {
        "risk": "Medium",
        "name": "Vehicle Detected",
        "recommendation":
            "Maintain separation between workers and moving vehicles."
    },

    "bus": {
        "risk": "Medium",
        "name": "Large Vehicle Detected",
        "recommendation":
            "Maintain a safe distance from the vehicle operating area."
    },

    "motorcycle": {
        "risk": "Medium",
        "name": "Motorcycle Detected",
        "recommendation":
            "Maintain separation from moving vehicles."
    },

    "person": {
        "risk": "Low",
        "name": "Worker Detected",
        "recommendation":
            "Continue monitoring PPE compliance and worker safety."
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk(label):

    label = str(label).lower().strip()

    if label in DANGER_RULES:
        return DANGER_RULES[label]["risk"]

    return "Low"


def get_name(label):

    label = str(label).lower().strip()

    if label in DANGER_RULES:
        return DANGER_RULES[label]["name"]

    return label.replace("_", " ").title()


def get_recommendation(label):

    label = str(label).lower().strip()

    if label in DANGER_RULES:
        return DANGER_RULES[label]["recommendation"]

    return "Continue monitoring the construction site."


def save_incident(
    source,
    danger,
    risk,
    confidence,
    recommendation
):

    conn = sqlite3.connect(DB)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO incidents
        (
            timestamp,
            source,
            danger,
            risk,
            confidence,
            recommendation
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        source,
        danger,
        risk,
        confidence,
        recommendation
    ))

    conn.commit()
    conn.close()


# ============================================================
# IMAGE ANALYSIS FUNCTION
# ============================================================

def analyze_image(image):

    if not MODEL_READY:

        return None, []

    image_cv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    results = model(
        image_cv,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            label = model.names[cls_id]

            risk = get_risk(label)

            display_name = get_name(label)

            detections.append({
                "label": label,
                "name": display_name,
                "risk": risk,
                "confidence": confidence
            })

    annotated = results[0].plot()

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    return annotated, detections


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div class="logo-box">

        <div class="logo-icon">
            🦺
        </div>

        <div class="logo-title">
            SiteWatch AI
        </div>

        <div class="logo-subtitle">
            Construction Safety Intelligence
        </div>

    </div>
    """)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📷 Image Analysis",
            "🎥 Video Analysis",
            "🚨 Incident History",
            "📊 Dashboard"
        ]
    )

    st.markdown("---")

    if MODEL_READY:

        st.success(
            "AI Model Ready"
        )

    else:

        st.error(
            "AI Model Error"
        )

        st.caption(
            MODEL_ERROR
        )


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.html("""
    <div class="hero">

        <h1>
            🛡️ Construction Safety Intelligence
        </h1>

        <p>
            AI-assisted monitoring for construction-site
            safety, PPE compliance and hazard detection.
        </p>

    </div>
    """)

    # Read database

    conn = sqlite3.connect(DB)

    df = pd.read_sql_query(
        "SELECT * FROM incidents",
        conn
    )

    conn.close()

    total_incidents = len(df)

    if not df.empty:

        high_count = len(
            df[
                df["risk"].isin(
                    ["High", "Critical"]
                )
            ]
        )

        critical_count = len(
            df[
                df["risk"] == "Critical"
            ]
        )

    else:

        high_count = 0
        critical_count = 0

    # Metrics

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.html(f"""
        <div class="metric-box">

            <div class="metric-number">
                {total_incidents}
            </div>

            <div class="metric-label">
                Total Incidents
            </div>

        </div>
        """)

    with c2:

        st.html(f"""
        <div class="metric-box">

            <div class="metric-number">
                {high_count}
            </div>

            <div class="metric-label">
                High Risk
            </div>

        </div>
        """)

    with c3:

        st.html(f"""
        <div class="metric-box">

            <div class="metric-number">
                {critical_count}
            </div>

            <div class="metric-label">
                Critical
            </div>

        </div>
        """)

    with c4:

        st.html("""
        <div class="metric-box">

            <div class="metric-number">
                AI
            </div>

            <div class="metric-label">
                Detection Engine
            </div>

        </div>
        """)

    # Project description

    st.html("""
    <div class="info-box">

        <h3>
            🚀 What SiteWatch Does
        </h3>

        <p>
            SiteWatch uses computer vision to analyze
            construction-site images and videos,
            identify objects and potential safety risks,
            classify incidents and maintain a digital
            incident history.
        </p>

    </div>
    """)

    # Feature cards

    col1, col2, col3 = st.columns(3)

    with col1:

        st.html("""
        <div class="info-box">

            <h3>
                📷 Image Monitoring
            </h3>

            <p>
                Upload construction-site images
                and analyze detected objects
                and safety risks.
            </p>

        </div>
        """)

    with col2:

        st.html("""
        <div class="info-box">

            <h3>
                🎥 Video Monitoring
            </h3>

            <p>
                Analyze construction videos and
                identify potential hazards
                frame by frame.
            </p>

        </div>
        """)

    with col3:

        st.html("""
        <div class="info-box">

            <h3>
                🚨 Incident Management
            </h3>

            <p>
                Automatically record detected
                incidents and maintain safety
                history.
            </p>

        </div>
        """)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "📷 Image Analysis":

    st.markdown(
        '<div class="title">📷 Image Safety Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Upload a construction-site image for AI analysis.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload construction image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Construction Site",
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Safety",
            use_container_width=True
        ):

            if not MODEL_READY:

                st.error(
                    "AI model could not be loaded."
                )

            else:

                with st.spinner(
                    "AI is analyzing the image..."
                ):

                    annotated, detections = analyze_image(
                        np.array(image)
                    )

                st.subheader(
                    "AI Detection Result"
                )

                st.image(
                    annotated,
                    caption="AI Annotated Image",
                    use_container_width=True
                )

                if len(detections) == 0:

                    st.html("""
                    <div class="safe">

                        <h3>
                            🟢 No objects detected
                        </h3>

                        <p>
                            The AI model did not identify
                            any objects in this image.
                        </p>

                    </div>
                    """)

                else:

                    # Remove duplicates

                    unique_detections = []

                    seen = set()

                    for item in detections:

                        key = (
                            item["name"],
                            item["risk"]
                        )

                        if key not in seen:

                            seen.add(key)

                            unique_detections.append(
                                item
                            )

                    for item in unique_detections:

                        risk = item["risk"]

                        confidence = (
                            item["confidence"] * 100
                        )

                        danger_name = item["name"]

                        recommendation = get_recommendation(
                            item["label"]
                        )

                        if risk == "Critical":

                            css_class = "critical"
                            icon = "🔴"

                        elif risk == "High":

                            css_class = "high"
                            icon = "🟠"

                        elif risk == "Medium":

                            css_class = "medium"
                            icon = "🟡"

                        else:

                            css_class = "safe"
                            icon = "🟢"

                        st.html(f"""
                        <div class="{css_class}">

                            <h3>
                                {icon} {danger_name}
                            </h3>

                            <p>
                                <b>Risk:</b>
                                {risk}
                            </p>

                            <p>
                                <b>Confidence:</b>
                                {confidence:.1f}%
                            </p>

                            <p>
                                <b>Recommendation:</b>
                                {recommendation}
                            </p>

                        </div>
                        """)

                        # Save real risks

                        if risk in [
                            "Medium",
                            "High",
                            "Critical"
                        ]:

                            save_incident(
                                source=uploaded_file.name,
                                danger=danger_name,
                                risk=risk,
                                confidence=item[
                                    "confidence"
                                ],
                                recommendation=recommendation
                            )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

elif page == "🎥 Video Analysis":

    st.markdown(
        '<div class="title">🎥 Video Safety Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Analyze construction videos for potential safety risks.</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload construction video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ]
    )

    if uploaded_video is not None:

        st.video(
            uploaded_video
        )

        if st.button(
            "🎬 Analyze Video",
            use_container_width=True
        ):

            if not MODEL_READY:

                st.error(
                    "AI model could not be loaded."
                )

            else:

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                temp_file.write(
                    uploaded_video.read()
                )

                temp_file.close()

                cap = cv2.VideoCapture(
                    temp_file.name
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps == 0:

                    fps = 25

                duration = (
                    total_frames / fps
                    if total_frames > 0
                    else 0
                )

                st.info(
                    f"Video duration: {duration:.1f} seconds"
                )

                frame_placeholder = st.empty()

                progress_bar = st.progress(0)

                detected_incidents = {}

                frame_number = 0

                # Analyze every 8th frame

                FRAME_SKIP = 8

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    if frame_number % FRAME_SKIP != 0:
                        continue

                    try:

                        results = model(
                            frame,
                            verbose=False
                        )

                        annotated = results[
                            0
                        ].plot()

                        annotated_rgb = cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        )

                        frame_placeholder.image(
                            annotated_rgb,
                            channels="RGB",
                            use_container_width=True
                        )

                        if results[0].boxes is not None:

                            for box in results[0].boxes:

                                cls_id = int(
                                    box.cls[0]
                                )

                                confidence = float(
                                    box.conf[0]
                                )

                                label = model.names[
                                    cls_id
                                ]

                                risk = get_risk(
                                    label
                                )

                                danger_name = get_name(
                                    label
                                )

                                if risk in [
                                    "Medium",
                                    "High",
                                    "Critical"
                                ]:

                                    if (
                                        danger_name
                                        not in
                                        detected_incidents
                                    ):

                                        detected_incidents[
                                            danger_name
                                        ] = {

                                            "risk": risk,

                                            "confidence":
                                                confidence,

                                            "recommendation":
                                                get_recommendation(
                                                    label
                                                )
                                        }

                    except Exception:

                        continue

                    if total_frames > 0:

                        progress = min(
                            frame_number /
                            total_frames,
                            1.0
                        )

                        progress_bar.progress(
                            progress
                        )

                cap.release()

                try:

                    os.remove(
                        temp_file.name
                    )

                except:

                    pass

                progress_bar.progress(
                    1.0
                )

                st.success(
                    "Video analysis completed."
                )

                if len(
                    detected_incidents
                ) == 0:

                    st.html("""
                    <div class="safe">

                        <h3>
                            🟢 No major safety incidents detected
                        </h3>

                        <p>
                            No medium, high or critical
                            detections were found by the model.
                        </p>

                    </div>
                    """)

                else:

                    st.subheader(
                        "🚨 Detected Safety Issues"
                    )

                    for danger, data in detected_incidents.items():

                        risk = data["risk"]

                        confidence = data[
                            "confidence"
                        ]

                        recommendation = data[
                            "recommendation"
                        ]

                        if risk == "Critical":

                            css_class = "critical"
                            icon = "🔴"

                        elif risk == "High":

                            css_class = "high"
                            icon = "🟠"

                        else:

                            css_class = "medium"
                            icon = "🟡"

                        st.html(f"""
                        <div class="{css_class}">

                            <h3>
                                {icon} {danger}
                            </h3>

                            <p>
                                <b>Risk:</b>
                                {risk}
                            </p>

                            <p>
                                <b>Confidence:</b>
                                {confidence * 100:.1f}%
                            </p>

                            <p>
                                <b>Recommendation:</b>
                                {recommendation}
                            </p>

                        </div>
                        """)

                        save_incident(
                            source=uploaded_video.name,
                            danger=danger,
                            risk=risk,
                            confidence=confidence,
                            recommendation=recommendation
                        )


# ============================================================
# INCIDENT HISTORY
# ============================================================

elif page == "🚨 Incident History":

    st.markdown(
        '<div class="title">🚨 Incident History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Previously detected construction safety incidents.</div>',
        unsafe_allow_html=True
    )

    conn = sqlite3.connect(DB)

    df = pd.read_sql_query(
        """
        SELECT
            id AS ID,
            timestamp AS Time,
            source AS Source,
            danger AS Danger,
            risk AS Risk,
            confidence AS Confidence,
            recommendation AS Recommendation

        FROM incidents

        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    if df.empty:

        st.info(
            "No incidents have been recorded yet."
        )

    else:

        display_df = df.copy()

        display_df["Confidence"] = (
            display_df["Confidence"] * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Risk Summary"
        )

        risk_counts = (
            df["Risk"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk",
            "Count"
        ]

        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Count",
            title="Incidents by Risk Level"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.markdown(
        '<div class="title">📊 Safety Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Construction-site safety analytics.</div>',
        unsafe_allow_html=True
    )

    conn = sqlite3.connect(DB)

    df = pd.read_sql_query(
        "SELECT * FROM incidents",
        conn
    )

    conn.close()

    if df.empty:

        st.info(
            "No incident data available. Analyze an image or video first."
        )

    else:

        # Metrics

        total = len(df)

        critical = len(
            df[
                df["risk"] == "Critical"
            ]
        )

        high = len(
            df[
                df["risk"] == "High"
            ]
        )

        medium = len(
            df[
                df["risk"] == "Medium"
            ]
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Total Incidents",
                total
            )

        with c2:

            st.metric(
                "Critical",
                critical
            )

        with c3:

            st.metric(
                "High",
                high
            )

        with c4:

            st.metric(
                "Medium",
                medium
            )

        st.markdown("")

        # Charts

        col1, col2 = st.columns(2)

        with col1:

            risk_data = (
                df["risk"]
                .value_counts()
                .reset_index()
            )

            risk_data.columns = [
                "Risk",
                "Count"
            ]

            fig1 = px.pie(
                risk_data,
                names="Risk",
                values="Count",
                title="Risk Distribution"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

        with col2:

            danger_data = (
                df["danger"]
                .value_counts()
                .reset_index()
            )

            danger_data.columns = [
                "Danger",
                "Count"
            ]

            fig2 = px.bar(
                danger_data,
                x="Danger",
                y="Count",
                title="Detected Safety Issues"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # Sources

        st.subheader(
            "📁 Incident Sources"
        )

        source_data = (
            df["source"]
            .value_counts()
            .reset_index()
        )

        source_data.columns = [
            "Source",
            "Count"
        ]

        st.dataframe(
            source_data,
            use_container_width=True,
            hide_index=True
        )

        # Recent incidents

        st.subheader(
            "🕒 Recent Incidents"
        )

        recent = df.sort_values(
            "id",
            ascending=False
        ).head(10)

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    SiteWatch AI • Construction Safety Intelligence
</div>
""")