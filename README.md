SiteWatch AI

Multimodal Construction Site Safety Compliance & Incident Reporting Agent

SiteWatch AI is an AI-assisted construction-site safety monitoring application developed as an academic project. It uses computer vision to analyze construction-site images and videos, detect objects and potential safety risks, classify risk levels, and maintain a digital incident history.

1. Project Overview

Construction-site safety inspections are often performed manually. Manual inspection can require significant time and may make continuous monitoring difficult.

SiteWatch AI provides a prototype solution that uses computer vision to assist with:

Construction-site image analysis

Construction-site video analysis

Object detection

PPE-related safety detection when supported by the trained model

Potential hazard identification

Risk classification

Safety recommendations

Incident recording

Incident history

Safety analytics dashboard

Note: SiteWatch AI is an academic prototype and should be treated as an AI-assisted monitoring system, not as a replacement for qualified safety professionals or official site-safety procedures.

2. Main Features

Image Analysis

Upload a construction-site image and SiteWatch AI will:

Process the image using YOLO.

Detect objects supported by the loaded model.

Draw bounding boxes around detections.

Assign a risk level according to the configured rules.

Display safety recommendations.

Store medium, high, and critical incidents in the database.

Video Analysis

Upload a construction video in supported formats:

MP4

MOV

AVI

MKV

The application analyzes selected video frames and displays detected objects and potential safety issues.

Incident History

The system stores:

Incident ID

Date and time

Source image/video

Detected danger

Risk level

Confidence

Recommendation

Dashboard

The dashboard provides:

Total incidents

Critical incidents

High-risk incidents

Medium-risk incidents

Risk distribution

Detected safety issues

Incident sources

Recent incidents

3. System Workflow

User
  |
  v
Streamlit Web Application
  |
  +----------------------+
  |                      |
  v                      v
Image Upload          Video Upload
  |                      |
  +----------+-----------+
             |
             v
       OpenCV / PIL
             |
             v
        YOLO Model
             |
             v
      Object Detection
             |
             v
       Risk Assessment
             |
             v
   Safety Recommendation
             |
             v
       SQLite Database
             |
             v
      Dashboard / History

4. Technology Stack

Technology

Purpose

Python

Main programming language

Streamlit

Web application interface

YOLO / Ultralytics

Object detection

OpenCV

Image and video processing

Pillow

Image handling

NumPy

Numerical processing

Pandas

Data handling

Plotly

Dashboard charts

SQLite

Incident database

VS Code

Development environment

Git/GitHub

Version control

5. Project Structure

SiteWatch/
│
├── app.py
├── requirements.txt
├── README.md
├── sitewatch.db
│
├── best.pt
│
├── runs/
│   └── sitewatch_ppe/
│       └── weights/
│           └── best.pt
│
└── sample/
    ├── images/
    └── videos/

Important

best.pt is optional.

If best.pt exists, SiteWatch uses it first.

Otherwise, the application checks:

runs/sitewatch_ppe/weights/best.pt

If no custom model exists, it uses:

yolo11n.pt

6. Installation

Step 1: Install Python

Recommended Python version:

Python 3.12

Python 3.14 may cause compatibility issues with some machine-learning packages.

Step 2: Create Virtual Environment

Open the project folder in VS Code.

Run:

py -3.12 -m venv venv

Step 3: Install Packages

Use the virtual-environment Python directly:

.env\Scripts\python.exe -m pip install -r requirements.txt

If PyTorch gives a DLL initialization error on Windows, install the CPU version:

.env\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y

Then:

.env\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

7. Run the Application

Run:

.env\Scripts\python.exe -m streamlit run app.py

The application normally opens at:

http://localhost:8501

8. Application Navigation

Overview

The Overview page provides:

Project introduction

Total incidents

High-risk incidents

Critical incidents

AI model status

Main system features

Image Analysis

Go to:

📷 Image Analysis

Upload an image and click:

Analyze Safety

The system displays the annotated image and detected risks.

Video Analysis

Go to:

🎥 Video Analysis

Upload a video and click:

Analyze Video

The application processes selected frames and displays detections.

Incident History

Go to:

🚨 Incident History

Previously recorded incidents are displayed in a table.

Dashboard

Go to:

📊 Dashboard

The dashboard displays charts and safety statistics.

9. Risk Classification

SiteWatch uses configurable rules to map detected labels to risk levels.

Example:

Detection

Risk

Fire

Critical

Smoke

High

No Helmet

High

No Vest

High

No Boots

High

No Goggles

High

No Gloves

Medium

Sharp Object

High

Heavy Vehicle

Medium

Vehicle

Medium

Worker

Low

These rules are defined in app.py and can be modified according to the project requirements.

10. YOLO Model

The application uses the Ultralytics YOLO framework.

The model returns object classes and confidence scores.

Example:

Person       91.2%
Truck        87.4%
Car          82.5%

The application then maps the detected label to a risk level.

Important Model Limitation

A standard general-purpose YOLO model can only detect classes that it was trained to recognize.

For example, a generic model does not automatically understand:

No Helmet
No Vest
No Gloves
Fire
Smoke

unless those classes are included in the trained model.

For reliable construction PPE and hazard detection, a construction/PPE-specific trained YOLO model should be used.

11. Database

SiteWatch uses SQLite.

Database file:

sitewatch.db

Table:

incidents

Fields:

id
timestamp
source
danger
risk
confidence
recommendation

The application automatically creates the database and performs basic schema migration if required columns are missing.

12. Example Incident

Example:

Danger:
Helmet Violation

Risk:
High

Confidence:
91.4%

Recommendation:
Worker must wear an approved safety helmet before
entering the construction zone.

13. Project Objectives

The main objectives are:

Develop an AI-assisted construction safety monitoring system.

Analyze construction images and videos.

Detect objects using computer vision.

Identify potential safety violations when supported by the model.

Classify detected risks.

Generate safety recommendations.

Maintain a digital incident database.

Provide an analytics dashboard.

Reduce manual monitoring effort.

Provide a foundation for future real-time monitoring.

14. Advantages

Faster analysis of visual evidence

Digital incident records

Centralized incident history

Automated risk classification

Visual detection results

Image and video support

Dashboard-based analytics

Easy academic demonstration

Extensible architecture

15. Future Scope

Future versions can include:

Real-time CCTV monitoring

Live camera feeds

Mobile application

Custom construction PPE dataset

Helmet/vest/gloves detection

Fire and smoke detection

Restricted-area detection

Worker tracking

Real-time alerts

SMS/email notifications

IoT sensor integration

Predictive safety analytics

Cloud database

Role-based access

AI-generated incident reports

Integration with enterprise safety-management platforms

16. Testing

Test the application using:

Image Tests

Construction site with workers

Workers wearing PPE

Workers without PPE

Construction vehicles

Equipment and machinery

Potentially hazardous scenes

Video Tests

Construction-site videos

Worker movement

Vehicle movement

PPE-related scenarios

Different camera angles

Different lighting conditions

Record:

Detection result

Confidence

Risk level

False detections

Missed detections

17. Limitations

The current academic prototype has several limitations:

Detection quality depends on the trained model.

A generic YOLO model cannot detect classes it was not trained on.

Video analysis is not necessarily real-time.

Lighting and camera angle can affect detection.

False positives and false negatives can occur.

Risk rules are configurable and simplified for the prototype.

The system should not be used as the sole basis for real-world safety decisions.

18. Troubleshooting

Streamlit command not working

Use:

.env\Scripts\python.exe -m streamlit run app.py

PowerShell activation blocked

You do not need to activate the environment.

Use:

.env\Scripts\python.exe

directly.

PyTorch DLL error

Use Python 3.12 and the CPU PyTorch build.

Database column error

The application automatically checks the existing SQLite database schema and adds missing columns.

Model not found

Place your trained model here:

best.pt

in the same folder as app.py.

19. GitHub Upload

Before uploading to GitHub, avoid committing unnecessary large or generated files.

Recommended repository structure:

SiteWatch/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

A .gitignore file can contain:

venv/
__pycache__/
*.pyc
sitewatch.db
.streamlit/
runs/

If best.pt is too large for normal GitHub storage, use Git LFS or keep the model outside the repository.

20. Academic Project Summary

Project Title

SiteWatch: Multimodal Construction Site Safety Compliance & Incident Reporting Agent

Domain

Artificial Intelligence / Machine Learning / Computer Vision

Application

Construction Site Safety Monitoring

Core Technologies

Python, Streamlit, YOLO, OpenCV, SQLite, Pandas and Plotly.

Core Input

Construction-site images and videos.

Core Output

Detected objects, risk classification, safety recommendations and incident records.

21. Team Presentation Flow

For project review, the system can be explained in this order:

Problem
   ↓
Manual Construction Safety Inspection
   ↓
Proposed SiteWatch AI
   ↓
Image / Video Input
   ↓
YOLO Object Detection
   ↓
Risk Classification
   ↓
Safety Recommendation
   ↓
Incident Database
   ↓
Dashboard

22. Conclusion

SiteWatch AI demonstrates how computer vision can be applied to construction-site safety monitoring. The prototype combines image/video analysis, object detection, risk assessment, incident storage and dashboard analytics in a single Streamlit application.

The architecture can later be extended with custom PPE datasets, real-time CCTV streams, advanced risk models, notification systems and cloud-based deployment.