import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
from fpdf import FPDF  # For PDF generation
import random
import string


# Mock OCR function for fake PDF extraction
def mock_ocr_extract(pdf_content):
    # Simulate extracting text from a "scanned" PDF
    # In reality, use pytesseract with PIL for image processing
    lines = pdf_content.split('\n')
    providers = []
    for line in lines:
        if ',' in line:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                providers.append({
                    'name': parts[0],
                    'specialty': parts[1],
                    'address': parts[2],
                    'phone': parts[3],
                    'license': parts[4],
                    'status': 'raw'
                })
    return pd.DataFrame(providers)


# Agent Classes
class DataValidationAgent:
    def __init__(self):
        self.logs = []

    def validate(self, df):
        self.logs.append(f"{datetime.now()}: Starting data validation.")
        validated_df = df.copy()
        for idx, row in validated_df.iterrows():
            # Mock API checks
            if not self.is_valid_name(row['name']):
                validated_df.at[idx, 'name'] = "Invalid Name"
            if not self.is_valid_address(row['address']):
                validated_df.at[idx, 'address'] = "Invalid Address"
            if not self.is_valid_phone(row['phone']):
                validated_df.at[idx, 'phone'] = "Invalid Phone"
            validated_df.at[idx, 'status'] = 'validated'
        self.logs.append(f"{datetime.now()}: Data validation completed.")
        return validated_df

    def is_valid_name(self, name):
        return len(name) > 2 and name.replace(' ', '').isalpha()

    def is_valid_address(self, addr):
        return len(addr) > 5  # Mock check

    def is_valid_phone(self, phone):
        return len(phone) == 10 and phone.isdigit()  # Mock US phone


class InformationEnrichmentAgent:
    def __init__(self):
        self.logs = []
        self.sample_data = {
            'specialty': ['Cardiology', 'Pediatrics', 'Oncology', 'Neurology'],
            'license': ['LIC12345', 'LIC67890', 'LIC11111', 'LIC22222']
        }

    def enrich(self, df):
        self.logs.append(f"{datetime.now()}: Starting information enrichment.")
        enriched_df = df.copy()
        for idx, row in enriched_df.iterrows():
            if pd.isna(row['specialty']) or row['specialty'] == '':
                enriched_df.at[idx, 'specialty'] = random.choice(self.sample_data['specialty'])
            if pd.isna(row['license']) or row['license'] == '':
                enriched_df.at[idx, 'license'] = random.choice(self.sample_data['license'])
        self.logs.append(f"{datetime.now()}: Information enrichment completed.")
        return enriched_df


class QualityAssuranceAgent:
    def __init__(self):
        self.logs = []

    def assess(self, original_df, validated_df):
        self.logs.append(f"{datetime.now()}: Starting quality assurance.")
        assessed_df = validated_df.copy()
        assessed_df['confidence'] = 0.0
        assessed_df['flagged'] = False
        for idx, row in assessed_df.iterrows():
            orig_row = original_df.iloc[idx]
            score = self.compute_confidence(orig_row, row)
            assessed_df.at[idx, 'confidence'] = score
            if score < 70 or 'Invalid' in str(row['name']) or 'Invalid' in str(row['address']) or 'Invalid' in str(
                    row['phone']):
                assessed_df.at[idx, 'flagged'] = True
        self.logs.append(f"{datetime.now()}: Quality assurance completed.")
        return assessed_df

    def compute_confidence(self, orig, val):
        score = 100
        if orig['name'] != val['name']: score -= 20
        if orig['address'] != val['address']: score -= 15
        if orig['phone'] != val['phone']: score -= 10
        if orig['specialty'] != val['specialty']: score -= 5
        return max(0, score)


class DirectoryManagementAgent:
    def __init__(self):
        self.logs = []

    def update_directory(self, df):
        self.logs.append(f"{datetime.now()}: Updating master directory.")
        # Simulate updating a master directory (in-memory)
        self.master_directory = df.copy()
        self.logs.append(f"{datetime.now()}: Directory updated.")
        return self.master_directory

    def generate_report(self, df):
        self.logs.append(f"{datetime.now()}: Generating summary report.")
        report = {
            'total_processed': len(df),
            'avg_confidence': df['confidence'].mean(),
            'flagged_count': df['flagged'].sum(),
            'time_saved': len(df) * 5  # Mock: 5 mins per provider saved
        }
        self.logs.append(f"{datetime.now()}: Report generated.")
        return report


class MasterAgent:
    def __init__(self):
        self.data_agent = DataValidationAgent()
        self.enrich_agent = InformationEnrichmentAgent()
        self.qa_agent = QualityAssuranceAgent()
        self.dir_agent = DirectoryManagementAgent()
        self.workflow_logs = []

    def orchestrate(self, df):
        self.workflow_logs.append(f"{datetime.now()}: Master Agent starting workflow.")

        # Step 1: Validation
        validated_df = self.data_agent.validate(df)
        self.workflow_logs.extend(self.data_agent.logs)

        # Step 2: Enrichment
        enriched_df = self.enrich_agent.enrich(validated_df)
        self.workflow_logs.extend(self.enrich_agent.logs)

        # Step 3: Quality Assurance
        assessed_df = self.qa_agent.assess(df, enriched_df)
        self.workflow_logs.extend(self.qa_agent.logs)

        # Step 4: Directory Update and Report
        final_df = self.dir_agent.update_directory(assessed_df)
        report = self.dir_agent.generate_report(final_df)
        self.workflow_logs.extend(self.dir_agent.logs)

        self.workflow_logs.append(f"{datetime.now()}: Master Agent workflow completed.")
        return final_df, report


# Streamlit App
st.set_page_config(page_title="ProValid AI", page_icon="🩺", layout="wide")
st.markdown("""
<style>
    .main {background-color: #E3F2FD;}
    .stButton>button {background-color: #00ACC1; color: white;}
    .stProgress > div > div > div > div {background-color: #00ACC1;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 ProValid AI")
st.subheader("Provider Data Validation and Directory Management – Agentic AI System")
st.markdown("*Automating trust in healthcare data*")

# Upload Section
uploaded_file = st.file_uploader("Upload Provider Dataset (CSV or Fake PDF)", type=['csv', 'pdf'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        # Mock PDF content
        pdf_content = uploaded_file.read().decode('utf-8', errors='ignore')
        df = mock_ocr_extract(pdf_content)

    st.write("Original Dataset:")
    st.dataframe(df.head())

    # Run Orchestration
    master = MasterAgent()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate progress
    for i in range(101):
        progress_bar.progress(i)
        if i == 25:
            status_text.text("Validating data...")
        elif i == 50:
            status_text.text("Enriching information...")
        elif i == 75:
            status_text.text("Assessing quality...")
        elif i == 100:
            status_text.text("Updating directory...")

    final_df, report = master.orchestrate(df)

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Validated Providers")
        st.dataframe(final_df[~final_df['flagged']])
    with col2:
        st.subheader("Flagged for Review")
        st.dataframe(final_df[final_df['flagged']])

    # Charts
    fig1 = px.histogram(final_df, x='confidence', title="Confidence Score Distribution")
    st.plotly_chart(fig1)

    fig2 = go.Figure(data=[go.Pie(labels=['Validated', 'Flagged'],
                                  values=[len(final_df[~final_df['flagged']]), len(final_df[final_df['flagged']])],
                                  title="Validation Status")])
    st.plotly_chart(fig2)

    # Logs
    st.subheader("Workflow Logs")
    for log in master.workflow_logs:
        st.text(log)

    # Generate Report
    if st.button("Generate Report"):
        # CSV Export
        csv = final_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="report.csv">Download CSV Report</a>'
        st.markdown(href, unsafe_allow_html=True)

        # PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="ProValid AI Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Total Processed: {report['total_processed']}", ln=True)
        pdf.cell(200, 10, txt=f"Avg Confidence: {report['avg_confidence']:.2f}%", ln=True)
        pdf.cell(200, 10, txt=f"Flagged: {report['flagged_count']}", ln=True)
        pdf.cell(200, 10, txt=f"Time Saved: {report['time_saved']} mins", ln=True)
        pdf_output = pdf.output(dest='S').encode('latin-1')
        b64_pdf = base64.b64encode(pdf_output).decode()
        href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="report.pdf">Download PDF Report</a>'
        st.markdown(href_pdf, unsafe_allow_html=True)

    # KPIs
    st.subheader("Overall KPIs")
    st.write(f"Total Providers Processed: {report['total_processed']}")
    st.write(f"Average Confidence Score: {report['avg_confidence']:.2f}%")
    st.write(f"Number of Flagged Providers: {report['flagged_count']}")
    st.write(f"Estimated Time Saved vs Manual Validation: {report['time_saved']} minutes")
