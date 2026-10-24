"""Streamlit Web GUI for Healthcare RAG Agent with Prescription Screener & pgvector Explorer."""

import json
import os
from pathlib import Path
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd

try:
    streamlit_secrets = st.secrets
except StreamlitSecretNotFoundError:
    streamlit_secrets = {}

for secret_name in (
    "DATABASE_URL",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "AI_PROVIDER",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
):
    if secret_name in streamlit_secrets:
        os.environ[secret_name] = str(streamlit_secrets[secret_name])

from src.config import config
from src.db import db
from src.agent import agent
from src.retriever import retriever
from src.ingest import run_ingestion
from src.chunker import chunker
from src.embedder import embedder

# Page configuration
st.set_page_config(
    page_title="Healthcare RAG & Prescription Screener",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0083B0; }
    .phase-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.9em; margin-bottom: 8px; }
    .phase-ingest { background-color: #e3f2fd; color: #0d47a1; border-left: 4px solid #1976d2; }
    .phase-query { background-color: #f3e5f5; color: #4a148c; border-left: 4px solid #7b1fa2; }
    .phase-audit { background-color: #e8f5e9; color: #1b5e20; border-left: 4px solid #388e3c; }
    .chunk-box { border-left: 4px solid #0083B0; padding: 10px 14px; background: #f8f9fa; margin-bottom: 10px; border-radius: 4px; font-size: 0.95em; }
    .verdict-box { padding: 12px 18px; border-radius: 6px; font-weight: bold; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# Load Sample Prescriptions
SAMPLE_RX_PATH = config.root_dir / "data" / "sample_prescriptions.json"
sample_prescriptions = []
if SAMPLE_RX_PATH.exists():
    with open(SAMPLE_RX_PATH, "r", encoding="utf-8") as f:
        sample_prescriptions = json.load(f)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        chunk_count = db.get_chunk_count()
        st.success("Connected to PostgreSQL")
        st.metric("Total pgvector Chunks", chunk_count)
    except Exception as e:
        st.error(f"Database Error: {e}")
        chunk_count = 0

    st.markdown("---")
    st.header("🤖 Models")
    st.info(f"**LLM:** `{config.llm_model}`\n\n**Embedder:** `{config.embedding_model}` (768-D)")

    st.markdown("---")
    st.header("🔄 Knowledge Base")
    if st.button("Re-Ingest All Medications", help="Re-chunks and re-embeds all monographs in data/medications"):
        with st.spinner("Ingesting medication monographs into pgvector..."):
            count = run_ingestion(reset_db=True)
            st.success(f"Successfully ingested {count} chunks!")
            st.rerun()

st.title("💊 Clinical RAG: Prescription Screener & Knowledge Explorer")
st.caption("Grounding patient prescriptions in authoritative medication monographs using local Ollama & pgvector.")

# Top-level tabs
tab_prescriptions, tab_chat, tab_ingest_studio, tab_explorer = st.tabs([
    "📋 Prescription Screener (Ingest & Query)",
    "💬 Clinical Q&A Chat",
    "📥 Monograph Ingestion Studio",
    "🗄️ Database Explorer (SQL View)"
])

# ----------------------------------------------------
# TAB 1: PRESCRIPTION SCREENER (PRIMARY WORKFLOW)
# ----------------------------------------------------
with tab_prescriptions:
    st.subheader("Automated Prescription Safety Audit")
    st.markdown(
        "Load sample clinical prescriptions to test how the RAG pipeline validates medications against **pgvector source of truth**."
    )

    col_selector, col_custom = st.columns([2, 1])

    # Sample Prescriptions Dropdown
    rx_titles = [f"{rx['id'].upper()} - {rx['title']}" for rx in sample_prescriptions]
    selected_idx = col_selector.selectbox(
        "Select a Sample Clinical Prescription to Test:",
        range(len(rx_titles)),
        format_func=lambda i: rx_titles[i]
    )
    current_sample = sample_prescriptions[selected_idx]

    # Display Prescription Form (Editable)
    with st.expander("📝 View / Edit Selected Patient Record & Prescription", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        pat_name = c1.text_input("Patient Name:", value=current_sample.get("patient_name", ""))
        pat_age = c2.number_input("Age:", value=int(current_sample.get("age", 40)), step=1)
        pat_gender = c3.selectbox("Gender:", ["Female", "Male", "Other"], index=0 if current_sample.get("gender") == "Female" else 1)

        pat_conditions = st.text_area("Clinical Conditions & Medical History:", value=current_sample.get("clinical_conditions", ""))
        
        ca, cb = st.columns(2)
        pat_allergies = ca.text_input("Known Allergies:", value=current_sample.get("known_allergies", "NKDA"))
        pat_current_meds = cb.text_input("Current Medications / Habits:", value=current_sample.get("current_medications", "None"))

        # Prescribed Items
        st.markdown("**Prescribed Medication(s):**")
        rx_items = current_sample.get("prescribed_rx", [])
        rx_text_lines = []
        for rx in rx_items:
            rx_text_lines.append(f"{rx.get('medication')} | Brand: {rx.get('brand')} | Dose: {rx.get('dosage')} | Frequency: {rx.get('frequency')} | Indication: {rx.get('indication')}")
        
        rx_display_text = st.text_area("Prescription Line Items:", value="\n".join(rx_text_lines), height=80)
        pat_notes = st.text_input("Clinical Notes / Context:", value=current_sample.get("notes", ""))

    btn_audit = st.button("🚀 Run RAG Safety Audit on Prescription", type="primary", use_container_width=True)

    if btn_audit:
        compiled_rx = {
            "patient_name": pat_name,
            "age": pat_age,
            "gender": pat_gender,
            "clinical_conditions": pat_conditions,
            "known_allergies": pat_allergies,
            "current_medications": pat_current_meds,
            "prescribed_rx": rx_items,
            "notes": pat_notes
        }

        # Visual Flow Section
        st.markdown("---")
        st.markdown("### 🔬 RAG Execution Pipeline Trace")

        # ----------------------------------------------------
        # PHASE 1: INGESTION & GROUNDING VERIFICATION
        # ----------------------------------------------------
        st.markdown('<div class="phase-badge phase-ingest">PHASE 1: KNOWLEDGE INGESTION & SOURCE OF TRUTH VERIFICATION</div>', unsafe_allow_html=True)
        
        prescribed_drugs = [rx.get("medication") for rx in rx_items]
        ingest_cols = st.columns(len(prescribed_drugs) if prescribed_drugs else 1)

        with db.get_connection() as conn:
            with conn.cursor() as cur:
                for idx, drug in enumerate(prescribed_drugs):
                    cur.execute(
                        "SELECT COUNT(*) FROM medication_chunks WHERE LOWER(medication_name) = LOWER(%s);",
                        (drug,)
                    )
                    count_for_drug = cur.fetchone()[0]
                    with ingest_cols[idx]:
                        if count_for_drug > 0:
                            st.success(f"**{drug}**\n\n[OK] Grounded in pgvector ({count_for_drug} chunks)")
                        else:
                            st.error(f"**{drug}**\n\n[X] Not in pgvector database! (Refusal Guardrail will trigger)")

        st.caption("The agent verifies that every drug in the prescription has authoritative monographs indexed in the vector database.")

        # ----------------------------------------------------
        # PHASE 2: RETRIEVAL & VECTOR QUERYING
        # ----------------------------------------------------
        st.markdown('<div class="phase-badge phase-query">PHASE 2: PGVECTOR RETRIEVAL & VECTOR SIMILARITY QUERYING</div>', unsafe_allow_html=True)

        with st.spinner("Embedding query & searching pgvector via cosine similarity (<=>)..."):
            audit_result = agent.audit_prescription(compiled_rx, verbose=False)
            retrieved_chunks = audit_result["retrieved_chunks"]

        st.markdown(f"**Retrieved Top-{len(retrieved_chunks)} Clinical Evidence Chunks from PostgreSQL:**")
        
        # Display retrieved chunks with confidence badges
        for i, chk in enumerate(retrieved_chunks, start=1):
            sim = chk.get("similarity", 0.0)
            dist = chk.get("distance", 1.0 - sim)
            med = chk.get("medication_name")
            sec = chk.get("section")
            st.markdown(f"""
            <div class="chunk-box">
                <strong>#{i} {med}</strong> | <span style="color:#7b1fa2;">{sec}</span> | 
                <strong>Match Confidence:</strong> <code style="color:green;">{sim * 100:.1f}%</code> | 
                <strong>Cosine Distance:</strong> <code>{dist:.4f}</code><br>
                <div style="margin-top: 5px; color: #333;">{chk.get('content')}</div>
            </div>
            """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # PHASE 3: CLINICAL AUDIT & LLM GENERATION
        # ----------------------------------------------------
        st.markdown('<div class="phase-badge phase-audit">PHASE 3: CLINICAL AUDIT GENERATION (Ollama llama3.2)</div>', unsafe_allow_html=True)
        st.markdown(audit_result["audit_report"])
        st.caption(f"⚡ Generation Latency: {audit_result['latency_ms']:.1f} ms | Evaluated strictly against pgvector context")

# ----------------------------------------------------
# TAB 2: CLINICAL Q&A CHAT
# ----------------------------------------------------
with tab_chat:
    st.subheader("Patient Medication Consultation")
    st.markdown("Ask any ad-hoc medication question grounded strictly in the medication handbook:")

    c1, c2, c3 = st.columns(3)
    preset_q = None
    if c1.button("💊 Can I take Ibuprofen with stomach ulcers?"):
        preset_q = "Can I take Ibuprofen if I have active stomach ulcers?"
    if c2.button("⏱️ Maximum daily dose of Paracetamol?"):
        preset_q = "What is the maximum daily dose of Paracetamol for adults?"
    if c3.button("🍇 Atorvastatin & Grapefruit juice?"):
        preset_q = "Why can't I drink grapefruit juice while taking Atorvastatin?"

    user_query = st.chat_input("Enter your medication inquiry...")
    active_query = preset_q or user_query

    if active_query:
        st.chat_message("user").write(active_query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving from pgvector and generating response..."):
                res = agent.ask(active_query, verbose=False)
            st.markdown(res["answer"])
            st.caption(f"⚡ Latency: {res['latency_ms']:.1f} ms | Grounded in pgvector")

            with st.expander("🔍 View Retrieved pgvector Evidence"):
                for i, chk in enumerate(res["retrieved_chunks"], start=1):
                    sim = chk.get("similarity", 0.0)
                    st.markdown(f"**#{i} {chk.get('medication_name')} ({chk.get('section')})** — Match: `{sim * 100:.1f}%`\n\n{chk.get('content')}\n---")

# ----------------------------------------------------
# TAB 3: MONOGRAPH INGESTION STUDIO (DYNAMIC INGEST)
# ----------------------------------------------------
with tab_ingest_studio:
    st.subheader("📥 Dynamic Medication Ingestion Studio")
    st.markdown("Ingest a new medication monograph directly into pgvector and watch the chunking and embedding happen live!")

    with st.form("ingest_form"):
        new_med_name = st.text_input("Medication Name (Generic):", value="Ciprofloxacin")
        new_brands = st.text_input("Brand Names (comma-separated):", value="Cipro, Ciproxin")
        new_class = st.text_input("Therapeutic Class:", value="Fluoroquinolone Antibacterial")
        new_indications = st.text_area("Indications & Uses:", value="Indicated for the treatment of severe bacterial infections including urinary tract infections, anthrax, bacterial prostatitis, and bone and joint infections.")
        new_dosages = st.text_area("Dosage & Administration:", value="Adults: 250 mg to 750 mg orally every 12 hours depending on infection severity. Avoid taking with dairy products or antacids containing magnesium/aluminum.")
        new_contra = st.text_area("Contraindications:", value="Hypersensitivity to ciprofloxacin or other quinolones. Concomitant administration with tizanidine. History of tendon rupture.")
        new_interactions = st.text_area("Drug & Food Interactions:", value="Co-administration with Theophylline leads to toxic theophylline accumulation. Multivalent cations (Calcium, Iron, Magnesium, Dairy) chelate ciprofloxacin, reducing absorption by up to 90%.")
        new_warnings = st.text_area("Black Box Warnings:", value="TENDON RUPTURE & TENDINITIS: Quinolones are associated with an increased risk of tendinitis and tendon rupture in all ages, especially the Achilles tendon. Avoid in myasthenia gravis.")

        btn_submit_ingest = st.form_submit_button("⚡ Ingest into pgvector Database")

    if btn_submit_ingest:
        new_doc = {
            "medication_name": new_med_name,
            "brand_names": [b.strip() for b in new_brands.split(",")],
            "therapeutic_class": new_class,
            "indications": new_indications,
            "dosage_and_administration": new_dosages,
            "contraindications": new_contra,
            "drug_interactions": new_interactions,
            "critical_warnings": new_warnings
        }
        
        with st.spinner(f"Chunking and generating embeddings for {new_med_name}..."):
            chunks = chunker.chunk_monograph(new_doc, verbose=False)
            st.info(f"Generated {len(chunks)} contextualized chunks.")

            for chk in chunks:
                vec = embedder.get_embedding(chk["content"], verbose=False)
                db.insert_chunk(
                    medication_name=chk["medication_name"],
                    section=chk["section"],
                    chunk_index=chk["chunk_index"],
                    content=chk["content"],
                    embedding=vec,
                    metadata=chk["metadata"],
                    verbose=False
                )

        st.success(f"🎉 Successfully ingested {len(chunks)} chunks for {new_med_name} into pgvector!")
        st.rerun()

# ----------------------------------------------------
# TAB 4: DATABASE EXPLORER (SQL VIEW)
# ----------------------------------------------------
with tab_explorer:
    st.subheader("Table: `public.medication_chunks`")
    st.markdown("Browse all records stored in pgvector, exactly like Oracle SQL Developer:")

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT medication_name FROM medication_chunks ORDER BY medication_name;")
            meds = [r[0] for r in cur.fetchall()]

    fcol1, fcol2 = st.columns([2, 1])
    selected_med = fcol1.selectbox("Filter by Medication:", ["All"] + meds)
    limit = fcol2.slider("Max Rows to Show:", 5, 100, 30, step=5)

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            if selected_med == "All":
                cur.execute("""
                    SELECT id::text, medication_name, section, chunk_index,
                           vector_dims(embedding) as dims,
                           substring(embedding::text from 1 for 40) || '...]' as vector_preview,
                           content, created_at
                    FROM medication_chunks
                    ORDER BY medication_name, chunk_index
                    LIMIT %s;
                """, (limit,))
            else:
                cur.execute("""
                    SELECT id::text, medication_name, section, chunk_index,
                           vector_dims(embedding) as dims,
                           substring(embedding::text from 1 for 40) || '...]' as vector_preview,
                           content, created_at
                    FROM medication_chunks
                    WHERE medication_name = %s
                    ORDER BY chunk_index
                    LIMIT %s;
                """, (selected_med, limit))
            rows = cur.fetchall()

    if rows:
        df = pd.DataFrame(rows, columns=[
            "UUID (Primary Key)", "Medication", "Section", "Chunk #",
            "Vector Dims", "Embedding (vector)", "Content Snippet", "Created At"
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No records found in database.")
