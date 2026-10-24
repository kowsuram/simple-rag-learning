"""Clinical Healthcare RAG Agent with prompt guardrails and configurable AI generation."""

import time
from typing import Dict, Any, Optional
import requests
from src.config import config
from src.retriever import retriever
from src.logger import (
    console,
    log_header,
    log_step,
    log_augmented_prompt,
    log_llm_response
)

SYSTEM_GUARDRAILS = """You are a specialized Clinical Medication Reference Assistant. Your role is to examine the provided official Medication Reference Data and provide factual, document-grounded answers to patient inquiries.

RULES:
1. GROUNDING IN SOURCE OF TRUTH: Answer strictly based on the provided medication monographs. State what the official monograph specifies regarding indications, contraindications, dosages, and warnings.
2. UNKNOWN MEDICATIONS: If the patient inquires about a medication or details not found in the reference chunks, state explicitly: "The requested medication or specific detail is not found in the approved clinical medication reference database." Do not guess or fabricate information.
3. CONTRAINDICATIONS & SAFETY: Always clearly state absolute contraindications (such as active peptic ulcers, pregnancy, severe hepatic/renal impairment), maximum recommended daily doses, and serious drug interactions.
4. FORMAT: Use clear bullet points for Indications, Dosages, Contraindications/Warnings, and Next Steps.
5. MANDATORY DISCLAIMER: Conclude every answer with:
"Clinical Notice: This guidance is based on approved medication monographs and is intended for informational/educational purposes only. Always consult a licensed physician or pharmacist before altering any medication regimen."
"""


class HealthcareAgent:
    """Clinical RAG Agent coordinating retrieval, prompt synthesis, and LLM inference."""

    def __init__(self, llm_model: Optional[str] = None):
        self.provider = config.ai_provider
        self.llm_model = llm_model or config.llm_model

    def _generate_chat_response(self, messages: list[dict[str, str]], temperature: float) -> str:
        if self.provider == "cloudflare":
            endpoint = f"{config.cloudflare_ai_base_url}/chat/completions"
            headers = config.cloudflare_headers
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
        else:
            endpoint = f"{config.ollama_base_url.rstrip('/')}/api/chat"
            headers = None
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to query {self.provider} chat API at {endpoint}."
            ) from e

        if self.provider == "cloudflare":
            choices = response_json.get("choices", [])
            answer = choices[0].get("message", {}).get("content", "") if choices else ""
        else:
            answer = response_json.get("message", {}).get("content", "")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"{self.provider} returned no chat response for model '{self.llm_model}'."
            )
        return answer

    def ask(self, patient_query: str, verbose: bool = True) -> Dict[str, Any]:
        """Process a patient inquiry through the full RAG pipeline."""
        if verbose:
            log_header(
                "HEALTHCARE RAG: PATIENT QUERY EXECUTION",
                f"Patient Question: \"{patient_query}\""
            )

        # 1. Retrieve most relevant medication context chunks
        chunks = retriever.retrieve(query=patient_query, verbose=verbose)
        context_text = retriever.format_context_for_prompt(chunks)

        # 2. Step: Assemble Augmented Prompt
        if verbose:
            log_step(
                3,
                "Augmented Prompt Construction",
                "Injecting retrieved pgvector chunks and strict clinical guardrails into the LLM prompt."
            )
            log_augmented_prompt(
                retrieved_context=context_text,
                system_guardrails=SYSTEM_GUARDRAILS,
                user_question=patient_query
            )

        # 3. Step: Call Ollama LLM
        if verbose:
            log_step(
                4,
                f"LLM Generation ({self.llm_model})",
                f"Sending augmented context and question to {self.provider} '{self.llm_model}'."
            )

        user_content = (
            f"Based on the following official Medication Reference Data, provide a clear, factual answer to the patient's inquiry.\n\n"
            f"[OFFICIAL REFERENCE DATA]\n"
            f"{context_text}\n\n"
            f"[PATIENT INQUIRY]\n"
            f"{patient_query}\n\n"
            f"Answer the inquiry directly citing the relevant facts, dosages, or contraindications from the reference data above."
        )

        messages = [
            {"role": "system", "content": SYSTEM_GUARDRAILS},
            {"role": "user", "content": user_content}
        ]

        start_time = time.perf_counter()
        answer = self._generate_chat_response(messages, temperature=0.1)

        latency_ms = (time.perf_counter() - start_time) * 1000

        if verbose:
            log_llm_response(answer, latency_ms)

        return {
            "query": patient_query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "latency_ms": latency_ms,
            "model": self.llm_model
        }

    def audit_prescription(self, prescription: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Perform clinical safety screening on a patient prescription using pgvector RAG."""
        patient_name = prescription.get("patient_name", "Anonymous Patient")
        age = prescription.get("age", "Unknown")
        conditions = prescription.get("clinical_conditions", "None listed")
        allergies = prescription.get("known_allergies", "NKDA")
        current_meds = prescription.get("current_medications", "None")
        prescribed_rx = prescription.get("prescribed_rx", [])
        notes = prescription.get("notes", "")

        # Format prescription summary
        rx_lines = []
        for rx in prescribed_rx:
            rx_lines.append(
                f"- Drug: {rx.get('medication')} ({rx.get('brand', '')}) | "
                f"Dosage: {rx.get('dosage')} | Freq: {rx.get('frequency')} | Indication: {rx.get('indication', 'N/A')}"
            )
        rx_summary = "\n".join(rx_lines)

        # 1. Retrieve clinical context from pgvector for every prescribed medication
        all_retrieved_chunks = []
        seen_chunk_ids = set()

        for rx in prescribed_rx:
            drug_name = rx.get("medication", "")
            # Query pgvector for this drug's contraindications, dosage, interactions
            queries = [
                f"{drug_name} contraindications risks warnings",
                f"{drug_name} maximum daily dose administration limits",
                f"{drug_name} drug food interactions {conditions} {allergies}"
            ]
            for q in queries:
                matches = retriever.retrieve(query=q, top_k=2, verbose=False)
                for m in matches:
                    if m["id"] not in seen_chunk_ids:
                        seen_chunk_ids.add(m["id"])
                        all_retrieved_chunks.append(m)

        # Sort all retrieved chunks by similarity
        all_retrieved_chunks.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        top_chunks = all_retrieved_chunks[:6]
        context_text = retriever.format_context_for_prompt(top_chunks)

        # 2. Construct Clinical Safety Screening Prompt
        eval_prompt = f"""You are a Senior Hospital Clinical Pharmacist and Safety Officer. Evaluate this prescription strictly against the provided authoritative Medication Reference Data.

[AUTHORITATIVE REFERENCE DATA FROM PGVECTOR]
{context_text}

[PATIENT RECORD & PRESCRIPTION TO AUDIT]
Patient: {patient_name} (Age: {age})
Clinical Conditions: {conditions}
Known Allergies: {allergies}
Current Medications / Habits: {current_meds}
Special Notes: {notes}

Prescription to Audit:
{rx_summary}

AUDIT INSTRUCTIONS:
Conduct a systematic clinical safety audit formatted with these clear sections:
1. CLINICAL SAFETY VERDICT: [APPROVED / CAUTION: DOSAGE ADJUSTMENT NEEDED / REJECTED: CONTRAINDICATED]
2. CRITICAL SAFETY ALERTS:
   - Identify any absolute contraindications against patient conditions or allergies.
   - Identify if maximum daily dosage limits are exceeded.
   - Identify severe drug-drug or food interactions (e.g. grapefruit juice, alcohol, other medications).
3. EVIDENCE FROM REFERENCE MONOGRAPHS: Quote the specific warnings or dosage limits from the reference data above.
4. PHARMACIST RECOMMENDATION: Provide concrete, safe clinical alternatives or next steps.
5. Conclude with: "Clinical Notice: This prescription screening is an informational safety check grounded in approved monographs. A licensed clinician must make final prescribing decisions."
"""

        start_time = time.perf_counter()
        audit_text = self._generate_chat_response(
            [
                {"role": "system", "content": "You are an expert Clinical Pharmacist conducting rigorous prescription safety audits grounded in approved pharmaceutical monographs."},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=0.1,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "patient_name": patient_name,
            "prescription": prescription,
            "audit_report": audit_text,
            "retrieved_chunks": top_chunks,
            "latency_ms": latency_ms,
            "model": self.llm_model
        }


# Global agent instance
agent = HealthcareAgent()
