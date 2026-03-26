"""
LLM extraction utilities — prompt building and JSON extraction via Ollama.

v2: Few-shot examples per form type for dramatically better accuracy.
"""
import json
import re
import time
import requests

from config import OLLAMA_URL, LLM_TEMPERATURE, LLM_TIMEOUT


def build_schema_description(form_type: str, schemas: dict) -> str:
    """Build human-readable schema from form_schemas.json."""
    schema = schemas.get(form_type, {})
    lines = []
    for section, fields in schema.get("sections", {}).items():
        lines.append(f"\n### {section}")
        for field in fields:
            lines.append(f'  - "{field}"')
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES per form type
# ════════════════════════════════════════════════════════════

TRIAGE_FEW_SHOT = """
EXAMPLE — here is a triage transcript and the correct extraction:

TRANSCRIPT:
"Triage form. Recording ID AG-099. Patient ID AG-099. Gender female. Date of birth 22nd of May 1975. Age 49. Unmet clinical need: chronic lower back pain with sciatica, failed conservative management over two years. Treatments tried: physiotherapy 10 sessions with limited benefit, codeine 30mg as needed, gabapentin 300mg twice daily discontinued due to dizziness. Current medication: paracetamol 1g four times daily, ibuprofen 400mg as needed. Past medical history: mild asthma. Allergy: penicillin. Cautions and contraindications: none identified. Black market use: no previous cannabis exposure. Admin notes: referral from GP, all documents received. Suggestions: suitable candidate, recommend initial consultation. Accepted for consultation: yes. Need further information: no. If no further information needed reason: all documentation complete. Indication: chronic pain. Eligible for treatment: yes. Clinician: Dr Smith, clinician ID."

CORRECT OUTPUT:
{
  "sections": {
    "PATIENT DETAILS": {
      "Patient ID": {"value": "AG-099", "confidence": "high"},
      "Gender": {"value": "Female", "confidence": "high"},
      "Date of Birth": {"value": "22-05-1975", "confidence": "high"},
      "Age": {"value": "49", "confidence": "high"}
    },
    "CLINICAL ASSESSMENT": {
      "Unmet Clinical Need": {"value": "Chronic lower back pain with sciatica, failed conservative management over two years", "confidence": "high"},
      "Treatments Tried": {"value": "Physiotherapy — 10 sessions with limited benefit. Codeine 30mg as needed. Gabapentin 300mg twice daily — discontinued due to dizziness.", "confidence": "high"},
      "Current Medication": {"value": "Paracetamol 1g four times daily. Ibuprofen 400mg as needed.", "confidence": "high"},
      "Past Medical History (PMHx)": {"value": "Mild asthma", "confidence": "high"},
      "Allergy": {"value": "Penicillin", "confidence": "high"},
      "Cautions and Contraindications": {"value": "None identified", "confidence": "high"},
      "Black-Market Use": {"value": "No previous cannabis exposure", "confidence": "high"}
    },
    "ADMIN AND CLINICAL NOTES": {
      "Admin Notes": {"value": "Referral from GP, all documents received", "confidence": "high"},
      "Suggestions": {"value": "Suitable candidate, recommend initial consultation", "confidence": "high"}
    },
    "TRIAGE DECISION": {
      "Accepted for Consultation?": {"value": "Yes", "confidence": "high"},
      "Need further information?": {"value": "No", "confidence": "high"},
      "If no further information needed, reason": {"value": "All documentation complete", "confidence": "high"},
      "If further information needed, what is required and actions": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Indication": {"value": "Chronic pain", "confidence": "high"},
      "Eligible for treatment?": {"value": "Yes", "confidence": "high"}
    },
    "CLINICIAN": {
      "Doctor Name / Medical Prescriber": {"value": "Dr Smith", "confidence": "high"}
    }
  }
}
"""

INITIAL_FEW_SHOT = """
EXAMPLE — here is a partial initial consultation transcript and the correct extraction:

TRANSCRIPT:
"Initial consultation form, voice note reading card. Recording ID AG-099. Clinician, clinician 5. Patient ID AG-099. Gender male. Date of birth 14th of March 1968. Age 56. ID checked yes. Patient consents to proceed with consultation yes. Safeguarding concerns no. History of psychosis or schizophrenia no. History of CVD or arrhythmias no. History of liver disease no. History of drug abuse no. Condition with unmet clinical need: fibromyalgia, widespread musculoskeletal pain with significant fatigue, referred by GP following rheumatology review confirming diagnosis, multiple treatment options exhausted with limited sustained benefit. History of complaint and diagnosis: pain onset approximately 3 years ago following a workplace injury, progressive worsening despite conservative management, reports 7 out of 10 pain at rest, 9 out of 10 on movement, sleep severely disrupted. Other relevant medical history: no other significant medical history reported. Alcohol intake and smoking history: non-smoker, occasional social drinker approximately 2 units per week. Social and work history: lives with partner and two children, works part time in administration, full driving licence held, advised regarding CBMP and driving. Current medication list: naproxen 500mg twice daily, omeprazole 20mg once daily. Treatment trialled unmet clinical need criteria: physiotherapy 12 sessions partial benefit only, amitriptyline discontinued due to side effects including drowsiness and dry mouth, naproxen partial benefit only. Allergies: no known drug allergies. Previous cannabis exposure yes. Type of flower previously used: indica dominant hybrid, illicit source, strain unknown. Amount of flower used: approximately 0.5g per day. Frequency of use: daily, evening use primarily. Was it effective: yes, reports significant improvement in sleep and moderate reduction in pain. Patient can understand information and retain it relevant to medical cannabis yes. Patient can make an informed decision yes. Patient has mental capacity yes. Suitable when licensed medications have been tried but are not useful yes. Patient should take caution when driving and using heavy machinery yes. Patient should take caution drinking alcohol when taking medical cannabis yes. Patient understands illicit cannabis may prevent treatment with CBPMs yes. Patient should not become pregnant whilst on medical cannabis yes. Relevant clinical guidelines taken into account yes. Medical cannabis side effects explained to patient yes. Patient is suitable for cannabis yes. MDT process and prescription subject to approval explained yes. Prescribed product 1: Curaleaf WPT 24% THC less than 1% CBD flos. Dosage instructions: vapourise up to 2g a day. Quantity: 60 grams. Additional plan: trial balanced oil, review in 4 weeks. Additional notes and considerations: patient engaged well with consultation, good understanding of treatment plan and associated risks. Clinician ID."

CORRECT OUTPUT:
{
  "sections": {
    "PATIENT DETAILS": {
      "Patient ID": {"value": "AG-099", "confidence": "high"},
      "Gender": {"value": "Male", "confidence": "high"},
      "Date of Birth": {"value": "14-03-1968", "confidence": "high"},
      "Age": {"value": "56", "confidence": "high"}
    },
    "PATIENT CHECK": {
      "ID checked?": {"value": "Yes", "confidence": "high"},
      "Patient consents to proceed with consultation?": {"value": "Yes", "confidence": "high"}
    },
    "TRIAGE SECTION": {
      "Safeguarding Concerns?": {"value": "No", "confidence": "high"},
      "History of Psychosis or Schizophrenia?": {"value": "No", "confidence": "high"},
      "History of CVD or Arrhythmias?": {"value": "No", "confidence": "high"},
      "History of Liver Disease?": {"value": "No", "confidence": "high"},
      "History of Drug Abuse?": {"value": "No", "confidence": "high"}
    },
    "MEDICAL HISTORY": {
      "Condition with unmet clinical need": {"value": "Fibromyalgia — widespread musculoskeletal pain with significant fatigue. Referred by GP following rheumatology review confirming diagnosis. Multiple treatment options exhausted with limited sustained benefit.", "confidence": "high"},
      "History of complaint and diagnosis": {"value": "Pain onset approximately 3 years ago following a workplace injury. Progressive worsening despite conservative management. Reports 7/10 pain at rest, 9/10 on movement. Sleep severely disrupted.", "confidence": "high"},
      "Other relevant medical history": {"value": "No other significant medical history reported.", "confidence": "high"},
      "Alcohol intake and smoking history": {"value": "Non-smoker. Occasional social drinker — approximately 2 units per week.", "confidence": "high"},
      "Social and work history": {"value": "Lives with partner and two children. Works part-time in administration. Full driving licence held — advised regarding CBMP and driving.", "confidence": "high"}
    },
    "CURRENT MEDICATION": {
      "Current medication list": {"value": "Naproxen 500mg twice daily. Omeprazole 20mg once daily.", "confidence": "high"},
      "Treatment trialed — unmet clinical need criteria": {"value": "Physiotherapy — 12 sessions, partial benefit only. Amitriptyline — discontinued due to side effects including drowsiness and dry mouth. Naproxen — partial benefit only.", "confidence": "high"},
      "Allergies": {"value": "No known drug allergies.", "confidence": "high"}
    },
    "PREVIOUS CANNABIS EXPOSURE": {
      "Previous cannabis exposure?": {"value": "Yes", "confidence": "high"},
      "Type of flower previously used": {"value": "Indica-dominant hybrid — illicit source, strain unknown", "confidence": "high"},
      "Amount of flower used": {"value": "Approximately 0.5g per day", "confidence": "high"},
      "Frequency of use": {"value": "Daily — evening use primarily", "confidence": "high"},
      "Was it effective?": {"value": "Yes — reports significant improvement in sleep and moderate reduction in pain", "confidence": "high"}
    },
    "ADDITIONAL QUESTIONS — PATIENT CAPACITY": {
      "Patient can understand information and retain it relevant to medical cannabis?": {"value": "Yes", "confidence": "high"},
      "Patient can make an informed decision?": {"value": "Yes", "confidence": "high"},
      "Patient has mental capacity?": {"value": "Yes", "confidence": "high"},
      "Suitable when licensed medications have been tried but are not useful?": {"value": "Yes", "confidence": "high"},
      "Patient should take caution when driving and using heavy machinery?": {"value": "Yes", "confidence": "high"},
      "Patient should take caution drinking alcohol when taking medical cannabis?": {"value": "Yes", "confidence": "high"},
      "Patient understands illicit cannabis may prevent treatment with CBPMs?": {"value": "Yes", "confidence": "high"},
      "Patient should not become pregnant whilst on medical cannabis?": {"value": "Yes", "confidence": "high"},
      "Relevant clinical guidelines taken into account?": {"value": "Yes", "confidence": "high"},
      "Medical cannabis side effects explained to patient?": {"value": "Yes", "confidence": "high"},
      "Patient is suitable for cannabis?": {"value": "Yes", "confidence": "high"},
      "MDT process and prescription subject to approval explained?": {"value": "Yes", "confidence": "high"}
    },
    "CBMP PRESCRIBED": {
      "Prescribed Product 1": {"value": "CuraleafWPT: 24% THC, <1% CBD Flos", "confidence": "high"},
      "Dosage Instructions": {"value": "Vapourise up to 2g a day", "confidence": "high"},
      "Quantity": {"value": "60 grams", "confidence": "high"},
      "Prescribed Product 2": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Number of Bottles": {"value": "NOT_MENTIONED", "confidence": "low"}
    },
    "ADDITIONAL INFORMATION": {
      "Additional plan": {"value": "Trial balanced oil. Review in 4 weeks.", "confidence": "high"},
      "Additional notes and considerations": {"value": "Patient engaged well with consultation. Good understanding of treatment plan and associated risks.", "confidence": "high"},
      "Additional info": {"value": "NOT_MENTIONED", "confidence": "low"}
    },
    "CLINICIAN": {
      "Doctor Name / Medical Prescriber": {"value": "Clinician 5", "confidence": "medium"}
    }
  }
}
"""

FOLLOW_UP_FEW_SHOT = """
EXAMPLE — here is a partial follow-up transcript and the correct extraction for key sections:

TRANSCRIPT:
"Follow up consultation form. Recording ID AG-099. Patient ID AG-099. Gender female. Date of birth 5th of July 1982. Age 42. Follow up year 2024. Is this the current treatment month and year from the treatment tracker yes. ID checked yes. Was this appointment booked to provide more evidence post rejection at initial consultation no. Previous medication 1 Curaleaf WPT 24% THC less than 1% CBD flos. Patient reviewed health score yes. Detail changes in outcome and treatment goals: patient reports improved sleep quality and reduction in pain from 8 out of 10 to 5 out of 10. Detail changes in additional symptoms: fatigue improved. Detail changes in mood: stable. Detail changes in anxiety: mild improvement. Has the patient experienced any side effects: yes. If yes please describe side effects: mild dry mouth and occasional drowsiness. Any new medical treatment from GP or hospital: no. Patient can understand information relevant to medical cannabis yes. Patient can retain relevant information yes. Patient can weigh up information to make a decision yes. Patient can communicate their decision yes. Patient has mental capacity yes. Are there any safeguarding concerns raised in this consultation no. Next step: continue current treatment. Prescribed product 1: Curaleaf WPT 24% THC less than 1% CBD flos. Dosage instructions: vapourise up to 2g a day. Quantity: 60 grams. Plan of action and follow up: continue current regime, review in 3 months. Next follow up: 3 months. Patient consents for prescription to be sent to pharmacy yes. Clinician 3."

CORRECT OUTPUT (showing key sections):
{
  "sections": {
    "PATIENT DETAILS": {
      "Patient ID": {"value": "AG-099", "confidence": "high"},
      "Gender": {"value": "Female", "confidence": "high"},
      "Date of Birth": {"value": "05-07-1982", "confidence": "high"},
      "Age": {"value": "42", "confidence": "high"}
    },
    "APPOINTMENT DETAILS": {
      "Follow Up Year": {"value": "2024", "confidence": "high"},
      "Is this the current treatment month and year from the Treatment Tracker?": {"value": "Yes", "confidence": "high"},
      "ID checked?": {"value": "Yes", "confidence": "high"},
      "Was this appointment booked to provide more evidence post rejection at initial consultation?": {"value": "No", "confidence": "high"}
    },
    "PREVIOUS MEDICATION": {
      "Previous Medication 1": {"value": "CuraleafWPT: 24% THC, <1% CBD Flos", "confidence": "high"},
      "Previous Medication 2": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Previous Medication": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Previous Medication 3": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Previous Medication 4": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Previous Medication 1  |  Quantity": {"value": "NOT_MENTIONED", "confidence": "low"}
    },
    "OUTCOMES": {
      "Patient reviewed health score?": {"value": "Yes", "confidence": "high"},
      "Detail changes in outcome and treatment goals": {"value": "Patient reports improved sleep quality and reduction in pain from 8/10 to 5/10.", "confidence": "high"},
      "Detail changes in additional symptoms": {"value": "Fatigue improved.", "confidence": "high"},
      "Detail changes in mood": {"value": "Stable.", "confidence": "high"},
      "Detail changes in anxiety": {"value": "Mild improvement.", "confidence": "high"}
    },
    "SIDE EFFECTS AND NEW TREATMENTS": {
      "Has the patient experienced any side effects?": {"value": "Yes", "confidence": "high"},
      "If yes, please describe side effects": {"value": "Mild dry mouth and occasional drowsiness.", "confidence": "high"},
      "Any new medical treatment from GP or Hospital?": {"value": "No", "confidence": "high"}
    },
    "MENTAL CAPACITY": {
      "Patient can understand information relevant to medical cannabis?": {"value": "Yes", "confidence": "high"},
      "Patient can retain relevant information?": {"value": "Yes", "confidence": "high"},
      "Patient can weigh up information to make a decision?": {"value": "Yes", "confidence": "high"},
      "Patient can communicate their decision?": {"value": "Yes", "confidence": "high"},
      "Patient has mental capacity?": {"value": "Yes", "confidence": "high"}
    },
    "SAFEGUARDING": {
      "Are there any safeguarding concerns raised in this consultation?": {"value": "No", "confidence": "high"},
      "If yes, please describe": {"value": "NOT_MENTIONED", "confidence": "low"}
    },
    "NEXT STEPS": {
      "Next step": {"value": "Continue current treatment", "confidence": "high"}
    },
    "CBMP PRESCRIBED": {
      "Prescribed Product 1": {"value": "CuraleafWPT: 24% THC, <1% CBD Flos", "confidence": "high"},
      "Dosage Instructions": {"value": "Vapourise up to 2g a day", "confidence": "high"},
      "Quantity": {"value": "60 grams", "confidence": "high"},
      "Number of Bottles / Units": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Prescribed Product 2": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Prescribed Product": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Prescribed Product 3": {"value": "NOT_MENTIONED", "confidence": "low"}
    },
    "TREATMENT PLAN": {
      "If only flower prescribed, does patient have background CBD on board?": {"value": "NOT_MENTIONED", "confidence": "low"},
      "If new medication prescribed, is THC new into the treatment plan?": {"value": "NOT_MENTIONED", "confidence": "low"},
      "If patient receiving greater than 60g of flower, must go through MDT?": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Plan of action and follow up": {"value": "Continue current regime, review in 3 months.", "confidence": "high"},
      "Next follow up": {"value": "3 months", "confidence": "high"},
      "Patient consents for prescription to be sent to pharmacy?": {"value": "Yes", "confidence": "high"},
      "Patient is able to have repeats and re-writes written by shared care colleagues?": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Patient is suitable for shared care check ups in between specialist reviews?": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Potential risk category": {"value": "NOT_MENTIONED", "confidence": "low"}
    },
    "ADDITIONAL INFORMATION": {
      "Additional information": {"value": "NOT_MENTIONED", "confidence": "low"}
    },
    "CLINICIAN": {
      "Doctor Name / Medical Prescriber": {"value": "Clinician 3", "confidence": "medium"}
    },
    "CURRENT MEDICATION AND MEDICAL HISTORY": {
      "Current medication": {"value": "NOT_MENTIONED", "confidence": "low"},
      "Past medical history (PMHx)": {"value": "NOT_MENTIONED", "confidence": "low"}
    }
  }
}
"""

FEW_SHOT_BY_TYPE = {
    "initial": INITIAL_FEW_SHOT,
    "follow_up": FOLLOW_UP_FEW_SHOT,
    "triage": TRIAGE_FEW_SHOT,
}


# ════════════════════════════════════════════════════════════
# Prompt builders
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a medical clinical form extraction system. You listen to consultation transcripts and extract structured data to populate specific medical forms.

CRITICAL RULES — follow these exactly:

1. OUTPUT: Return ONLY a valid JSON object. No text before or after. No markdown code fences. No explanations.

2. FIELD NAMES: Copy field names EXACTLY from the schema. Do not rename, abbreviate, or rephrase them. "History of CVD or Arrhythmias?" must appear exactly as "History of CVD or Arrhythmias?" — not "CVD History" or "Arrhythmias".

3. ALL SECTIONS REQUIRED: Your output must include EVERY section and EVERY field from the schema, even if the transcript doesn't mention them. Use "NOT_MENTIONED" for missing fields.

4. YES/NO FIELDS: Output exactly "Yes" or "No". Not "yes", not "Y", not "true".

5. DATES: Format as DD-MM-YYYY. Convert spoken dates: "14th of March 1968" becomes "14-03-1968". "5th of July 1982" becomes "05-07-1982".

6. PATIENT ID: Extract the AG-XXX identifier exactly as spoken. "AG-001" or "AG 001" becomes "AG-001".

7. LONG TEXT FIELDS: For fields like "Condition with unmet clinical need" or "History of complaint and diagnosis", extract the FULL clinical detail — do not summarize or truncate. Include all symptoms, durations, severity scores (e.g., "7/10 pain"), and clinical findings mentioned.

8. MEDICATIONS: Extract drug name, dose, and frequency exactly. "Naproxen 500mg twice daily" — not "Naproxen" alone. Include ALL medications mentioned, separated by periods.

9. PRODUCTS: Cannabis product names should include the full specification: "CuraleafWPT: 24% THC, <1% CBD Flos" — not just "Curaleaf".

10. CONFIDENCE: Rate each field "high" (clearly stated), "medium" (implied or partially stated), or "low" (guessing or not mentioned).

11. NEVER INVENT: If information is not in the transcript, output "NOT_MENTIONED". Never guess patient IDs, dates, or dosages.

12. LANGUAGE: The transcript may be in any language. Always extract and output values in English, regardless of the input language. Translate field values to English where necessary.

13. NUMBERS: Convert spoken numbers to digits. "Two grams" becomes "2 grams". "Thirty milligrams" becomes "30 milligrams". "Seven out of ten pain" becomes "7/10 pain".
"""


def build_extraction_prompt(form_type: str, transcript: str, schemas: dict):
    """Build system + user prompt with few-shot example for the specific form type."""
    schema_desc = build_schema_description(form_type, schemas)
    form_display = form_type.replace("_", " ").title()
    few_shot = FEW_SHOT_BY_TYPE.get(form_type, "")

    user_prompt = f"""Extract data for a **{form_display} Consultation Form**.

TARGET SCHEMA — you must extract exactly these fields, using these exact names:
{schema_desc}

{few_shot}

NOW EXTRACT FROM THIS TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

Remember: output ONLY the JSON object with ALL sections and ALL fields from the schema. Use "NOT_MENTIONED" for any field not found in the transcript."""

    return SYSTEM_PROMPT, user_prompt


# ════════════════════════════════════════════════════════════
# Ollama API
# ════════════════════════════════════════════════════════════

def clean_json_response(text: str) -> str:
    """Extract JSON from LLM response, stripping markdown etc."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first:last + 1]
    return text


def extract_with_ollama(model_name: str, system_prompt: str, user_prompt: str) -> dict:
    """Send extraction request to Ollama and parse response."""
    start = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": 8192,   # Doubled from 4096 — full form JSON is large
                    "top_p": 0.9,
                },
            },
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        elapsed = time.time() - start

        cleaned = clean_json_response(raw)
        try:
            parsed = json.loads(cleaned)
            return {
                "extracted_json": parsed,
                "raw_response": raw,
                "elapsed_sec": round(elapsed, 1),
                "success": True,
                "error": None,
            }
        except json.JSONDecodeError as e:
            return {
                "extracted_json": None,
                "raw_response": raw,
                "elapsed_sec": round(elapsed, 1),
                "success": False,
                "error": f"JSON parse error: {e}",
            }

    except requests.exceptions.Timeout:
        return {"extracted_json": None, "raw_response": None,
                "elapsed_sec": round(time.time() - start, 1),
                "success": False, "error": "Timeout"}
    except Exception as e:
        return {"extracted_json": None, "raw_response": None,
                "elapsed_sec": round(time.time() - start, 1),
                "success": False, "error": str(e)}
