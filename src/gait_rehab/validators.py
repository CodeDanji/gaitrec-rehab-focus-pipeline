import re
from typing import List

FORBIDDEN_TERMS = [
    # English
    "diagnosis", "prescription", "cause confirmed", "patient torque estimation",
    "confirmed muscle weakness", "fully explained by", "proves", "confirms", 
    "caused by", "mechanism is", "improves classifier", "improved classifier",
    "patient's torque", "patient's emg abnormality",
    # Korean
    "원인은", "원인이다", "원인으로 확인", "원인 확정", "확정했다", "입증", 
    "증명", "진단한다", "처방한다", "근약화 확인", "진단", "처방", "환자 토크 추정"
]

def validate_report_claims(text: str) -> List[str]:
    """
    Scans text for forbidden diagnostic, causal, or definitive language.
    Returns a list of violations found.
    """
    violations = []
    
    # Case-insensitive search
    lower_text = text.lower()
    for term in FORBIDDEN_TERMS:
        if term.lower() in lower_text:
            violations.append(term)
            
    return violations
