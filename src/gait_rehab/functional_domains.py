from typing import Dict, Any, List

VALID_MAPPING_STRENGTHS = {"strong", "moderate", "weak"}
VALID_CLAIM_LEVELS = {"candidate", "consistent_with", "hypothesis_only"}

# This is the single source of truth for the functional-domain interpretation contract.
FEATURE_DOMAIN_MAPPING = {
    "push_off_index": {
        "primary_domain": "push_off_propulsion",
        "secondary_domains": [],
        "mapping_strength": "strong",
        "claim_level": "hypothesis_only",
        "caveat": "Not direct muscle evidence; affected by speed and pain avoidance.",
        "allowed_template": "consistent with a shared push-off / propulsion functional-domain hypothesis",
        "forbidden_terms": ["proves", "confirms", "diagnosis", "caused by", "원인은", "입증"],
        "requires_gait_only_evidence": True
    },
    "ap_propulsion_impulse_asym": {
        "primary_domain": "push_off_propulsion",
        "secondary_domains": [],
        "mapping_strength": "strong",
        "claim_level": "hypothesis_only",
        "caveat": "Direction depends on side mapping quality.",
        "allowed_template": "candidate for propulsion deficit",
        "forbidden_terms": ["proves", "confirms", "diagnosis", "caused by"],
        "requires_gait_only_evidence": True
    },
    "loading_rate_asym": {
        "primary_domain": "loading_response_weight_acceptance",
        "secondary_domains": [],
        "mapping_strength": "moderate",
        "claim_level": "hypothesis_only",
        "caveat": "Loading can reflect compensation, speed, or guarding.",
        "allowed_template": "consistent with loading response alterations",
        "forbidden_terms": ["proves", "diagnosis", "환자 토크 추정"],
        "requires_gait_only_evidence": True
    },
    "vgrf_peak_aff": {
        "primary_domain": "loading_response_weight_acceptance",
        "secondary_domains": [],
        "mapping_strength": "weak",
        "claim_level": "candidate",
        "caveat": "Strongly confounded by speed, body mass, and pain avoidance.",
        "allowed_template": "may indicate generalized weight acceptance issues",
        "forbidden_terms": ["proves", "confirms", "처방한다", "진단한다"],
        "requires_gait_only_evidence": True
    },
    "ap_braking_impulse_asym": {
        "primary_domain": "braking_early_stance_control",
        "secondary_domains": [],
        "mapping_strength": "moderate",
        "claim_level": "hypothesis_only",
        "caveat": "Braking can overlap with loading response.",
        "allowed_template": "consistent with early stance control hypotheses",
        "forbidden_terms": ["원인 확정", "proves"],
        "requires_gait_only_evidence": True
    },
    "cop_path_length_aff": {
        "primary_domain": "stability_weight_shift",
        "secondary_domains": [],
        "mapping_strength": "moderate",
        "claim_level": "hypothesis_only",
        "caveat": "COP path can reflect compensation rather than primary instability.",
        "allowed_template": "candidate for stability and weight shift observation",
        "forbidden_terms": ["confirms", "proves"],
        "requires_gait_only_evidence": True
    },
    "cop_ap_range_aff": {
        "primary_domain": "stability_weight_shift",
        "secondary_domains": ["rollover_progression"],
        "mapping_strength": "moderate",
        "claim_level": "hypothesis_only",
        "caveat": "Also related to rollover and progression.",
        "allowed_template": "consistent with rollover or weight shift alterations",
        "forbidden_terms": ["proves", "diagnosis"],
        "requires_gait_only_evidence": True
    },
    "cop_ml_range_aff": {
        "primary_domain": "stability_weight_shift",
        "secondary_domains": [],
        "mapping_strength": "moderate",
        "claim_level": "hypothesis_only",
        "caveat": "Sensitive to foot placement and measurement conditions.",
        "allowed_template": "candidate marker for ML stability hypothesis",
        "forbidden_terms": ["proves", "diagnosis", "근약화 확인"],
        "requires_gait_only_evidence": True
    }
}

def get_feature_domain_mapping() -> Dict[str, Dict[str, Any]]:
    return FEATURE_DOMAIN_MAPPING

def validate_feature_domain_mapping(mapping: Dict[str, Dict[str, Any]]) -> bool:
    required_keys = {
        "primary_domain", "mapping_strength", "claim_level", 
        "caveat", "allowed_template", "forbidden_terms", "requires_gait_only_evidence"
    }
    
    for feature, meta in mapping.items():
        missing = required_keys - set(meta.keys())
        if missing:
            raise ValueError(f"Feature '{feature}' is missing required mapping keys: {missing}")
            
        if meta["mapping_strength"] not in VALID_MAPPING_STRENGTHS:
            raise ValueError(f"Feature '{feature}' has invalid mapping_strength: {meta['mapping_strength']}")
            
        if meta["claim_level"] not in VALID_CLAIM_LEVELS:
            raise ValueError(f"Feature '{feature}' has invalid claim_level: {meta['claim_level']}")
            
    return True

def validate_no_siat_features_in_model_inputs(feature_names: List[str]) -> bool:
    forbidden_substrings = ["siat", "emg", "torque", "phase", "lag", "functional_domain"]
    
    for f in feature_names:
        lower_f = f.lower()
        for forbidden in forbidden_substrings:
            if forbidden in lower_f:
                raise ValueError(f"Forbidden SIAT-related feature found in model inputs: {f}")
    return True

def map_features_to_domains(feature_summary: List[str], mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mapped_features = {}
    for f in feature_summary:
        if f in mapping:
            mapped_features[f] = mapping[f]
        else:
            mapped_features[f] = {"status": "unmapped"}
    return mapped_features
