import pytest
from gait_rehab.functional_domains import (
    get_feature_domain_mapping, 
    validate_feature_domain_mapping,
    validate_no_siat_features_in_model_inputs
)
from gait_rehab.validators import validate_report_claims

def test_feature_domain_mapping_is_valid():
    mapping = get_feature_domain_mapping()
    assert validate_feature_domain_mapping(mapping) is True

def test_validate_no_siat_features_rejects_siat():
    features = ["age", "vgrf_peak", "siat_emg_mean"]
    with pytest.raises(ValueError, match="Forbidden SIAT-related feature"):
        validate_no_siat_features_in_model_inputs(features)

def test_validate_no_siat_features_accepts_clean():
    features = ["age", "vgrf_peak", "push_off_index"]
    assert validate_no_siat_features_in_model_inputs(features) is True

def test_report_validator_rejects_forbidden_english():
    text = "The model diagnosis is clear. This proves the cause."
    violations = validate_report_claims(text)
    assert "diagnosis" in violations
    assert "proves" in violations

def test_report_validator_rejects_forbidden_korean():
    text = "SIAT가 환자의 근약화 확인을 했습니다. 이것이 원인 확정입니다."
    violations = validate_report_claims(text)
    assert "근약화 확인" in violations
    assert "원인 확정" in violations

def test_report_validator_accepts_cautious_language():
    text = "The evidence is consistent with a hypothesis of propulsion deficit."
    violations = validate_report_claims(text)
    assert not violations
