"""Study constants and serializable run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SEED = 42
DEVELOPMENT_CAP = 100_000
PCA_COMPONENTS = (2, 3, 5, 10, 11)

PAPER_REDUCTIONS = {
    "minimal": {
        "Reconnaissance": 0.975,
        "Discovery": 0.0,
        "Multinomial": 0.97,
    },
    "pca": {
        "Reconnaissance": 0.975,
        "Discovery": 0.10,
        "Multinomial": 0.95,
    },
    "lda": {
        "Reconnaissance": 0.925,
        "Discovery": 0.05,
        "Multinomial": 0.97,
    },
}

LOW_CARDINALITY_CATEGORICAL = ("service", "proto", "conn_state", "history")
HIGH_CARDINALITY_CATEGORICAL = ("src_ip_zeek", "dest_ip_zeek", "community_id")

DATASETS = {
    "UWF-ZeekData22": {
        "directory": "UWF-ZeekData22",
        "binary_tactics": ("Reconnaissance", "Discovery"),
        "multinomial_tactics": ("Reconnaissance", "Discovery"),
        "expected_files": 8,
        "expected_rows": 18_562_468,
        "expected_counts": {
            "none": 9_281_599,
            "Reconnaissance": 9_278_722,
            "Discovery": 2_086,
            "Credential Access": 31,
            "Privilege Escalation": 13,
            "Exfiltration": 7,
            "Lateral Movement": 4,
            "Resource Development": 3,
            "Defense Evasion": 1,
            "Initial Access": 1,
            "Persistence": 1,
        },
    },
    "UWF-ZeekDataFall22": {
        "directory": "UWF-ZeekDataFall22",
        "binary_tactics": (
            "Reconnaissance",
            "Discovery",
            "Defense Evasion",
            "Privilege Escalation",
            "Resource Development",
        ),
        "multinomial_tactics": (
            "Reconnaissance",
            "Discovery",
            "Defense Evasion",
            "Privilege Escalation",
            "Resource Development",
        ),
        "expected_files": 13,
        "expected_rows": 700_340,
        "expected_counts": {
            "none": 350_339,
            "Resource Development": 275_471,
            "Reconnaissance": 51_492,
            "Discovery": 16_819,
            "Privilege Escalation": 3_066,
            "Defense Evasion": 3_064,
            "Execution": 30,
            "Initial Access": 19,
            "Command and Control": 17,
            "Lateral Movement": 11,
            "Persistence": 10,
            "Collection": 1,
            "Credential Access": 1,
        },
    },
}


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    output_root: Path
    stage: str
    profile: str
    method: str
    datasets: tuple[str, ...]
    seed: int = SEED
    development_cap: int = DEVELOPMENT_CAP
    repetitions: int = 3
    warmup: bool = True
    feature_policy: str | None = None
    svm_max_iter: int | None = None
    svm_threshold: float | None = None
    minimal_scaling: str | None = None
    diagnostic_sweep: bool = False
    sensitivity_run: bool = False

    def protocol(self) -> dict:
        """Resolve immutable, serializable settings for the selected study track."""
        if self.method == "paper-compatible":
            defaults = {
                "split_strategy": "paper-seeded",
                "preprocessing_fit_scope": "train+test",
                "feature_policy": "legacy-ordinal",
                "svm_max_iter": 10,
                "svm_threshold": 0.00001,
                "minimal_scaling": "scaled",
            }
        else:
            defaults = {
                "split_strategy": "deterministic-stratified",
                "preprocessing_fit_scope": "train-only",
                "feature_policy": "corrected",
                "svm_max_iter": 100,
                "svm_threshold": 0.0,
                "minimal_scaling": "scaled",
            }
        for name in ("feature_policy", "svm_max_iter", "svm_threshold", "minimal_scaling"):
            value = getattr(self, name)
            if value is not None:
                defaults[name] = value
        defaults.update(
            {
                "protocol_version": 2,
                "method": self.method,
                "seed": self.seed,
                "sensitivity_run": self.sensitivity_run,
            }
        )
        return defaults

    def to_dict(self) -> dict:
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        result["output_root"] = str(self.output_root)
        result["datasets"] = list(self.datasets)
        return result
