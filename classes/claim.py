# Imports from external libraries

# Imports from internal libraries
from util.general_util import normalize_nfd

class Claim:
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NEI = "NOT ENOUGH INFO"

    def __init__(self, id: int, label: str, claim: str, evidence: list):
        assert label in [Claim.SUPPORTS, Claim.REFUTES, Claim.NEI]

        self.id = id
        self.label = label
        self.claim = normalize_nfd(claim)
        self.evidence = evidence


    def _clear_evidence(self, evidence: list):
        return None
