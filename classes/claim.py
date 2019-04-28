# Imports from external libraries
from typing import List

# Imports from internal libraries
from util.general_util import normalize_nfd
from classes.wiki_article import WikiArticle

class Claim:
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NEI = "NOT ENOUGH INFO"

    def __init__(self, id: str, label: str, claim: str, evidence: list, decode_brackets=True):
        assert label in [Claim.SUPPORTS, Claim.REFUTES, Claim.NEI]

        self.id = str(id)
        self.label = label
        self.claim = normalize_nfd(claim)
        self.evidence = self._clear_evidence(evidence, decode_brackets)

    def _clear_evidence(self, all_evidence: list, decode_brackets):
        if self.verifiable():
            return [[(evidence[2] if not decode_brackets else WikiArticle._decode_bracket_tokens(evidence[2]), int(evidence[3])) for evidence in set_of_evidences] for set_of_evidences in all_evidence]
        else:
            return None

    def verifiable(self) -> bool:
        return self.label in [Claim.SUPPORTS, Claim.REFUTES]

    def all_evidence_documents(self) -> set:
        documents = set()
        for doc_set in self.evidence:
            documents.update({doc[0] for doc in doc_set})
        return documents

    def evidence_document_sets(self) -> List[set]:
        document_sets = []
        for doc_set in self.evidence:
            document_sets.append({doc[0] for doc in doc_set})
        return document_sets

    @staticmethod
    def from_json(json_dump, decode_brackets=True):
        return Claim(json_dump['id'], json_dump['label'], json_dump['claim'], json_dump['evidence'], decode_brackets)

    def __str__(self):
        return f"{self.label}: {self.claim}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

