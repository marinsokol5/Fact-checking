# Imports from external libraries
import re

# Imports from internal libraries
from util.general_util import normalize_nfd




class WikiArticle:
    """
    It's the hyperlinks in the Wikipedia page.
    Listed in pairs, the first entry is the surface form in the sentence,
    the second entry is the canonical wikipedia page


    In field lines for a wiki article not all lines start with numbers.
    We have found some lines that end with \t and then in the next line
    they are hyperlinks continued 
    """

    def __init__(self, id: str, text: str, lines: str, decode_brackets=True):
        id = normalize_nfd(id)
        lines = normalize_nfd(lines)

        if decode_brackets:
            id = WikiArticle._decode_bracket_tokens(id)
            lines = WikiArticle._decode_bracket_tokens(lines)

        self.id = id
        self.title = self.id.replace("_", " ")

        lines = self._remove_wrong_new_lines(lines)
        self.lines_dict = self._parse_lines(lines)
        self.hyperlinks_dict = self._parse_hyperlinks_dict(lines)
        self.hyperlinks = self._parse_hyperlinks(self.hyperlinks_dict)

    def _parse_lines(self, lines: str) -> dict:
        lines_parsed = [line for line in lines.split("\n") if len(line.strip()) > 1]
        tuples = [tuple(line.split("\t", 1)) for line in lines_parsed]
        return {int(id): line.strip().split("\t", 1)[0] for id, line in tuples if line}

    def _remove_wrong_new_lines(self, lines):
        lines = re.sub("\n+", "\n", lines)
        lines = re.sub("\n *\t", "\t", lines)
        while True:
            search_match = re.search("[0-9]+\t.+\t *\n", lines)
            if search_match is None:
                return lines
            corrupted_part = search_match.group()
            proper_part = re.sub("\t *\n", "\t", corrupted_part)
            lines = lines.replace(corrupted_part, proper_part)


    def _parse_hyperlinks_dict(self, lines: str) -> dict:
        lines_parsed = [line for line in lines.split("\n") if len(line.strip()) > 1]
        tuples = [tuple(line.split("\t", 1)) for line in lines_parsed]
        titles_dict = {int(id): (line.strip().split("\t", 1)[1] if len(line.strip().split("\t", 1)) == 2 else "") for id, line in tuples if line }
        titles_dict = {k: (set(v.split("\t")[1::2] if len(v) else set())) for k, v in titles_dict.items()}
        return {k: {s.replace(" ", "_") for s in v} for k, v in titles_dict.items()}

    def _parse_hyperlinks(self, hyperlinks_dict):
        hyperlinks = set()
        for v in hyperlinks_dict.values():
            hyperlinks.update(v)
        return hyperlinks

    @staticmethod
    def from_json(json_dump, decode_brackets=True):
        return WikiArticle(json_dump['id'], json_dump['text'], json_dump['lines'], decode_brackets)

    def __str__(self):
        return f"Wiki article: {self.title}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    @staticmethod
    def _decode_bracket_tokens(text: str) -> str:
        return text.replace("-LRB-", "(").replace("-RRB-", ")")

