"""Text normalisation for entity resolution (Phase 6). Country-agnostic."""
import re, unicodedata

# Legal-form suffixes only. Deliberately excludes "societe": it is a common French
# NAME word (Societe Generale), not a suffix, so folding it would corrupt 15% of test.
LEGAL = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd",
    "private": "pvt", "and": "&",
}
ADDR = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd", "drive": "dr",
    "lane": "ln", "court": "ct", "place": "pl", "square": "sq", "highway": "hwy",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg",
    "block": "blk", "sector": "sec", "north": "n", "south": "s", "east": "e",
    "west": "w", "saint": "st", "sainte": "st", "avenida": "ave",
}
JUNK = re.compile(r"^[^0-9a-z]+|[^0-9a-z]+$")
NONALNUM = re.compile(r"[^0-9a-z&]+")
WS = re.compile(r"\s+")
PIN = re.compile(r"^\d{5,6}$")
NUM = re.compile(r"\b\d+\b")


def fold(text):
    """Unicode-fold to ASCII and lowercase. Handles French accents, transliterations."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.encode("ascii", "ignore").decode("ascii").lower()


def _canon(text, table):
    toks = [t for t in NONALNUM.sub(" ", text).split() if t]
    return [table.get(t, t) for t in toks]


def norm_name(raw):
    """Canonical business name: folded, junk-stripped, legal suffixes unified."""
    t = JUNK.sub("", fold(raw))
    return WS.sub(" ", " ".join(_canon(t, LEGAL))).strip()


def norm_addr(raw):
    """Canonical address: folded, abbreviations unified."""
    t = JUNK.sub("", fold(raw))
    return WS.sub(" ", " ".join(_canon(t, ADDR))).strip()


def name_tokens(norm):
    return set(norm.split())


def addr_tokens(norm):
    return set(norm.split())


def pincode(norm_address):
    """Postcode - US ZIP, India PIN, France code postal. "" when absent.

    Scans tokens from the END and skips token index 0: a leading 5-6 digit run is a
    house number (e.g. "17560 ellis rd tahlequah ok"), not a postcode. Treating it as
    one would create false blocking keys and merge unrelated businesses.
    """
    toks = norm_address.split()
    for i in range(len(toks) - 1, 0, -1):
        if PIN.match(toks[i]):
            return toks[i]
    return ""


def street_nums(norm_address):
    """All digit runs; catches house/building numbers regardless of position."""
    return set(NUM.findall(norm_address))


def norm_country(raw):
    return fold(raw).strip()
