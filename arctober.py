"""Arc filter: whether a message says arc-tober, October, or arc.

bot.py decides whose messages are checked and deletes them. This file only
answers "does this text say it", however it is disguised. Every check works on
plain lowercase letters, so the first job is turning whatever was typed into
those letters (fold). After that there are four checks, any one of which is
enough:

1. The whole message with everything but letters removed contains "october"
   or "arctober". Spacing, punctuation, spoilers and emoji between letters stop
   mattering: "o.c.t-o b e r" and "||arc||tober" are both caught here. This is
   the only check that ignores spaces, so it is also the strictest: no loose
   matching, and k does not count as c.
2. A word contains "tober": spooktober, rocktober, arctober.
3. A word is close to october or arc-tober: one letter extra, missing, wrong,
   or two swapped, after sound-alikes are merged (k, q and ck read as c, h is
   ignored, any vowel counts as any other). "arrcckt0bher", "octobr",
   "oktober", "octubre", "arktuber" all land here.
4. A word is "arc" or "oct" on its own.

Text typed backwards is not checked. Reversed, "reboot computer" contains
october, and "rebooting" contains "tober".

Checks 2-4 are per word, not across the whole message, because the loose
matching would find words where there are none: "elect a barber" run together
reads as october once vowels are merged. Check 1 stays exact for the same
reason. A word here is whatever sits between spaces, so "arc-tober" is one
word; letters typed one at a time ("o c t o b e r") are joined back into one.
"""

import re
import unicodedata

# Several characters standing in for one letter. Replaced first, while the
# characters are still as typed.
_SEQUENCES = {
    "()": "o", "[]": "o", "{}": "o", "<>": "o",
    "|3": "b", "|2": "r", "/\\": "a", "|<": "k", "|(": "k",
    "ß": "b",
}

# Lookalikes that Unicode normalization leaves alone: leetspeak, Cyrillic and
# Greek letters that render the same as Latin ones, small caps, upside-down
# letters, and the letter emoji Discord renders (regional indicators, and the
# boxed and circled letter emoji).
_LOOKALIKES = str.maketrans({
    "4": "a", "@": "a", "а": "a", "α": "a", "ᴀ": "a", "ɑ": "a", "ɐ": "a",
    "8": "b", "в": "b", "β": "b", "ʙ": "b", "ь": "b", "б": "b",
    "¢": "c", "©": "c", "с": "c", "ᴄ": "c", "(": "c", "<": "c", "ɔ": "c",
    "3": "e", "€": "e", "е": "e", "ε": "e", "ᴇ": "e", "ǝ": "e", "ə": "e",
    "к": "k", "κ": "k", "ᴋ": "k", "ʞ": "k",
    "0": "o", "о": "o", "ο": "o", "ᴏ": "o", "ø": "o", "°": "o", "º": "o",
    "⭕": "o", "σ": "o",
    "®": "r", "г": "r", "я": "r", "ʀ": "r", "ɹ": "r", "ɾ": "r",
    "7": "t", "+": "t", "т": "t", "τ": "t", "ᴛ": "t", "ʇ": "t", "†": "t",
    **{chr(0x1F1E6 + i): chr(ord("a") + i) for i in range(26)},
    **{chr(0x1F150 + i): chr(ord("a") + i) for i in range(26)},
    **{chr(0x1F170 + i): chr(ord("a") + i) for i in range(26)},
})

# What check 3 compares words against, already passed through _sounds.
_NEAR_TARGETS = ("octobor", "orctobor")


def _fold(text: str) -> str:
    """Lowercase plain letters, whatever alphabet or styling they arrived in.

    NFKD folds fullwidth, bold, script, and circled letters to plain ones and
    splits accents off; the tables above cover what it misses. Accents and
    invisible characters such as zero-width spaces are dropped outright rather
    than becoming separators, so they cannot break a word in two.
    """
    for seq, letter in _SEQUENCES.items():
        text = text.replace(seq, letter)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) not in ("Mn", "Cf"))
    return text.casefold().translate(_LOOKALIKES)


def _squash(letters: str) -> str:
    """Collapse repeated letters: "aaarrrc" -> "arc"."""
    return re.sub(r"(.)\1+", r"\1", letters)


def _c_sounds(letters: str) -> str:
    """Read k and q as c, then collapse repeats, so "ck" is one c too."""
    return _squash(re.sub(r"[kq]", "c", letters))


def _sounds(letters: str) -> str:
    """The loose reading for check 3: c-sounds merged, h ignored, every run of
    vowels one "o". "arrcckt0bher" and "arktuber" both become "orctobor"."""
    letters = _c_sounds(letters.replace("h", ""))
    return re.sub(r"[aeiouy]+", "o", letters)


def _distance(target: str, word: str) -> int:
    """Fewest edits turning target into some stretch of word.

    An edit is one letter added, removed, or changed, or two neighbours
    swapped. Matching a stretch rather than the whole word means "doctober" is
    0 from "octobor", not 1.
    """
    m, n = len(target), len(word)
    # d[i][j]: target's first i letters against a stretch ending at word[j].
    # Row 0 is all zeros because the stretch may start anywhere.
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        d[i][0] = i
        for j in range(1, n + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (target[i - 1] != word[j - 1]),
            )
            if (i > 1 and j > 1 and target[i - 1] == word[j - 2]
                    and target[i - 2] == word[j - 1]):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return min(d[m])


def _join_spelled(words: list[str]) -> list[str]:
    """Join runs of one-letter words back together: "a r c" -> "arc"."""
    joined, spelled = [], ""
    for word in words:
        if len(word) == 1:
            spelled += word
            continue
        if spelled:
            joined.append(spelled)
            spelled = ""
        joined.append(word)
    if spelled:
        joined.append(spelled)
    return joined


def mentions_arc(text: str) -> bool:
    """Whether text says arc-tober, October, or arc, however disguised."""
    text = _fold(text)

    # 1. Across the whole message. No k for c here: "ok to be real" run
    # together says oktober. The word checks below still read k as c.
    if re.search(r"a?rctober|october", _squash(re.sub(r"[^a-z]", "", text))):
        return True

    # Words as split by spaces, punctuation inside them dropped.
    words = _join_spelled(
        [w for w in (re.sub(r"[^a-z]", "", t) for t in text.split()) if w]
    )

    # 2. "tober" inside a word. Never across two words: "go to Berlin".
    if any("tober" in _c_sounds(w.replace("h", "")) for w in words):
        return True

    # 3. Close to october or arc-tober. Also tried on each pair of neighbouring
    # words, for a gap typed in the middle: "octo bre". The c, t, b must all be
    # there with only vowels between: with vowels merged, "youtuber" and
    # "contract over" are otherwise one edit from october. And the c must open
    # the word or follow a, o or r, as it does in october, arc-tober and the
    # "actober" typo: "topic tuber" run together is otherwise october exactly.
    pairs = [a + b for a, b in zip(words, words[1:])]
    for w in words + pairs:
        loose = _sounds(w)
        if "ctb" not in loose.replace("o", ""):
            continue
        if not re.search(r"^c|[aor]c", _c_sounds(w.replace("h", ""))):
            continue
        if any(_distance(t, loose) <= 1 for t in _NEAR_TARGETS):
            return True

    # 4. Arc or oct as a word of its own. Split at every non-letter here, so
    # "arc's" and "oct." count. Only c-sounds are merged: across everything,
    # arc would be found in "search" and "march", and with h ignored "arch"
    # would become "arc". Arc keeps its c because "ark" is a word in its own
    # right.
    for word in _join_spelled(re.findall(r"[a-z]+", text)):
        if _squash(word) in ("arc", "arcs"):
            return True
        if _c_sounds(word) in ("oct", "octs"):
            return True
    return False
