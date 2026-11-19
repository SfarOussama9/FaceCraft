"""Template caption engine: turns CelebA's 40 yes/no labels into varied English sentences.

Example: {Male: 1, Young: 1, Smiling: 1, Eyeglasses: 1, Black_Hair: 1}
      -> "a smiling young man with black hair, wearing glasses"
      or "a photo of a young guy with spectacles, a smile and black hair" ...

Variety matters: if every caption looked identical, the text encoder would only learn our template,
not language. So each call randomly picks a sentence pattern, synonyms and the order of details.

Rules that keep captions truthful:
  - never mention subjective/out-of-frame labels (Attractive, Blurry, Wearing_Necktie, Wearing_Necklace)
  - at most one hair color (if CelebA marks 2+, the label is unreliable -> mention none)
  - Bald removes hair color, bangs and hair texture
  - facial hair is only mentioned for men (CelebA's female facial-hair labels are mostly noise)
"""
import random

from data.celeba import ATTR_NAMES

EXCLUDED = ["Attractive", "Blurry", "Wearing_Necktie", "Wearing_Necklace"]
CAPTION_ATTRS = [a for a in ATTR_NAMES if a not in EXCLUDED]

HAIR_COLORS = {"Black_Hair": ["black"], "Blond_Hair": ["blond", "blonde"], "Brown_Hair": ["brown"],
               "Gray_Hair": ["gray", "grey"]}
FACIAL_HAIR = ["Goatee", "Mustache", "5_o_Clock_Shadow", "Sideburns"]

# Noun phrases used after "with ..." (one is picked at random)
WITH_PHRASES = {
    "Bangs": ["bangs", "a fringe"],
    "Receding_Hairline": ["a receding hairline"],
    "Goatee": ["a goatee"],
    "Mustache": ["a mustache", "a moustache"],
    "5_o_Clock_Shadow": ["stubble", "a five o'clock shadow"],
    "Sideburns": ["sideburns"],
    "Arched_Eyebrows": ["arched eyebrows"],
    "Bushy_Eyebrows": ["bushy eyebrows", "thick eyebrows"],
    "Big_Lips": ["big lips", "full lips"],
    "Big_Nose": ["a big nose", "a large nose"],
    "Pointy_Nose": ["a pointy nose", "a pointed nose"],
    "Narrow_Eyes": ["narrow eyes"],
    "Bags_Under_Eyes": ["bags under the eyes"],
    "High_Cheekbones": ["high cheekbones"],
    "Rosy_Cheeks": ["rosy cheeks"],
    "Pale_Skin": ["pale skin", "a pale complexion"],
    "Oval_Face": ["an oval face"],
    "Double_Chin": ["a double chin"],
    "Mouth_Slightly_Open": ["the mouth slightly open", "slightly parted lips"],
    "Heavy_Makeup": ["heavy makeup"],
    "Smiling": ["a smile", "a big smile"],
}
# Things a person wears: "wearing ..." (or "with ...")
WEAR_PHRASES = {
    "Eyeglasses": ["glasses", "eyeglasses", "spectacles"],
    "Wearing_Hat": ["a hat"],
    "Wearing_Earrings": ["earrings"],
    "Wearing_Lipstick": ["lipstick"],
}

PATTERNS = [
    "{subj} with {with_}",
    "a photo of {subj} with {with_}",
    "a portrait of {subj} with {with_}",
    "a close-up of {subj} with {with_}",
    "{subj} who has {with_}",
    "this is {subj} with {with_}",
    "{subj} that has {with_}",
    "a face of {subj} with {with_}",
    "{subj}. {pron} has {with_}",
    "a picture of {subj} having {with_}",
    "{subj} featuring {with_}",
    "an image of {subj} with {with_}",
    "{subj}, with {with_}",
    "a headshot of {subj} with {with_}",
    "{subj} showing {with_}",
]
NO_DETAIL_PATTERNS = ["a photo of {subj}", "a portrait of {subj}", "a close-up of {subj}",
                      "a picture of {subj}", "a headshot of {subj}"]


def _join(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _article(phrase: str) -> str:
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def _subject(attrs: dict, rng: random.Random, smiling_adj: bool) -> str:
    male = attrs["Male"] == 1
    noun = rng.choice(["man", "man", "guy", "gentleman"] if male else ["woman", "woman", "lady"])
    age = rng.choice(["young"]) if attrs["Young"] == 1 else rng.choice(["older", "middle-aged", "mature"])
    words = []
    if smiling_adj:
        words.append("smiling")
    if attrs["Chubby"] == 1:
        words.append("chubby")
    words.append(age)
    return _article(" ".join(words + [noun]))


def _hair_phrase(attrs: dict, rng: random.Random):
    """One phrase about the hair, e.g. "wavy blond hair", or None. Applies the hair rules."""
    if attrs["Bald"] == 1:
        return rng.choice(["a bald head", "no hair"])
    colors = [c for c in HAIR_COLORS if attrs[c] == 1]
    color = rng.choice(HAIR_COLORS[colors[0]]) if len(colors) == 1 else None
    texture = "wavy" if attrs["Wavy_Hair"] == 1 else ("straight" if attrs["Straight_Hair"] == 1 else None)
    words = [w for w in [texture, color] if w]
    return " ".join(words + ["hair"]) if words else None


def template_caption(attrs: dict, rng: random.Random) -> str:
    """Build one random caption that only mentions labels set to 1 (under the rules above)."""
    male = attrs["Male"] == 1
    smiling_adj = attrs["Smiling"] == 1 and rng.random() < 0.5   # "a smiling man" vs "... with a smile"
    subj = _subject(attrs, rng, smiling_adj)

    details = []
    hair = _hair_phrase(attrs, rng)
    if hair:
        details.append(hair)
    for name, options in WITH_PHRASES.items():
        if attrs[name] != 1:
            continue
        if name in FACIAL_HAIR and not male:
            continue
        if name in ("Bangs",) and attrs["Bald"] == 1:
            continue
        if name == "Smiling" and smiling_adj:
            continue
        details.append(rng.choice(options))
    # CelebA's No_Beard=0 mostly means "some facial hair" (CelebA-Dialog: about half are light stubble,
    # ~2% a full beard). So: never claim a FULL beard, and if stubble is already mentioned, say nothing more.
    if male and attrs["No_Beard"] == 0 and attrs["Goatee"] == 0 and attrs["5_o_Clock_Shadow"] == 0:
        details.append(rng.choice(["a beard", "facial hair"]))
    worn = [rng.choice(opts) for name, opts in WEAR_PHRASES.items() if attrs[name] == 1]

    rng.shuffle(details)
    rng.shuffle(worn)
    if worn and (not details or rng.random() < 0.6):
        wearing = "wearing " + _join(worn)       # "..., wearing glasses and a hat"
    else:
        details += worn
        wearing = None

    if not details:
        text = (f"{subj} {wearing}" if wearing else rng.choice(NO_DETAIL_PATTERNS).format(subj=subj))
        if wearing and rng.random() < 0.5:
            text = f"a photo of {text}"
        return text
    text = rng.choice(PATTERNS).format(subj=subj, with_=_join(details), pron="He" if male else "She")
    if wearing:
        text += ", " + wearing
    return text
