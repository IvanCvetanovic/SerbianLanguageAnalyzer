from cyrtranslit import to_latin as _to_lat, to_cyrillic as _to_cyr


def lat_to_cyr(text: str) -> str:
    return _to_cyr(text, "sr")


def cyr_to_lat(text: str) -> str:
    return _to_lat(text, "sr")


def words_to_latin(words: list) -> list:
    return [_to_lat(w, "sr") for w in words]


def words_to_cyrillic(words: list) -> list:
    return [_to_cyr(w, "sr") for w in words]
