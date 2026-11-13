"""Baseline từ điển/quy tắc, không LLM, không mạng, không ngẫu nhiên.

Giữ giao diện của khung baseline.py: run(cfg, **kwargs). Khi tích hợp vào src/,
dùng chung tiện ích, cấu hình, hằng số và select_inputs từ llm.py.
Chỉ select_inputs của LLM được phép chọn tập đầu vào; không tự chọn mẫu khác.

Đọc reviews_clean.csv; ghi baseline_results.csv với đúng llm.RESULT_COLUMNS.
Hỗ trợ từ vựng es/en/pt/fr, bảy khía cạnh mặc định (kèm bí danh tiếng Anh).
Đây là đối chứng giải thích được; chưa có accuracy/F1 nếu chưa có nhãn tay.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .llm import (
    ASPECT_SEPARATOR, RESULT_COLUMNS, SENTIMENTS, UNKNOWN_LANGUAGE,
    get_logger, path, get_llm_config, select_inputs,
)

log = get_logger("baseline")
IN_REVIEWS = "reviews_clean.csv"
OUT_RESULTS = "baseline_results.csv"
SUPPORTED_LANGUAGES = ["es", "en", "pt", "fr"]
MIXED_RATIO_THRESHOLD = 0.4
DEFAULT_SENTIMENT = "positive"
MIN_STOPWORD_HITS = 2
NEGATION_WINDOW = 3
TOKEN_REGEX = r"[a-z]+"
SENTENCE_SPLIT_REGEX = r"[.!?\n]+"
# Ranh giới mệnh đề giúp phủ định không lan qua 'but/pero', dấu phẩy, chấm phẩy.
CLAUSE_SPLIT_REGEX = r"[,;:]+|\b(?:but|however|although|yet|pero|aunque|sin embargo|mas|porem|contudo|mais|cependant|pourtant)\b"

# Từ điển cố định, không được xây từ nhãn test. Có thể bổ sung từ tập huấn luyện
# sau khi nhóm khảo sát; không dùng nhãn tay đánh giá để chỉnh từ rồi báo điểm.
POSITIVE_WORDS: set[str] = set("""
good great excellent amazing wonderful lovely fantastic perfect beautiful nice
clean spotless comfortable cozy spacious quiet safe helpful friendly welcoming
convenient central affordable cheap modern equipped recommend recommended love
loved pleasant responsive kind superb outstanding
bueno buena buenos buenas excelente excelentes genial maravilloso maravillosa
fantastico fantastica perfecto perfecta hermoso hermosa bonito bonita agradable
limpio limpia limpios limpias impecable comodo comoda comodas comodos acogedor
acogedora amplio amplia amplios amplias tranquilo tranquila silencioso silenciosa
seguro segura amable amables atento atenta atentos atentas recomendable recomendado
recomendada recomiendo encantador encantadora cordial equipado equipada economico
economica centrico centrica
bom boa bons boas otimo otima otimos otimas maravilhoso maravilhosa maravilhoso
perfeito perfeita lindo linda bonito bonita limpo limpa limpos limpas confortavel
confortaveis aconchegante espacoso espacosa tranquilo tranquila silencioso silenciosa
seguro segura simpatico simpatica atencioso atenciosa acolhedor acolhedora recomendo
recomendado recomendada equipado equipada barato barata excelente agradavel
bon bonne bons bonnes excellent excellente excellents excellentes magnifique
merveilleux merveilleuse parfait parfaite beau belle propre propres impeccable
confortable confortables agreable agreables chaleureux chaleureuse spacieux spacieuse
calme tranquille securise securisee accueillant accueillante aimable sympathique pratique
recommande recommandees recommande sympa formidable
""".split())
NEGATIVE_WORDS: set[str] = set("""
bad awful terrible horrible poor dirty filthy noisy loud unsafe dangerous rude
uncomfortable broken expensive overpriced small tiny disappointing disappointed
disgusting smell smelly problem problems difficult unhelpful unpleasant cold
bugs cockroaches theft scam unreliable
malo mala malos malas pesimo pesima horrible sucio sucia sucios sucias ruidoso
ruidosa ruido ruidos inseguro insegura peligroso peligrosa incomodo incomoda roto
rota rotos rotas caro cara caros caras pequeno pequena decepcionante decepcion
olor olores problema problemas dificil desagradable frio fria cucarachas estafa
barulho barulhento barulhenta ruim ruins pessimo pessima sujo suja sujos sujas
inseguro insegura perigoso perigosa desconfortavel quebrado quebrada caro cara
pequeno pequena decepcionante cheiro problema problemas dificil desagradavel frio
fria baratas golpe
mau mauvais mauvaise sales sale bruyant bruyante bruit dangereux dangereuse
inconfortable casse cassee cher chere minuscule decevant decevante odeur probleme
problemes difficile desagreable froid froide cafards arnaque impoli impolie
""".split())
# 'baratas' (PT: gián) không cùng 'barata' (PT/ES: rẻ). Không stemming tuỳ tiện.
# Tránh từ vừa mang cực tính dương vừa âm giữa các ngôn ngữ.
NEGATIVE_WORDS.difference_update(POSITIVE_WORDS)
NEGATORS: set[str] = {"not", "no", "never", "neither", "without", "nunca", "jamas",
                       "sin", "nao", "nem", "sem", "pas", "jamais", "sans"}
CONTRAST_TOKENS = {"but", "however", "although", "yet", "pero", "aunque", "mas", "porem",
                   "contudo", "mais", "cependant", "pourtant"}

# Tên cấu hình được giữ nguyên trong đầu ra; từ khoá đều đã bỏ dấu.
ASPECT_KEYWORDS: dict[str, set[str]] = {
    "vị trí": set("location located central convenient neighbourhood neighborhood area transport subway metro ubicacion ubicado ubicada zona barrio cerca centro localizacao localizado localizada perto local endroit emplacement quartier proche situe situee".split()),
    "sạch sẽ": set("clean cleanliness spotless dirty filthy smell smelly bugs cockroaches limpio limpia limpios limpias limpieza sucio sucia sucios sucias olor olores cucarachas limpo limpa limpos limpas limpeza sujo suja sujos sujas cheiro baratas propre propres proprete sale sales odeur cafards".split()),
    "chủ nhà": set("host hosts owner owners communication helpful friendly rude anfitrion anfitriona dueno duena propietario propietaria comunicacion anfitriao anfitria proprietario proprietaria comunicacao hote hotesse proprietaire accueil accueillant accueillante communication".split()),
    "giá trị": set("value price cost expensive overpriced affordable cheap precio valor costo caro cara caros caras economico economica barato barata preco custo rapport prix cher chere".split()),
    "tiện nghi": set("amenities amenity wifi internet kitchen bathroom bed beds shower heating air conditioning equipment equipped comfortable spacious broken cocina bano cama camas ducha calefaccion aire acondicionado equipado equipada comodo comoda amplio amplia roto rota cozinha banheiro chuveiro aquecimento equipado equipada confortavel espacoso espacosa quebrado quebrada equipement equipements cuisine lit lits douche chauffage confortable spacieux spacieuse casse cassee".split()),
    "độ ồn": set("noise noisy loud quiet silent ruido ruidos ruidoso ruidosa silencioso silenciosa tranquilo tranquila barulho barulhento barulhenta silencio bruit bruyant bruyante calme tranquille".split()),
    "an toàn": set("safe safety unsafe dangerous secure security theft scam seguro segura seguridad inseguro insegura peligroso peligrosa robo estafa seguranca perigoso perigosa golpe securise securisee securite dangereux dangereuse vol arnaque".split()),
}
ASPECT_ALIASES = {
    "vi tri": "vị trí", "location": "vị trí",
    "sach se": "sạch sẽ", "cleanliness": "sạch sẽ",
    "chu nha": "chủ nhà", "host": "chủ nhà", "communication": "chủ nhà",
    "gia tri": "giá trị", "value": "giá trị", "value for money": "giá trị", "price": "giá trị",
    "tien nghi": "tiện nghi", "amenities": "tiện nghi", "comfort": "tiện nghi",
    "do on": "độ ồn", "noise": "độ ồn",
    "an toan": "an toàn", "safety": "an toàn", "security": "an toàn",
}
STOPWORDS: dict[str, set[str]] = {
    "es": set("el la los las un una unos unas de del al y o que en con para por muy fue fueron era estaba estaban es esta estuvo estuvimos nos nuestro nuestra nosotros pero como todo todos tambien lo se su sus hay hemos volveria".split()),
    "en": set("the a an and or of to in on at with for from was were is are this that these those very we our us you your it its had have has would will can could there here but not as by".split()),
    "pt": set("o a os as um uma uns umas de do da dos das e ou que em no na nos nas com para por muito foi foram era estava estavam estava esteve ficamos nosso nossa fomos tambem tudo mas pelo pela voce voces ele ela".split()),
    "fr": set("le la les un une des de du au aux et ou que en dans avec pour par tres est sont etait etaient nous notre nos vous votre ils elles ce cette ces c et il elle on mais pas aussi tout bien avons avons ete".split()),
}


def norm_text(text: str) -> str:
    """Viết thường, bỏ dấu NFD; None/NaN/pd.NA trở thành chuỗi rỗng."""
    if text is None or text is pd.NA:
        return ""
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Tách từ Latin; mở rộng phủ định tiếng Anh trước khi bỏ dấu nháy."""
    normalized = norm_text(text).replace("’", "'")
    normalized = re.sub(r"\bcannot\b", "can not", normalized)
    normalized = re.sub(r"\b([a-z]+)n't\b", r"\1 not", normalized)
    return re.findall(TOKEN_REGEX, normalized)


def split_sentences(text: str) -> list[str]:
    if text is None or text is pd.NA or (not isinstance(text, str) and pd.isna(text)):
        return []
    return [part.strip() for part in re.split(SENTENCE_SPLIT_REGEX, str(text)) if part.strip()]


def _clauses(text: str) -> list[str]:
    # Chuẩn hoá trước khi tách để 'porém' và 'porem' có cùng hành vi.
    return [part.strip() for sentence in split_sentences(text)
            for part in re.split(CLAUSE_SPLIT_REGEX, norm_text(sentence)) if part.strip()]


def apply_negation(tokens: list[str]) -> list[tuple[str, int]]:
    """Đảo cực tính trong tối đa 3 từ kế tiếp, đặt lại ở từ nối tương phản.

    Phủ định liên tiếp gia hạn cửa sổ (hợp với 'no ... nunca' trong ES), không
    cố giải quyết mọi cấu trúc phủ định kép. Bỏ qua 'not only/no solo/nao so'.
    """
    result, remaining = [], 0
    language = detect_language(tokens)
    for i, token in enumerate(tokens):
        if token in CONTRAST_TOKENS:
            remaining = 0
        next_word = tokens[i + 1] if i + 1 < len(tokens) else ""
        only = (token, next_word) in {("not", "only"), ("no", "solo"), ("no", "solamente"), ("nao", "so")}
        # PT 'no apartamento' = 'trong căn hộ', không phải phủ định ES 'no'.
        pt_locative = token == "no" and (language == "pt" or next_word in {
            "apartamento", "apartamentos", "quarto", "quartos", "banheiro", "banheiros",
            "predio", "bairro", "local", "hotel", "centro"})
        if token in NEGATORS and not only and not pt_locative:
            result.append((token, 1))
            remaining = NEGATION_WINDOW
        else:
            result.append((token, -1 if remaining else 1))
            remaining = max(0, remaining - 1)
    return result


def _sentiment_evidence(tokens: list[str]) -> list[tuple[int, int]]:
    evidence = []
    language = detect_language(tokens)
    for i, (token, factor) in enumerate(apply_negation(tokens)):
        # FR sale = bẩn; EN sale = giảm giá. Chỉ dùng nghĩa FR khi đủ dấu hiệu.
        if token in {"sale", "sales"} and language != "fr":
            continue
        base = int(token in POSITIVE_WORDS) - int(token in NEGATIVE_WORDS)
        if base:
            evidence.append((i, base * factor))
    return evidence


def count_sentiment_words(tokens: list[str]) -> tuple[int, int]:
    evidence = _sentiment_evidence(tokens)
    return sum(polarity > 0 for _, polarity in evidence), sum(polarity < 0 for _, polarity in evidence)


def decide_sentiment(n_pos: int, n_neg: int) -> str:
    """Hai phía đủ cân bằng -> mixed; còn lại dùng phía trội, mặc định positive."""
    if n_pos < 0 or n_neg < 0:
        raise ValueError("Số từ cảm xúc phải không âm")
    if n_pos == n_neg == 0:
        return DEFAULT_SENTIMENT
    if n_pos and n_neg and min(n_pos, n_neg) / max(n_pos, n_neg) >= MIXED_RATIO_THRESHOLD:
        return "mixed"
    return "negative" if n_neg > n_pos else "positive"


def _keywords(aspect: str) -> set[str]:
    if aspect in ASPECT_KEYWORDS:
        return ASPECT_KEYWORDS[aspect]
    # NFD không tự chuyển đ -> d; xử lý riêng cho bí danh tiếng Việt.
    canonical = ASPECT_ALIASES.get(norm_text(aspect).replace("đ", "d").strip())
    if canonical is None:
        raise ValueError(f"Chưa có từ điển cho khía cạnh {aspect!r}; bổ sung ASPECT_KEYWORDS")
    return ASPECT_KEYWORDS[canonical]


def _configured_aspects(cfg: dict) -> list[str]:
    aspects = get_llm_config(cfg)["aspects"]
    if not isinstance(aspects, list) or not aspects or any(not isinstance(a, str) or not a.strip() for a in aspects):
        raise ValueError("cfg['llm']['aspects'] phải là danh sách tên khía cạnh không rỗng")
    if not isinstance(ASPECT_SEPARATOR, str) or not ASPECT_SEPARATOR:
        raise ValueError("llm.ASPECT_SEPARATOR phải là chuỗi không rỗng")
    if len(aspects) != len(set(aspects)) or any(ASPECT_SEPARATOR in a for a in aspects):
        raise ValueError("Tên khía cạnh bị trùng hoặc chứa ASPECT_SEPARATOR")
    for aspect in aspects:
        _keywords(aspect)
    return aspects


def extract_aspects(text: str, aspects: list[str]) -> tuple[set[str], set[str]]:
    """Gán cực tính của từ cảm xúc gần từ khoá nhất trong cùng mệnh đề.

    Cùng khoảng cách có cả khen/chê thì giữ cả hai; mệnh đề không có bằng chứng
    cảm xúc không sinh khía cạnh. Không dùng nhãn positive mặc định để suy ra khen.
    Một khía cạnh có thể thuộc cả hai tập nếu review vừa khen vừa chê nó.
    """
    positive, negative = set(), set()
    dictionaries = {aspect: _keywords(aspect) for aspect in aspects}
    for clause in _clauses(text):
        tokens = tokenize(clause)
        language = detect_language(tokens)
        evidence = _sentiment_evidence(tokens)
        if not evidence:
            continue
        for aspect, keywords in dictionaries.items():
            positions = [i for i, token in enumerate(tokens) if token in keywords
                         and not (token in {"sale", "sales"} and language != "fr")]
            for position in positions:
                distance = min(abs(i - position) for i, _ in evidence)
                polarities = {polarity for i, polarity in evidence if abs(i - position) == distance}
                if 1 in polarities:
                    positive.add(aspect)
                if -1 in polarities:
                    negative.add(aspect)
    return positive, negative


def detect_language(tokens: list[str]) -> str:
    """Ít nhất 2 từ dừng PHÂN BIỆT; hoà điểm trả und thay vì đoán tuỳ tiện."""
    vocabulary = set(tokens)
    scores = {language: len(vocabulary & STOPWORDS[language]) for language in SUPPORTED_LANGUAGES}
    best = max(scores.values(), default=0)
    winners = [language for language, score in scores.items() if score == best]
    return winners[0] if best >= MIN_STOPWORD_HITS and len(winners) == 1 else UNKNOWN_LANGUAGE


def classify_review(text: str, cfg: dict) -> dict:
    aspects = _configured_aspects(cfg)
    # Chấm riêng mỗi mệnh đề để 'not clean. Great host.' không đảo 'great'.
    counts = [count_sentiment_words(tokenize(clause)) for clause in _clauses(text)]
    n_pos, n_neg = sum(p for p, _ in counts), sum(n for _, n in counts)
    positive, negative = extract_aspects(text, aspects)
    return {"sentiment": decide_sentiment(n_pos, n_neg),
            "language": detect_language(tokenize(text)),
            "aspects_positive": ASPECT_SEPARATOR.join(sorted(positive)),
            "aspects_negative": ASPECT_SEPARATOR.join(sorted(negative))}


def _exact_ids(series: pd.Series, column: str) -> pd.Series:
    """Chỉ nhận số nguyên chính xác; không phục hồi ID đã từng bị làm tròn qua float."""
    values = []
    for index, value in series.items():
        if isinstance(value, bool) or pd.isna(value):
            raise ValueError(f"{column}: ID thiếu/sai ở index {index!r}")
        # isinstance(np.int64, int) là False; chuyển qua str vẫn giữ chính xác.
        if isinstance(value, float) or not re.fullmatch(r"[0-9]+", str(value)):
            raise ValueError(f"{column}: cần ID nguyên, không dùng float; index {index!r}: {value!r}")
        number = int(value)
        if not 0 < number <= 2**63 - 1:
            raise ValueError(f"{column}: ID ngoài miền int64 dương tại index {index!r}")
        values.append(number)
    return pd.Series(values, index=series.index, dtype="int64", name=column)


def _boolean_column(series: pd.Series, column: str) -> pd.Series:
    text = series.astype("string").fillna("").str.strip().str.lower()
    true = {"true", "t", "1", "1.0", "x", "yes", "y"}
    false = {"false", "f", "0", "0.0", "", "no", "n"}
    bad = ~text.isin(true | false)
    if bad.any():
        raise ValueError(f"{column}: giá trị bool không hợp lệ ở index {text.index[bad][0]!r}")
    return text.isin(true)


def load_inputs(cfg: dict) -> pd.DataFrame:
    """Đọc an toàn ID/bool, rồi dùng DUY NHẤT selector chung của llm.py."""
    source = path(cfg, "processed") / IN_REVIEWS
    reviews = pd.read_csv(source, dtype="string", keep_default_na=False)
    required = {"id", "comments_clean", "use_for_llm", "llm_sample"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"{source.name} thiếu cột: {', '.join(sorted(missing))}")
    reviews["id"] = _exact_ids(reviews["id"], "id")
    for col in ("use_for_llm", "llm_sample", "hand_label"):
        if col in reviews:
            reviews[col] = _boolean_column(reviews[col], col)
    # Không tự lọc use_for_llm/llm_sample lần nữa hoặc lấy head() ở đây.
    inputs = select_inputs(cfg, reviews)
    if not isinstance(inputs, pd.DataFrame):
        raise ValueError("llm.select_inputs phải trả DataFrame có id, comments_clean")
    return inputs


def build_results_frame(cfg: dict, inputs: pd.DataFrame) -> pd.DataFrame:
    _configured_aspects(cfg)
    if not {"id", "comments_clean"}.issubset(inputs.columns):
        raise ValueError("inputs cần cột id và comments_clean từ llm.select_inputs")
    ids = _exact_ids(inputs["id"], "id")
    if ids.duplicated().any():
        duplicate = int(ids.loc[ids.duplicated()].iloc[0])
        raise ValueError(f"select_inputs trả review ID trùng: {duplicate}")
    rows = [{"review_id": int(review_id), **classify_review(text, cfg)}
            for review_id, text in zip(ids, inputs["comments_clean"])]
    # Chỉ định dtype ID cả với bảng rỗng; không để pandas suy ra float.
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results["review_id"] = pd.Series(ids.to_numpy(), dtype="int64")
    return results


def validate_results(results: pd.DataFrame, cfg: dict) -> None:
    aspects = set(_configured_aspects(cfg))
    if list(results.columns) != list(RESULT_COLUMNS):
        raise ValueError(f"Sai cột/thứ tự: cần {list(RESULT_COLUMNS)!r}, nhận {list(results.columns)!r}")
    if not pd.api.types.is_integer_dtype(results["review_id"].dtype):
        raise ValueError("review_id phải có dtype số nguyên, không dùng float/chuỗi")
    _exact_ids(results["review_id"], "review_id")
    if results["review_id"].duplicated().any():
        index = results.index[results["review_id"].duplicated()][0]
        raise ValueError(f"review_id trùng tại index {index!r}")
    for index, row in results.iterrows():
        if not isinstance(row["sentiment"], str) or row["sentiment"] not in SENTIMENTS:
            raise ValueError(f"sentiment không hợp lệ ở index {index!r}: {row['sentiment']!r}")
        # Chỉ xuất mã đã hỗ trợ hoặc und; không nhận chuỗi hai chữ tuỳ ý như 'xx'.
        if not isinstance(row["language"], str) or row["language"] not in {*SUPPORTED_LANGUAGES, UNKNOWN_LANGUAGE}:
            raise ValueError(f"language không hợp lệ ở index {index!r}: {row['language']!r}")
        for col in ("aspects_positive", "aspects_negative"):
            value = row[col]
            if not isinstance(value, str):
                raise ValueError(f"{col} phải là chuỗi, index {index!r}")
            labels = value.split(ASPECT_SEPARATOR) if value else []
            if any(label not in aspects for label in labels) or labels != sorted(set(labels)):
                raise ValueError(f"{col} chứa khía cạnh lạ/trùng/chưa sắp xếp ở index {index!r}: {value!r}")


def save_results(cfg: dict, results: pd.DataFrame) -> None:
    validate_results(results, cfg)
    destination = path(cfg, "processed") / OUT_RESULTS
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")


def run(cfg: dict, **_) -> None:
    inputs = load_inputs(cfg)
    results = build_results_frame(cfg, inputs)
    validate_results(results, cfg)
    save_results(cfg, results)
    log.info("Baseline: %d reviews; sentiment=%s; language=%s", len(results),
             results["sentiment"].value_counts().sort_index().to_dict(),
             results["language"].value_counts().sort_index().to_dict())
