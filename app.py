# app.py
# Exploratory Hierarchical Search (Flask + OpenAI + D3)
# v0.3.2 (experiment logging by subject_id)
# - Parent refinement (generate-many -> server filter -> backoff to ensure 2–3)
# - Recenter with history kept
# - JA/EN translate endpoint
# - Known meta with robust alias resolution (CNN/GAN/RNN…略語が確実に略語表示)

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from openai import OpenAI, APIStatusError, APITimeoutError
from dotenv import load_dotenv

import os, json, time, re, unicodedata, uuid, pathlib, random
from datetime import timedelta, datetime
from typing import List, Dict, Any


# === Load env & Flask ===
load_dotenv()
BASE_DIR = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
META_PATH = STATIC_DIR / "meta" / "known_meta.json"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(BASE_DIR / "templates"),
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "tree-demo")
app.permanent_session_lifetime = timedelta(days=7)

# === OpenAI client ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === LLM defaults ===
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EXPAND_TEMP = float(os.getenv("EXPAND_TEMP", "0.2"))
EXPAND_MAXTOK = int(os.getenv("EXPAND_MAXTOK", "350"))
EXPAND_TIMEOUT = int(os.getenv("EXPAND_TIMEOUT_S", "12"))
DESCR_TEMP = float(os.getenv("DESCR_TEMP", "0.3"))
DESCR_MAXTOK = int(os.getenv("DESCR_MAXTOK", "220"))
DESCR_TIMEOUT = int(os.getenv("DESCR_TIMEOUT_S", "8"))
RETRIES = int(os.getenv("LLM_RETRIES", "3"))

# === Parent generation policy ===
PARENT_GEN_COUNT = int(os.getenv("PARENT_GEN_COUNT", "6"))
PARENT_TARGET_MIN = int(os.getenv("PARENT_TARGET_MIN", "2"))
PARENT_TARGET_MAX = int(os.getenv("PARENT_TARGET_MAX", "3"))

# === Child generation policy ===
CHILD_TARGET_MIN = int(os.getenv("CHILD_TARGET_MIN", "3"))
CHILD_TARGET_MAX = int(os.getenv("CHILD_TARGET_MAX", "5"))
CHILD_RETRY_MAX = int(os.getenv("CHILD_RETRY_MAX", "1"))

# === Logging (NDJSON, subject_idごとのファイル) ===
LOG_DIR = pathlib.Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs" / "events")))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _ymd(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y%m%d")


def _safe_id(v: Any) -> str:
    """ファイル名に使うIDを安全な形式にする"""
    s = str(v) if v is not None else "unknown"
    return re.sub(r"[^0-9A-Za-z_\-]", "_", s)


def write_event(evt: Dict[str, Any]):
    """
    1行1イベントのNDJSONログ。
    ファイルは日付ディレクトリ + 被験者IDごとに分割:
        logs/events/YYYYMMDD/<subject_id>.ndjson
    被験者IDが無い場合は、sid ベースで落とす。
    """
    ts = evt.get("ts", time.time())

    subject_id = evt.get("subject_id") or session.get("subject_id")
    sid = evt.get("sid") or session.get("sid")

    if not sid:
        sid = session["sid"] = uuid.uuid4().hex[:12]

    if not subject_id:
        subject_id = sid

    evt.setdefault("subject_id", subject_id)
    evt.setdefault("sid", sid)

    day_dir = LOG_DIR / _ymd(ts)
    day_dir.mkdir(parents=True, exist_ok=True)

    safe_subject = _safe_id(subject_id)
    fpath = day_dir / f"{safe_subject}.ndjson"

    with open(fpath, "a", encoding="utf-8") as w:
        w.write(json.dumps(evt, ensure_ascii=False) + "\n")


# === Canonicalization & duplicates ===
_SYNO = {
    "convolutional neural network": "cnn",
    "neural network": "nn",
    "generative adversarial network": "gan",
    "recurrent neural network": "rnn",
}
_HYPHEN_CHARS = r"[\-‐-‒–—―]"


def canon(s: str) -> str:
    """表記ゆれ吸収用の強めの正規化（日本語は残す）"""
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = s.replace("・", " ")
    s = re.sub(_HYPHEN_CHARS, "-", s)
    s = re.sub(r"[\s_/]+", " ", s)
    s = re.sub(r"[^\w\s\-]", "", s)
    s = _SYNO.get(s, s)
    return s


def is_dup(norm_t: str, prevs: set[str]) -> bool:
    return norm_t in prevs


def ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [s.strip() for s in x.replace("、", ",").split(",") if s.strip()]
    return []


# === Known meta: 単一ソース + エイリアス解決 ===
_BASE_KNOWN_META: Dict[str, Dict[str, str]] = {
    "cnn": {
        "short": "CNN",
        "full": "Convolutional Neural Network（畳み込みニューラルネットワーク）",
        "desc": "画像などの空間構造を畳み込み演算で特徴抽出するニューラルネットワーク。",
    },
    "gan": {
        "short": "GAN",
        "full": "Generative Adversarial Network（敵対的生成ネットワーク）",
        "desc": "生成器と識別器を競わせて高品質なデータを生成するモデル。",
    },
    "rnn": {
        "short": "RNN",
        "full": "Recurrent Neural Network（再帰型ニューラルネットワーク）",
        "desc": "系列データを扱うための循環結合を持つNN。",
    },
    "lstm": {
        "short": "LSTM",
        "full": "Long Short-Term Memory",
        "desc": "長期依存を扱うためゲート構造を持つRNN拡張。",
    },
    "vae": {
        "short": "VAE",
        "full": "Variational Autoencoder",
        "desc": "確率的潜在変数で生成を行うオートエンコーダ。",
    },
    "bert": {
        "short": "BERT",
        "full": "Bidirectional Encoder Representations from Transformers",
        "desc": "双方向文脈を学習するTransformerベースの言語モデル。",
    },
    "gpt": {
        "short": "GPT",
        "full": "Generative Pre-trained Transformer",
        "desc": "事前学習したTransformerで生成を行うLLM。",
    },
    "svm": {
        "short": "SVM",
        "full": "Support Vector Machine",
        "desc": "最大マージンで分類境界を学習する手法。",
    },
    "pca": {
        "short": "PCA",
        "full": "Principal Component Analysis",
        "desc": "主成分に射影して次元削減する手法。",
    },
}

KNOWN_ALIASES: Dict[str, str] = {
    "convolutional neural network": "cnn",
    "畳み込みニューラルネットワーク": "cnn",
    "generative adversarial network": "gan",
    "敵対的生成ネットワーク": "gan",
    "recurrent neural network": "rnn",
    "再帰型ニューラルネットワーク": "rnn",
}

try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        external = json.load(f)
        if isinstance(external, dict):
            for k, v in external.items():
                ck = canon(k)
                if isinstance(v, dict) and "short" in v:
                    _BASE_KNOWN_META[ck] = v
except FileNotFoundError:
    pass


def get_known_meta(term: str):
    ck = canon(term)
    if ck in _BASE_KNOWN_META:
        return _BASE_KNOWN_META[ck]
    if ck in KNOWN_ALIASES and KNOWN_ALIASES[ck] in _BASE_KNOWN_META:
        return _BASE_KNOWN_META[KNOWN_ALIASES[ck]]
    return None


def export_known_meta_with_aliases() -> Dict[str, Dict[str, str]]:
    """フロントに返す /meta 用。略語キー + エイリアスキーを全部展開"""
    out = dict(_BASE_KNOWN_META)
    for alias_ck, master in KNOWN_ALIASES.items():
        if master in _BASE_KNOWN_META:
            out[alias_ck] = _BASE_KNOWN_META[master]
    return out


# === OpenAI call with JSON enforcement + backoff ===
def call_chat_json(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    retries: int = 3,
):
    delay = 1.0
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
                response_format={"type": "json_object"},
            )
        except (APIStatusError, APITimeoutError, Exception):
            if i == retries - 1:
                raise
            time.sleep(delay + random.random() * 0.2)
            delay *= 2


# === Parent refinement helpers ===
GENERIC_PARENTS = {
    "科学",
    "技術",
    "工学",
    "情報科学",
    "コンピュータサイエンス",
    "データサイエンス",
    "機械学習",
    "人工知能",
    "アルゴリズム",
    "最適化",
    "統計学",
    "数学",
    "計算機科学",
}


def _char_ngrams(s: str, n: int = 2) -> set[str]:
    s = re.sub(r"\s+", "", s or "")
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _similarity_char_ngram(a: str, b: str, n: int = 2) -> float:
    A, B = _char_ngrams(a, n), _char_ngrams(b, n)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union


def is_generic_parent(term: str) -> bool:
    cset = {canon(x) for x in GENERIC_PARENTS}
    return canon(term) in cset or term in GENERIC_PARENTS


def refine_parent_candidates(
    query: str,
    parents: list[str],
    *,
    jacc_min: float = 0.15,
    jacc_max: float = 0.85,
    limit: int | None = None,
) -> list[str]:
    """
    - 過度に汎用な語を除外
    - 文字バイグラムのJaccard類似で「近すぎ/遠すぎ」を除外
    - canon 重複を排除
    """
    q = re.sub(r"\s+", "", query or "")
    out: list[str] = []
    seen: set[str] = set()
    for p in ensure_list(parents):
        if not p:
            continue
        if is_generic_parent(p):
            continue
        sim = _similarity_char_ngram(q, p, n=2)
        if sim < jacc_min or sim > jacc_max:
            continue
        ck = canon(p)
        if ck and ck not in seen:
            seen.add(ck)
            out.append(p)
    if limit is not None:
        return out[:limit]
    return out


# === LLM prompts ===
def build_expand_messages(query: str, history: List[str], mode: str) -> List[Dict[str, str]]:
    mode_jp = "下位トピック" if mode == "child" else "上位分野"
    recent_history = history[-8:] if history else []
    history_str = ", ".join(recent_history) if recent_history else "なし"

    sys = "あなたは学術分野の階層構造を作成するAIです。"

    parent_rules = f"""
【出力粒度（親/parent のみ厳守）】
- 「直上（1階層上）の上位概念」を返すこと。例：入力が「RAG」なら「LLM応用」「情報検索応用」は可、「人工知能」「情報科学」は不可。
- 避ける語：['科学','技術','工学','情報科学','コンピュータサイエンス','データサイエンス','機械学習','人工知能','アルゴリズム','最適化'] など過度に汎用的な語。
- 入力に含まれる語のヘッド（例:『検索』『支援』『評価』『設計』『インタフェース』『システム』等）を保ったまま1段だけ上げること。
- 返す語は具体的で、研究計画書や学会トラック名として違和感がないもの。
- 多様性: 「方法論」「応用領域」「システム/UI」「評価/実験設計」など視点が被らないよう候補を挙げること。
- 件数: 一次候補は {PARENT_GEN_COUNT} 件 程度（過剰なら近いものから省略可）
良い例:
- 入力:「文献検索支援」→ 親:「学術情報検索」「研究支援システム」「探索的検索インタフェース」
- 入力:「注意機構」→ 親:「ニューラル注意」「Transformerアーキテクチャ」「系列モデリング手法」
悪い例:
- 入力:何であれ → 親:「情報科学」「AI」「コンピュータサイエンス」
""".strip()

    usr = f"""
中央のトピック「{query}」に対して、{mode_jp}を生成してください。
{parent_rules if mode=='parent' else ''}

【指示】
- 「{query}」と完全一致または部分一致する語は禁止。
- これまでの履歴 ({history_str}) に含まれる語は絶対に出さない。
- 同義語（AIと人工知能など）も避ける。
- 下位トピック（child）は具体的な研究テーマ・手法を3〜5個。
- 上位分野（parent）は{query}を包含する「直上の」学問領域を{PARENT_GEN_COUNT}個 程度挙げる（過度に汎用な語は禁止）。
- 出力は必ず JSON形式 のみで行う。
- 形式：{{ "{mode}s": ["語1", "語2", "語3"] }}
""".strip()

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


def build_describe_messages(term: str) -> List[Dict[str, str]]:
    sys = "あなたは学術用語の略語と説明を返すアシスタントです。"
    usr = f"""
用語: 「{term}」

要件:
- 出力はJSONのみ。
- キーは "short", "full", "desc"。
- "short": 広く使われる略語が存在すればそれ（最大10文字）。なければ元語を10文字以内に短縮。
- "full": 正式名称（日本語があれば日本語、英語があれば英語も併記可）。
- "desc": 日本語で120字以内の要点説明（専門的すぎない簡潔さ）。
""".strip()
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


def build_parent_axes_messages(query: str, history: List[str]) -> List[Dict[str, str]]:
    recent_history = history[-8:] if history else []
    history_str = ", ".join(recent_history) if recent_history else "なし"
    sys = "あなたは学術分野の階層構造を作成するAIです。"
    usr = f"""
中央のトピック「{query}」に対して、「直上の上位概念」を多様な視点から提案してください。
【軸】方法論 / 応用領域 / システム・UI / 評価・実験設計
【禁止】
- 「{query}」と完全一致または部分一致する語
- 履歴({history_str})に含まれる語
- 汎用語（AI, 情報科学, 機械学習 など）
【出力】
- JSONのみ。スキーマ: {{"parents": ["語1","語2","語3","語4"]}}
- それぞれの軸から重複しないよう 4 件 程度。
""".strip()
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


def build_translate_messages(terms: List[str]) -> List[Dict[str, str]]:
    sys = "You are a precise technical translator for academic search queries. Output JSON only."
    usr = {
        "instruction": """
Return English research keywords for each Japanese term.
Rules:
- Keep 1-2 concise English noun phrases per item (no sentences).
- Prefer field-specific terms used in academic literature.
- If input already looks English, just repeat it.
- JSON only, schema: {"translations":[{"src":"<input>","en":["<en1>","<en2>"]}, ...]}
""".strip(),
        "terms": terms,
    }
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(usr, ensure_ascii=False)},
    ]


def build_child_axes_messages(query: str, history: List[str], need_n: int) -> List[Dict[str, str]]:
    recent_history = history[-12:] if history else []
    history_str = ", ".join(recent_history) if recent_history else "なし"

    sys = "あなたは学術分野の探索を支援するために、具体的な研究トピック候補を挙げるアシスタントです。"
    usr = f"""
中央のトピック「{query}」に対して、下位トピック（具体的な研究テーマ・手法）を追加で提案してください。

【ねらい】
- すでに出た候補が不足しているため、追加分として {need_n} 件以上を挙げる。

【観点（偏り防止のヒント）】
- 研究課題 / 対象 / 手法 / 評価 のいずれかが異なるように、なるべく多様にする。

【禁止】
- 「{query}」と完全一致または部分一致する語
- 履歴({history_str})に含まれる語（表記ゆれ・同義語も避ける）
- 文章や説明文ではなく、検索語として使える名詞句にする

【出力】
- JSONのみ。スキーマ: {{"children": ["語1","語2","語3", ...]}}
""".strip()

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


# === LLM: 子/親生成 ===
def generate_related_terms(query: str, history: list, mode="child"):
    t0 = time.time()
    messages = build_expand_messages(query, history, mode)
    res = None
    try:
        res = call_chat_json(
            model=MODEL,
            messages=messages,
            temperature=EXPAND_TEMP,
            max_tokens=EXPAND_MAXTOK,
            timeout_s=EXPAND_TIMEOUT,
            retries=RETRIES,
        )
        content = res.choices[0].message.content or "{}"
        obj = json.loads(content)
    except Exception as e:
        print("⚠️ JSON parse error:", e)
        obj = {f"{mode}s": []}

    keys = {k.lower(): k for k in obj.keys()}
    normalized = {"children": [], "parents": []}
    for k in keys:
        if "child" in k:
            normalized["children"] = ensure_list(obj[keys[k]])
        if "parent" in k:
            normalized["parents"] = ensure_list(obj[keys[k]])

    prevs = {canon(w) for w in (history + [query])}
    for side in ("children", "parents"):
        cleaned = []
        for t in normalized[side]:
            c = canon(t)
            if not c or is_dup(c, prevs):
                continue
            cleaned.append(t)
            prevs.add(c)
        normalized[side] = cleaned

    if mode == "parent":
        refined = refine_parent_candidates(
            query,
            normalized.get("parents", []),
            jacc_min=0.15,
            jacc_max=0.85,
            limit=PARENT_TARGET_MAX,
        )
        if not refined:
            try:
                messages2 = build_expand_messages(query, history, mode="parent")
                messages2[-1]["content"] += "\n必須: 直上1階層のみ。汎用語は禁止。入力語のヘッドを保持すること。"
                res2 = call_chat_json(
                    model=MODEL,
                    messages=messages2,
                    temperature=max(0.1, EXPAND_TEMP - 0.1),
                    max_tokens=EXPAND_MAXTOK,
                    timeout_s=EXPAND_TIMEOUT,
                    retries=RETRIES,
                )
                obj2 = json.loads((res2.choices[0].message.content or "{}"))
                parents2 = obj2.get("parents", obj2.get("parent", [])) or []
                refined = refine_parent_candidates(
                    query,
                    ensure_list(parents2),
                    jacc_min=0.15,
                    jacc_max=0.85,
                    limit=PARENT_TARGET_MAX,
                )
            except Exception:
                refined = []

        if len(refined) < PARENT_TARGET_MIN:
            try:
                res3 = call_chat_json(
                    model=MODEL,
                    messages=build_parent_axes_messages(query, history),
                    temperature=min(0.5, EXPAND_TEMP + 0.2),
                    max_tokens=EXPAND_MAXTOK,
                    timeout_s=EXPAND_TIMEOUT,
                    retries=RETRIES,
                )
                obj3 = json.loads((res3.choices[0].message.content or "{}"))
                parents3 = ensure_list(obj3.get("parents") or obj3.get("parent") or [])
                extra = refine_parent_candidates(
                    query,
                    parents3,
                    jacc_min=0.12,
                    jacc_max=0.9,
                    limit=None,
                )
                pool = []
                seen = set(canon(p) for p in refined)
                for p in extra:
                    ck = canon(p)
                    if ck and ck not in seen:
                        seen.add(ck)
                        pool.append(p)
                refined = (refined + pool)[:PARENT_TARGET_MAX]
            except Exception:
                pass

        if len(refined) < PARENT_TARGET_MIN:
            seed = [p for p in normalized.get("parents", []) if not is_generic_parent(p)]
            for p in seed:
                ck = canon(p)
                if all(canon(x) != ck for x in refined):
                    refined.append(p)
                if len(refined) >= PARENT_TARGET_MIN:
                    break

        normalized["parents"] = refined[:PARENT_TARGET_MAX]

    if mode == "child":
        children = normalized.get("children", [])
        if len(children) < CHILD_TARGET_MIN and CHILD_RETRY_MAX > 0:
            need_n = CHILD_TARGET_MIN - len(children)
            try:
                hist2 = list(history)
                for w in children:
                    if w not in hist2:
                        hist2.append(w)

                res_c = call_chat_json(
                    model=MODEL,
                    messages=build_child_axes_messages(query, hist2, need_n),
                    temperature=min(0.6, EXPAND_TEMP + 0.2),
                    max_tokens=EXPAND_MAXTOK,
                    timeout_s=EXPAND_TIMEOUT,
                    retries=RETRIES,
                )
                obj_c = json.loads((res_c.choices[0].message.content or "{}"))
                extra_children = ensure_list(obj_c.get("children") or obj_c.get("child") or [])

                prevs2 = {canon(w) for w in (history + [query] + children)}
                for t in extra_children:
                    c = canon(t)
                    if not c or is_dup(c, prevs2):
                        continue
                    children.append(t)
                    prevs2.add(c)

                normalized["children"] = children[:CHILD_TARGET_MAX]
            except Exception:
                normalized["children"] = children[:CHILD_TARGET_MAX]

    new_hist = list(history)
    for w in normalized.get("parents", []) + normalized.get("children", []):
        if w not in new_hist:
            new_hist.append(w)

    latency_ms = int((time.time() - t0) * 1000)
    usage = None
    try:
        if res and getattr(res, "usage", None):
            u = res.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
    except Exception:
        usage = None

    write_event(
        {
            "action": "expand",
            "ts": time.time(),
            "sid": session.get("sid"),
            "node": query,
            "side": mode,
            "latency_ms": latency_ms,
            "token_usage": usage,
        }
    )

    return normalized, new_hist


# === LLM: 用語説明 & 略語提案 ===
def describe_term(term: str):
    m = get_known_meta(term)
    if m:
        return m

    meta_cache = session.get("meta_cache", {})
    key = canon(term)
    if key in meta_cache:
        return meta_cache[key]

    try:
        res = call_chat_json(
            model=MODEL,
            messages=build_describe_messages(term),
            temperature=DESCR_TEMP,
            max_tokens=DESCR_MAXTOK,
            timeout_s=DESCR_TIMEOUT,
            retries=RETRIES,
        )
        obj = json.loads(res.choices[0].message.content or "{}")
        short = (obj.get("short") or term)[:10]
        full = obj.get("full") or term
        desc = (obj.get("desc") or "")[:200]
        meta = {"short": short, "full": full, "desc": desc}
    except Exception as e:
        print("⚠️ describe_term error:", e)
        meta = {"short": term[:10], "full": term, "desc": "説明を生成できませんでした。"}

    meta_cache[key] = meta
    session["meta_cache"] = meta
    return meta


# === Routes ===
@app.route("/", methods=["GET", "POST"])
def enter_subject():
    """
    実験用の最初の画面：被験者ID（P01〜P08など）を入力してもらう。
    - GET: ID未入力ならフォームを表示、入力済みなら /graph へリダイレクト
    - POST: subject_id をセッションに保存し /graph へ
    """
    session.permanent = True

    if request.method == "POST":
        subject_id = (request.form.get("subject_id") or "").strip()
        if subject_id:
            session.clear()
            session.permanent = True
            session["subject_id"] = subject_id
            session["sid"] = uuid.uuid4().hex[:12]
            session["history"] = []
            session["last_root"] = None
            session["meta_cache"] = {}
            session["trans_cache"] = {}

            write_event({"action": "session_start", "ts": time.time(), "subject_id": subject_id})
            return redirect(url_for("graph"))

    if session.get("subject_id"):
        return redirect(url_for("graph"))

    return render_template("subject_id.html")


@app.route("/graph")
def graph():
    """メインの探索UI。subject_id が未設定ならトップに戻す。"""
    if "subject_id" not in session:
        return redirect(url_for("enter_subject"))

    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex[:12]

    return render_template("graph.html")


@app.route("/end", methods=["POST"])
def end_session():
    """実験終了時にフロントから呼ぶ用のエンドポイント（任意）。"""
    write_event(
        {
            "action": "session_end",
            "ts": time.time(),
            "sid": session.get("sid"),
            "subject_id": session.get("subject_id"),
        }
    )
    return jsonify({"status": "ok"})


@app.route("/meta", methods=["GET"])
def meta():
    return jsonify(export_known_meta_with_aliases())


@app.route("/expand", methods=["POST"])
def expand():
    data = request.get_json(silent=True) or {}
    node = (data.get("node") or "").strip()
    mode = data.get("mode", "child")

    if not node:
        return jsonify({"parents": [], "children": []})

    if mode == "reset":
        session["history"] = []
        session["last_root"] = node
        write_event({"action": "reset", "ts": time.time(), "sid": session.get("sid"), "node": node})
        return jsonify({"parents": [], "children": []})

    if mode == "recenter":
        session["last_root"] = node
        history = session.get("history", [])
        if node not in history:
            history.append(node)
        session["history"] = history
        write_event({"action": "recenter", "ts": time.time(), "sid": session.get("sid"), "node": node})
        return jsonify({"parents": [], "children": []})

    history = session.get("history", [])
    if node not in history:
        history.append(node)

    result, new_hist = generate_related_terms(node, history, mode)
    session["history"] = new_hist
    session["last_root"] = session.get("last_root", node)
    return jsonify(result)


@app.route("/describe", methods=["POST"])
def describe():
    data = request.get_json(silent=True) or {}
    term = (data.get("term") or "").strip()
    if not term:
        return jsonify({"short": "", "full": "", "desc": ""})
    return jsonify(describe_term(term))


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}
    terms = data.get("terms") or []
    terms = [t for t in terms if isinstance(t, str) and t.strip()]
    if not terms:
        return jsonify({"map": {}})

    tcache = session.get("trans_cache", {})
    out_map: Dict[str, List[str]] = {}
    need: List[str] = []

    for t in terms:
        ck = canon(t)
        if ck in tcache:
            out_map[t] = tcache[ck]
        else:
            need.append(t)

    if need:
        try:
            res = call_chat_json(
                model=MODEL,
                messages=build_translate_messages(need),
                temperature=max(0.1, EXPAND_TEMP - 0.1),
                max_tokens=300,
                timeout_s=8,
                retries=RETRIES,
            )
            obj = json.loads(res.choices[0].message.content or "{}")
            items = obj.get("translations", [])
            for it in items:
                src = (it.get("src") or "").strip()
                ens = [e.strip() for e in (it.get("en") or []) if e and isinstance(e, str)]
                if src and ens:
                    out_map[src] = ens
                    tcache[canon(src)] = ens
        except Exception as e:
            print("⚠️ translate error:", e)

    session["trans_cache"] = tcache
    write_event({"action": "translate", "ts": time.time(), "sid": session.get("sid"), "count": len(terms)})
    return jsonify({"map": out_map})


@app.route("/log_ui", methods=["POST"])
def log_ui():
    """
    検索UIまわりの操作ログ用エンドポイント。
    例:
      event_type: "keyword_add" / "keyword_remove" / "search_submit"
      payload:    { ... 任意の情報 ... }
    """
    data = request.get_json(silent=True) or {}
    event_type = (data.get("event_type") or "").strip() or "ui"
    payload = data.get("payload") or {}

    write_event(
        {
            "action": "ui",
            "ui_event": event_type,
            "ts": time.time(),
            "sid": session.get("sid"),
            "subject_id": session.get("subject_id"),
            "detail": payload,
        }
    )
    return jsonify({"status": "ok"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    debug = os.getenv("APP_ENV", "development") != "production"
    app.run(host="127.0.0.1", port=5000, debug=debug)
