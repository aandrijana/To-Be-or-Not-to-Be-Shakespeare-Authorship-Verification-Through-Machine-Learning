import re
import html


def get_highlighted_text(text, feature_names, shap_values, max_chars=2000, top_n=60, min_abs_impact=0.001):
    """
    Scans the preprocessed text sequence and highlights character n-grams
    or function tokens based on their localized model impact scores.

    Matches are computed against the text in a single pass and then
    rendered once, so overlapping/short features (e.g. "or", "in") can never
    match inside HTML already injected by a previous replacement.
    """
    if not text:
        return "No text context found for display."

    display_segment = text[:max_chars]
    shap_dict = dict(zip(feature_names, shap_values))

    candidate_features = [
        f for f in feature_names
        if isinstance(f, str) and len(f) > 1 and abs(shap_dict.get(f, 0)) > min_abs_impact
    ]

    candidate_features = sorted(candidate_features, key=lambda f: abs(shap_dict.get(f, 0)), reverse=True)[:top_n]

    spans = []
    for feat in candidate_features:
        impact = shap_dict.get(feat, 0)
        try:
            pattern = re.compile(re.escape(feat), flags=re.IGNORECASE)
        except re.error:
            continue
        for m in pattern.finditer(display_segment):
            spans.append((m.start(), m.end(), impact))

    if not spans:
        return html.escape(display_segment)

    spans.sort(key=lambda s: (-(s[1] - s[0]), -abs(s[2])))
    occupied = [False] * len(display_segment)
    chosen = []
    for start, end, impact in spans:
        if any(occupied[start:end]):
            continue
        occupied[start:end] = [True] * (end - start)
        chosen.append((start, end, impact))
    chosen.sort(key=lambda s: s[0])

    pieces = []
    cursor = 0
    for start, end, impact in chosen:
        pieces.append(html.escape(display_segment[cursor:start]))
        color = "#9ef59e" if impact > 0 else "#ff8f8f"
        token = html.escape(display_segment[start:end])
        pieces.append(
            f'<span style="background-color:{color}; padding:1px 2px; border-radius:2px; '
            f'font-weight:bold; border: 1px solid rgba(0,0,0,0.15)">{token}</span>'
        )
        cursor = end
    pieces.append(html.escape(display_segment[cursor:]))

    return "".join(pieces)