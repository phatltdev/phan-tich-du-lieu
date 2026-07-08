from __future__ import annotations

import html
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "reuters21578"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"
LOG_PATH = LOGS / "reuters_analysis.log"


def ensure_dirs() -> None:
    for directory in [FIGURES, TABLES, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, filemode="w", format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def tag_values(block: str, tag: str) -> list[str]:
    m = re.search(fr"<{tag}>(.*?)</{tag}>", block, flags=re.S | re.I)
    if not m:
        return []
    return re.findall(r"<D>(.*?)</D>", m.group(1), flags=re.S | re.I)


def tag_text(block: str, tag: str) -> str:
    m = re.search(fr"<{tag}>(.*?)</{tag}>", block, flags=re.S | re.I)
    if not m:
        return ""
    text = re.sub(r"<[^>]+>", " ", m.group(1))
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def parse_docs() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.glob("reut2-*.sgm")):
        raw = path.read_text(encoding="latin-1", errors="ignore")
        for block in re.findall(r"<REUTERS\b.*?</REUTERS>", raw, flags=re.S | re.I):
            newid = re.search(r'NEWID="(\d+)"', block)
            title = tag_text(block, "TITLE")
            body = tag_text(block, "BODY")
            text = f"{title} {body}".strip()
            topics = tag_values(block, "TOPICS")
            rows.append({
                "newid": int(newid.group(1)) if newid else None,
                "topics": ";".join(topics),
                "primary_topic": topics[0] if topics else "NO_TOPIC",
                "word_count": len(re.findall(r"\b\w+\b", text)),
                "char_count": len(text),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    setup_logging()
    sns.set_theme(style="whitegrid")

    df = parse_docs()
    df.to_csv(TABLES / "reuters_text_features.csv", index=False)

    topic_freq = df[df["primary_topic"] != "NO_TOPIC"]["primary_topic"].value_counts().head(15).rename_axis("topic").reset_index(name="count")
    topic_freq.to_csv(TABLES / "reuters_top_topics.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=topic_freq, x="count", y="topic", color="#C44E52")
    plt.title("Reuters-21578 top primary topics")
    plt.tight_layout()
    plt.savefig(FIGURES / "reuters_top_topics.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df.loc[df["word_count"] > 0, "word_count"].clip(upper=1000), bins=40, color="#8172B3")
    plt.title("Reuters document word-count distribution (clipped at 1000)")
    plt.xlabel("word_count")
    plt.tight_layout()
    plt.savefig(FIGURES / "reuters_word_count_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df.loc[df["char_count"] > 0, "char_count"].clip(upper=6000), bins=40, color="#4C78A8")
    plt.title("Reuters document character-count distribution (clipped at 6000)")
    plt.xlabel("char_count")
    plt.tight_layout()
    plt.savefig(FIGURES / "reuters_char_count_histogram.png", dpi=200)
    plt.close()

    sample = df[(df["word_count"] > 0) & (df["char_count"] > 0)].sample(
        min(5000, len(df[(df["word_count"] > 0) & (df["char_count"] > 0)])),
        random_state=42,
    )
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=sample,
        x=sample["word_count"].clip(upper=1000),
        y=sample["char_count"].clip(upper=6000),
        hue="primary_topic",
        legend=False,
        alpha=0.35,
        s=14,
    )
    plt.title("Reuters word count vs character count")
    plt.xlabel("word_count (clipped at 1000)")
    plt.ylabel("char_count (clipped at 6000)")
    plt.tight_layout()
    plt.savefig(FIGURES / "reuters_word_char_scatter.png", dpi=200)
    plt.close()

    logging.info("Reuters docs=%s", len(df))


if __name__ == "__main__":
    main()
