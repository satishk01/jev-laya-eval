"""A minimal ReAct agent: reads queries from stdin, writes progress and answers to stdout.

No agent framework, just the OpenAI SDK (Responses API). The agent loop, LLM
calls and tool calls are traced as deepeval spans with @observe, so evals can
inspect what happens inside a run, not just the final answer. Each LLM call is
constrained to a JSON schema describing one ReAct step: a thought, then either
an action (keyword search of the knowledge base) or a final answer. The loop runs the
action, feeds the observation back, and repeats until the model answers.

Configuration (from .env):
    AGENT_BACKEND         "llm" (default) or "extractive" (no LLM: a deterministic
                          stand-in that searches and quotes the knowledge base,
                          so the eval stack runs with zero API keys)
    INFERENCE_BASE_URL    API root, e.g. https://.../v1
    INFERENCE_API_KEY     API key for that endpoint
    INFERENCE_MODEL_NAME  model id
    INFERENCE_API_STYLE   "responses" (OpenAI Responses API) or "chat"
                          (chat completions, for Ollama/LM Studio/vLLM etc.)
    AGENT_TOOLS           comma list; "search_knowledge" is always on. Add
                          "list_sections" to give the agent a second tool that
                          returns the knowledge base's section outline.

Usage:
    uv run agent.py            # one query per line; /exit or Ctrl-D to quit
    uv run agent.py -v         # also print the ReAct thoughts to stderr
    echo "What is RAG?" | uv run agent.py
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Traces are only for evals; don't print Confident AI trace-posting notices on
# normal runs. Set before importing deepeval, which reads it at startup.
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")

from nltk.stem import PorterStemmer  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

from deepeval.tracing import observe, update_current_span  # noqa: E402
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

KNOWLEDGE_PATH = Path(__file__).parent / "agent-knowledge.md"
BACKEND = os.environ.get("AGENT_BACKEND", "llm")
API_STYLE = os.environ.get("INFERENCE_API_STYLE", "responses")
MODEL = os.environ.get("INFERENCE_MODEL_NAME", "extractive")
MAX_USER_TURNS = 5  # context window: keep only the last 5 user messages
MAX_STEPS = 5  # ReAct steps per query, so a confused model can't loop forever
MAX_OUTPUT_TOKENS = 1024  # per LLM call; reasoning models count reasoning tokens here too
MAX_SEARCH_RESULTS = 24  # paragraphs per search, so a broad keyword can't return the whole file
MAX_SEARCH_KEYWORDS = 5  # the tool's contract, mirrored in STEP_SCHEMA below
VERBOSE = "-v" in sys.argv[1:]

# Tools the agent may call. search_knowledge is always on; list_sections (the
# knowledge base's own section outline, for "what topics do you cover" style
# meta-questions keyword search can't answer) is opt-in.
TOOLS = {"search_knowledge"} | {
    t.strip() for t in os.environ.get("AGENT_TOOLS", "").split(",") if t.strip()
}
ACTIONS = ["search_knowledge", "final_answer"]
if "list_sections" in TOOLS:
    ACTIONS.insert(1, "list_sections")

_ACTION_DESC = (
    'either "search_knowledge" or "final_answer"'
    if len(ACTIONS) == 2
    else "one of " + ", ".join(f'"{a}"' for a in ACTIONS)
)
_LIST_SECTIONS_PROMPT = """
The "list_sections" action returns the knowledge base's section outline -- use
it for questions about what the knowledge base covers ("What topics do you know
about?"), not for questions about the topics themselves. It takes no input, so
"keywords" stays [].
""" if "list_sections" in TOOLS else ""

SYSTEM_PROMPT = f"""You are a helpful assistant that answers beginner questions about AI engineering.

You work in a ReAct loop. Each reply is one step, as JSON:
- "thought": brief reasoning about what to do next.
- "action": {_ACTION_DESC}.
- "keywords": for "search_knowledge", 1-5 single-word keywords to search for, otherwise [].
- "answer": the answer for the user when action is "final_answer", otherwise "".

The "search_knowledge" action does a keyword search of the knowledge base and
returns the best-matching paragraphs (up to a limit) with the headings they
sit under, as an observation. Paragraphs matching more keywords rank higher. A paragraph matches if it contains any keyword; a
keyword also matches words that start with it, so "token" matches "tokens".
Prefer specific terms and synonyms (e.g. "RAG", "retrieval") over generic
words like "model". If the results are not enough, search again with
different keywords.
{_LIST_SECTIONS_PROMPT}
Search before answering any question about AI engineering, unless the relevant
passages are already present in this conversation. Base your answers on the
knowledge base. If it does not cover the question, say so rather than guessing.

If a search matches nothing, search again before giving up: split hyphenated
terms into separate words, and try other spellings and word forms, e.g.
"summarisation" for "summarization", or a shorter stem such as "summar".

If any query is not found in the results of "search_knowledge", answer exactly "Sorry I cannot help with that."

Answer in plain text, in 3-6 sentences that cover every relevant detail from the search results."""

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ACTIONS},
        # Enforces the tool's contract during decoding: at most 5 keywords, each
        # a single word (no whitespace). Empty for final_answer, so no minItems.
        "keywords": {
            "type": "array",
            "items": {"type": "string", "pattern": "^\\S+$"},
            "maxItems": MAX_SEARCH_KEYWORDS,
        },
        "answer": {"type": "string"},
    },
    "required": ["thought", "action", "keywords", "answer"],
    "additionalProperties": False,
}


def load_paragraphs(path: Path) -> list[tuple[list[str], str]]:
    """Split a Markdown file into (headings it sits under, paragraph text) pairs."""
    paragraphs = []
    headings: list[tuple[int, str]] = []  # (level, heading line), outermost first
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        block = block.strip()
        heading = re.match(r"(#+)\s", block)
        if heading:
            level = len(heading.group(1))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, block))
        elif block:
            paragraphs.append(([line for _, line in headings], block))
    return paragraphs


_paragraphs: Optional[list[tuple[list[str], str]]] = None


def knowledge_paragraphs() -> list[tuple[list[str], str]]:
    """The knowledge base, parsed once per process."""
    global _paragraphs
    if _paragraphs is None:
        _paragraphs = load_paragraphs(KNOWLEDGE_PATH)
    return _paragraphs


def word_pattern(word: str) -> re.Pattern:
    """The search's matching rule: word boundary, prefix match, case-insensitive."""
    return re.compile(r"\b" + re.escape(word), re.IGNORECASE)


@observe(type="tool")
def search_knowledge(keywords: list[str]) -> str:
    """The agent's default tool: keyword search over the knowledge base paragraphs."""
    # The span records the keywords exactly as the model sent them, as JSON.
    update_current_span(input=to_json({"keywords": keywords}))
    # Phrases are split into words, so "RAG chunking" searches for "RAG" and "chunking".
    keywords = list(dict.fromkeys(word for k in keywords for word in k.split()))
    if not keywords:
        return "No keywords given. Provide 1-5 keywords to search for."

    # Match at the start of a word, case-insensitively, so "token" also matches "tokens".
    patterns = [word_pattern(k) for k in keywords]
    matches = []
    for position, (headings, text) in enumerate(knowledge_paragraphs()):
        hits = sum(1 for pattern in patterns if pattern.search(text))
        if hits:
            matches.append((hits, position, headings, text))
    if not matches:
        return f"No paragraphs matched: {', '.join(keywords)}"

    # Keep the paragraphs matching the most keywords, then show them in file order.
    top = sorted(matches, key=lambda m: (-m[0], m[1]))[:MAX_SEARCH_RESULTS]
    results = [
        "\n".join(headings + [text]) for _, _, headings, text in sorted(top, key=lambda m: m[1])
    ]
    summary = f"{len(matches)} paragraphs matched; showing {len(top)}."
    return "\n\n---\n\n".join([summary] + results)


@observe(type="tool")
def list_sections() -> str:
    """Opt-in second tool: the knowledge base's section outline.

    Answers meta-questions about coverage ("What topics do you know about?"),
    which search_knowledge can't -- no paragraph in the file describes the
    file's own contents. Enabled via AGENT_TOOLS=search_knowledge,list_sections.
    """
    update_current_span(input=to_json({}))
    seen: set[str] = set()
    sections = []
    for headings, _ in knowledge_paragraphs():
        for heading in headings:
            if heading.startswith("## ") and heading not in seen:
                seen.add(heading)
                sections.append(heading[3:])
    return f"{len(sections)} sections:\n" + "\n".join(f"- {s}" for s in sections)


def make_client() -> Optional[OpenAI]:
    """None under the extractive backend, which makes no LLM calls."""
    if BACKEND == "extractive":
        return None
    return OpenAI(
        base_url=os.environ["INFERENCE_BASE_URL"],
        api_key=os.environ["INFERENCE_API_KEY"],
    )


# Words dropped when the extractive backend turns a query into search keywords.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "its", "me", "of", "on", "or", "the",
    "to", "what", "when", "where", "which", "why", "with", "you",
    # Comparison scaffolding: "the difference between X and Y" is about X and
    # Y -- "difference" and "between" describe the question's shape, not its
    # topic, yet they're rare enough in the file to outrank the topical words.
    "difference", "differences", "between", "versus", "vs", "compare",
    "comparing", "comparison",
    # "AI" is the whole knowledge base's topic: it appears in every heading
    # and nearly every paragraph, so as a keyword it only adds noise.
    "ai",
}


def match_count(pattern: re.Pattern) -> int:
    """How many knowledge-base paragraphs match the pattern."""
    return sum(1 for _, text in knowledge_paragraphs() if pattern.search(text))


_stemmer = PorterStemmer()
# A stem only counts as a keyword when it matched words of the query word's own
# lexeme -- otherwise "brea" (from "bread") happily prefix-matches "break" and
# an out-of-scope query gets an answer instead of a refusal.
STEM_RELATED_CUTOFF = 85  # rapidfuzz ratio; "bread"~"break" is 80, real spelling variants score 90+


def stem_surface_words(stem: str) -> set[str]:
    """The whole words in the knowledge base a stem prefix-matches."""
    pattern = re.compile(r"\b" + re.escape(stem) + r"\w*", re.IGNORECASE)
    return {
        m.group(0).lower()
        for _, text in knowledge_paragraphs()
        for m in pattern.finditer(text)
    }


def stem_is_related(stem: str, word: str) -> bool:
    """Whether the words a stem matched plausibly share the query word's lexeme:
    "summari" hits "summarisation" for "summarization", but "brea" only hits
    "break" for "bread". Porter stems cover morphological variants
    ("moving"~"move"), fuzzy ratio covers spelling variants the stemmer splits
    ("summarization"~"summarisation")."""
    base = _stemmer.stem(word.lower())
    return any(
        _stemmer.stem(matched) == base
        or fuzz.ratio(word.lower(), matched) >= STEM_RELATED_CUTOFF
        for matched in stem_surface_words(stem)
    )


def keyword_variants(word: str) -> set[str]:
    """A query word as searchable keyword(s): its singular form, hyphen-separated
    parts, or its longest stem that occurs in the knowledge base.

    The variants keep queries that don't quote the text searchable: "traces"
    finds "a trace", "sliding-window" finds "sliding window", and
    "summarization" finds "summarisation". A shorter matching form dominates
    the word itself because the tool's match is a prefix match, so the stem
    also covers the word. A stem that only matched unrelated words is
    rejected -- the full word is searched instead, so out-of-scope queries
    still reach the refusal path.
    """
    if "-" in word:
        parts = {p for p in word.split("-") if len(p) > 1 and match_count(word_pattern(p))}
        if parts:
            return parts
    if word.endswith("s") and len(word) > 4 and word[-2] not in "sui":
        singular = word[:-1]
        if match_count(word_pattern(singular)):
            return {singular}
    if match_count(word_pattern(word)):
        return {word}
    for cut in range(len(word) - 1, 3, -1):
        stem = word[:cut]
        if match_count(word_pattern(stem)) and stem_is_related(stem, word):
            return {stem}
    return {word}


def extractive_keywords(query: str) -> list[str]:
    """The query's content words resolved into search keywords, most useful first.

    Only five keywords fit a search call, and the first words of a query are
    usually the generic ones ("What does PII include in AI engineering, such
    as ..."), so keywords are ranked by how few paragraphs they match: rare
    terms like "PII" narrow the search where "AI" or "engineering" match the
    whole file. Words absent from the knowledge base rank last -- they match
    nothing -- and the query's own order breaks remaining ties.
    """
    position = {}
    for w in dict.fromkeys(
        w for w in re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z]+)?", query)
        if w.lower() not in STOPWORDS and len(w) > 1
    ):
        for k in keyword_variants(w):
            position.setdefault(k, len(position))
    keywords = list(position)
    frequency = {k: match_count(word_pattern(k)) for k in keywords}
    keywords.sort(key=lambda k: (frequency[k] == 0, frequency[k], position[k]))
    return keywords


# A query asks about the knowledge base's own coverage rather than a topic
# inside it -- disjoint from every topical query by design, so routing these
# to list_sections never steals a query search_knowledge would have answered.
META_QUERY_PATTERNS = (
    r"\bwhat (?:can|do|does) you\b",
    r"\b(?:topics?|sections?|subjects?|areas|categories)\b[^?]*\b(?:cover|list|offer|have|know|answer|support)\b",
    r"\b(?:cover|list|show|tell)\b[^?]*\b(?:topics?|sections?|subjects?|areas|categories)\b",
    r"\bknowledge base\b[^?]*\b(?:cover|contain|include|about|offer|have|list)\b",
)


def wants_section_listing(query: str) -> bool:
    return any(re.search(p, query, re.IGNORECASE) for p in META_QUERY_PATTERNS)


def extractive_step(turn: list[dict]) -> dict:
    """The extractive backend's ReAct step: real search calls, a stitched answer.

    Same JSON shape as a call_llm step, so the tool spans it produces look
    identical to the LLM agent's in a trace."""
    query = turn[0]["content"]
    observations = [
        m["content"].split(":", 1)[1]
        for m in turn
        if m["role"] == "user" and m["content"].startswith("Observation from search_knowledge:")
    ]
    section_observations = [
        m["content"].split(":", 1)[1]
        for m in turn
        if m["role"] == "user" and m["content"].startswith("Observation from list_sections:")
    ]
    words = extractive_keywords(query)

    if "list_sections" in TOOLS and wants_section_listing(query):
        if not section_observations:
            return {
                "thought": "The query asks what the knowledge base covers; listing its sections.",
                "action": "list_sections",
                "keywords": [],
                "answer": "",
            }
        listing = section_observations[-1].strip()
        items = [
            line[2:].strip() for line in listing.splitlines() if line.startswith("- ")
        ]
        return {
            "thought": "The section outline answers the question.",
            "action": "final_answer",
            "keywords": [],
            "answer": f"The knowledge base covers {len(items)} sections: "
            + ", ".join(items)
            + ".",
        }
    if not observations:
        keywords = words[:5] or ["model"]
        return {
            "thought": "I'll search the knowledge base for the query's terms.",
            "action": "search_knowledge",
            "keywords": keywords,
            "answer": "",
        }

    if all(o.startswith("No paragraphs matched") for o in observations):
        if len(observations) == 1:
            # Retry once, like the system prompt says: split hyphenated terms
            # and fall back to the longest word stems -- verified, so a stem
            # can't resurface unrelated words ("brea" matching "break").
            retried = [
                w for word in words for w in word.split("-") if w.lower() not in STOPWORDS
            ] or [
                w[:7]
                for w in sorted(words, key=len, reverse=True)
                if len(w) > 7 and stem_is_related(w[:7], w)
            ][:2]
            if not retried:
                return {
                    "thought": "The knowledge base does not cover this question.",
                    "action": "final_answer",
                    "keywords": [],
                    "answer": "Sorry I cannot help with that.",
                }
            return {
                "thought": "Nothing matched; retrying with simpler keywords.",
                "action": "search_knowledge",
                "keywords": list(dict.fromkeys(retried))[:5],
                "answer": "",
            }
        return {
            "thought": "The knowledge base does not cover this question.",
            "action": "final_answer",
            "keywords": [],
            "answer": "Sorry I cannot help with that.",
        }

    # A query often asks about more facets than five keywords reach, so spend
    # further searches on the leftover terms before answering.
    next_keywords = words[len(observations) * MAX_SEARCH_KEYWORDS :]
    if next_keywords and len(observations) < 3:
        return {
            "thought": "Searching for the query's remaining terms too.",
            "action": "search_knowledge",
            "keywords": next_keywords[:MAX_SEARCH_KEYWORDS],
            "answer": "",
        }

    # Stitch an answer from the best-matching paragraphs across the
    # observations, up to two sentences from each top block rather than the
    # first four of one. Hits are weighted by 1/frequency, so a rare topical
    # term ("sliding") outweighs generic ones ("model", "system") that match
    # everywhere; a keyword in a block's heading lines counts double because
    # the section title says what the paragraph is about.
    def block_hits(block: str) -> set[str]:
        # Body hits only: blocks from one section share their heading lines,
        # so heading hits are the same for all of them and can't discriminate.
        body = " ".join(
            line for line in block.splitlines() if not line.startswith("#")
        )
        return {w for w, p in zip(words, patterns) if p.search(body)}

    def block_score(block: str) -> float:
        lines = block.splitlines()
        headings = " ".join(line for line in lines if line.startswith("#"))
        body = " ".join(line for line in lines if not line.startswith("#"))
        return (
            2 * sum(weight[w] for w, p in zip(words, patterns) if p.search(headings))
            + sum(weight[w] for w, p in zip(words, patterns) if p.search(body))
        )

    blocks = list(dict.fromkeys(
        block
        for observation in observations
        for block in observation.split("\n\n---\n\n")[1:]  # [0] is the summary
    ))
    patterns = [word_pattern(w) for w in words]
    weight = {w: 1.0 / max(match_count(word_pattern(w)), 1) for w in words}
    if not blocks:
        selected = []
    else:
        blocks.sort(key=lambda b: -block_score(b))
        index = {text: i for i, (_, text) in enumerate(knowledge_paragraphs())}
        selected = []

        def add_block(block: str) -> None:
            """Add a block plus its file successor, if either fits. The
            knowledge base is written definition-then-elaboration, so the
            paragraph after a match often continues its answer (e.g. the
            hallucination definition is followed by why they happen) -- and a
            keyword search can't surface it, because its distinctive words
            aren't in the query."""
            if block not in selected and len(selected) < 4:
                selected.append(block)
            body = "\n".join(
                line for line in block.splitlines() if not line.startswith("#")
            ).strip()
            position = index.get(body)
            if (
                position is not None
                and position + 1 < len(knowledge_paragraphs())
                and len(selected) < 4
            ):
                headings, text = knowledge_paragraphs()[position + 1]
                successor = "\n".join(headings + [text])
                if successor not in selected:
                    selected.append(successor)

        add_block(blocks[0])
        # Then greedily add blocks that cover keywords the selection misses:
        # any new hit suffices -- how many new words matters less than which
        # ones, so among covering blocks the highest-scoring (most relevant)
        # wins, and once coverage saturates it falls back to score order.
        covered = set().union(*(block_hits(b) for b in selected))
        remaining = [b for b in blocks if b not in selected]
        while len(selected) < 4 and remaining:
            best = max(
                remaining,
                key=lambda b: (bool(block_hits(b) - covered), block_score(b)),
            )
            add_block(best)
            covered |= block_hits(best)
            remaining.remove(best)

    sentences = []
    for block in selected:
        if len(sentences) >= 8:
            break
        text = " ".join(
            line for line in block.splitlines()
            if line.strip() and not line.startswith("#")
        )
        sentences.extend(re.split(r"(?<=[.!?])\s+", text)[:2])
    answer = " ".join(sentences[:8]).strip()
    return {
        "thought": "The search results answer the question.",
        "action": "final_answer",
        "keywords": [],
        "answer": answer or "Sorry I cannot help with that.",
    }


@observe(type="llm", model=MODEL)
def call_llm(client: OpenAI, messages: list[dict]) -> dict:
    update_current_span(input=messages)  # rather than the default, which includes the client
    if API_STYLE == "chat":
        # Chat-completions style: works with Ollama, LM Studio, vLLM, and any
        # other OpenAI-compatible endpoint without the Responses API.
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "react_step", "schema": STEP_SCHEMA, "strict": True},
                },
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except OpenAIError:
            # Endpoints without strict JSON-schema support still get JSON out
            # of the prompt's own contract.
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                response_format={"type": "json_object"},
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        return json.loads(response.choices[0].message.content)

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "react_step",
                "schema": STEP_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    if response.status == "incomplete":
        raise RuntimeError(f"LLM response incomplete: {response.incomplete_details}")
    return json.loads(response.output_text)


def build_context(history: list[list[dict]]) -> list[dict]:
    """The last MAX_USER_TURNS turns, flattened into messages.

    Each turn starts with a user message, followed by that query's AI steps and
    tool observations, so dropping a turn drops all of them together.
    """
    return [message for turn in history[-MAX_USER_TURNS:] for message in turn]


def log(text: str) -> None:
    if VERBOSE:
        print(text, file=sys.stderr, flush=True)


def show(text: str) -> None:
    """Progress output on stdout, alongside the answers."""
    print(text, flush=True)


def truncate_values(value, limit: int = 50):
    """Shorten every string longer than `limit` to `limit` characters plus "…"."""
    if isinstance(value, str):
        return value[:limit] + "…" if len(value) > limit else value
    if isinstance(value, list):
        return [truncate_values(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: truncate_values(item, limit) for key, item in value.items()}
    return value


def to_json(value) -> str:
    # Like JS JSON.stringify: compact, and non-ASCII characters left as-is.
    return json.dumps(value, ensure_ascii=False)


@observe(type="agent")
def answer_query(client: OpenAI, history: list[list[dict]], query: str) -> str:
    update_current_span(input=query)
    turn = [{"role": "user", "content": query}]
    history.append(turn)
    del history[:-MAX_USER_TURNS]  # older turns can never be sent again

    try:
        for _ in range(MAX_STEPS):
            show("* Flibbertigibbeting")
            if BACKEND == "extractive":
                step = extractive_step(turn)
            else:
                step = call_llm(client, build_context(history))
            turn.append({"role": "assistant", "content": json.dumps(step)})
            log(f"[thought] {step['thought']}\n[action] {step['action']}")

            if step["action"] == "final_answer":
                return step["answer"]

            action = step["action"]
            show(f"> > tool: {action} {to_json({'keywords': step['keywords']})}")
            if action == "search_knowledge":
                observation = search_knowledge(step["keywords"])
            elif action == "list_sections" and "list_sections" in TOOLS:
                observation = list_sections()
            else:
                observation = f"Unknown tool: {action}. Use one of: {', '.join(sorted(TOOLS))}."
            # Display only: the model still gets the full observation.
            show(f"< < tool: {action} {to_json(truncate_values(observation))}")
            log(f"[keywords] {step['keywords']}\n[observation] {observation.splitlines()[0]}")
            turn.append({"role": "user", "content": f"Observation from {action}:\n{observation}"})
    except Exception:
        history.remove(turn)  # don't keep a half-finished turn in the context
        raise

    return f"Sorry, I couldn't reach an answer within {MAX_STEPS} steps."


def main() -> None:
    client = make_client()
    history: list[list[dict]] = []
    interactive = sys.stdin.isatty()

    while True:
        if interactive:
            print("> ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        query = line.strip()
        if not query:
            continue
        if query == "/exit":
            break
        try:
            answer = answer_query(client, history, query)
        except (OpenAIError, RuntimeError, json.JSONDecodeError) as e:
            print(f"error: {e}", file=sys.stderr, flush=True)
            continue
        show(f"< {answer}")


if __name__ == "__main__":
    main()
