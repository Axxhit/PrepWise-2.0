import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def message_to_text(message: dict) -> str:
    """Convert a message dict to plain text for token counting."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    # tool parts — stringify them
    return str(content)

def trim_messages(
    messages: list[dict],
    max_tokens: int = 6000,
    system_prompt: str = ""
) -> list[dict]:
    """
    Keep system prompt + as many recent messages as fit within max_tokens.
    Always keeps the first (user) message and trims from the middle.
    """
    system_tokens = count_tokens(system_prompt)
    budget = max_tokens - system_tokens

    # always keep first message (original user task)
    first = messages[:1]
    rest = messages[1:]

    first_tokens = count_tokens(message_to_text(first[0]))
    budget -= first_tokens

    # walk from most recent → oldest, collect until budget exhausted
    kept = []
    for msg in reversed(rest):
        tokens = count_tokens(message_to_text(msg))
        if budget - tokens < 0:
            print(f"  [context trimmer] dropped message ({tokens} tokens, budget remaining: {budget})")
            break
        kept.insert(0, msg)
        budget -= tokens

    trimmed = first + kept
    return trimmed


def log_token_usage(messages: list[dict], label: str = ""):
    total = sum(count_tokens(message_to_text(m)) for m in messages)
    print(f"  [token usage] {label}: {total} tokens across {len(messages)} messages")
    for i, m in enumerate(messages):
        t = count_tokens(message_to_text(m))
        role = m.get("role", "?")
        print(f"    msg[{i}] role={role} tokens={t}")
