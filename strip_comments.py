import pathlib, tokenize, io, sys

root = pathlib.Path(__file__).parent
for path in root.rglob('*.py'):
    if '.venv' in path.parts or path.name == 'strip_comments.py':
        continue
    src = path.read_text(encoding='utf-8')
    tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    result_tokens = []
    prev_toktype = tokenize.INDENT
    for tok in tokens:
        token_type = tok.type
        token_string = tok.string
        if token_type == tokenize.COMMENT:
            continue  # drop comments
        # drop docstrings (a STRING right after INDENT or at file start)
        if token_type == tokenize.STRING and prev_toktype == tokenize.INDENT:
            prev_toktype = token_type
            continue
        result_tokens.append((token_type, token_string))
        prev_toktype = token_type

    # Reconstruct source preserving newlines
    new_src = tokenize.untokenize(result_tokens)
    path.write_text(new_src, encoding='utf-8')
    print(f"Stripped comments from {path.relative_to(root)}")

print("Done.")
