from src.models.tokenizer import Tokenizer


def test_encode_decode_roundtrip():
    tok = Tokenizer(vocab_size=128)
    ids = tok.encode("hello world", add_bos=True, add_eos=True)
    assert isinstance(ids, list)
    assert ids[0] == 0  # BOS
    assert ids[-1] == 1  # EOS
    assert len(ids) == 4  # BOS + 2 words + EOS


def test_truncate():
    tok = Tokenizer(vocab_size=128)
    long_text = " ".join(["word"] * 2000)
    ids = tok.encode(long_text, add_bos=False, add_eos=False)
    truncated = tok.truncate(ids, max_length=1000)
    assert len(truncated) == 1000
