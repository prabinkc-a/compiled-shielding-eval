"""Open grammar adapter; candidate corpus is explicit and independently owned.

The sequence policy scores lexical tokens in the inherited trie. Supplying a
finite corpus limits what can be discovered; it does not learn Python syntax.
"""
from phase3.python_grammar import PythonGrammar as BaseGrammar


class PythonGrammar(BaseGrammar):
    def __init__(self, tokenizer, suite, backend, sources):
        self.tokenizer, self.backend, self.suite = tokenizer, backend, suite
        self.entries = [dict(source=source) for source in sources]
        self.encoded = {source:tokenizer.encode(source,add_special_tokens=False)+[tokenizer.eos_token_id]
                        for source in sources}
        for source, ids in self.encoded.items():
            if tokenizer.decode(ids[:-1]) != source:
                raise ValueError('Tokenizer must round-trip the supplied grammar')
        self.max_tokens = max(map(len,self.encoded.values()))
        self.separator = tokenizer.convert_tokens_to_ids('<CODE>')
        self.refresh()
