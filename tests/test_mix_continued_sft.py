"""Checks for the continued-SFT sampler and local-template length accounting."""
import unittest

from scripts.mix_continued_sft import LengthChecker, ROOT, Sample


def row(text):
    return {'conversations': [{'role': 'user', 'content': text},
                              {'role': 'assistant', 'content': 'answer'}]}


class FakeChecker:
    def count(self, value):
        return len(value['conversations'][0]['content'])


class ContinuedSFTTests(unittest.TestCase):
    def test_complete_rows_only_and_deduplicated(self):
        sample = Sample(3, 42, FakeChecker(), 5)
        for text in ['a', 'a', 'bb', 'cccccc', 'ccc']:
            sample.add(row(text), {})
        self.assertEqual(len(sample.heap), 3)
        self.assertEqual(sample.stats['duplicate'], 1)
        self.assertEqual({e[2]['conversations'][0]['content'] for e in sample.heap},
                         {'a', 'bb', 'ccc'})

    def test_seed_reproducible(self):
        samples = [Sample(5, 17, FakeChecker(), 100) for _ in range(2)]
        for sample in samples:
            for n in range(100):
                sample.add(row(str(n)), {})
        self.assertEqual(samples[0].take(5), samples[1].take(5))

    def test_reasoning_count_matches_hf_reference(self):
        # Independently checked using AutoTokenizer in the installed Jarvis env.
        checker = LengthChecker(ROOT / 'model')
        thought = {'conversations': [
            {'role': 'user', 'content': '计算 3 + 4'},
            {'role': 'assistant', 'content': '7',
             'reasoning_content': '把 3 和 4 相加，得到 7。'}]}
        ordinary = {'conversations': [{'role': 'user', 'content': 'Hello'},
                                       {'role': 'assistant', 'content': 'Hi!'}]}
        self.assertEqual(checker.count(thought), 35)
        self.assertEqual(checker.count(ordinary), 23)
        self.assertEqual(checker.count_batch([thought, ordinary]), [35, 23])


if __name__ == '__main__':
    unittest.main()
