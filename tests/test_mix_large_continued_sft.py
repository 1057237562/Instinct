import json
import random
import sqlite3
import unittest
from collections import Counter

from scripts.mix_large_continued_sft import (
    classify_replay, fingerprints, has_reasoning, select_balanced_replay, select_source,
)


class Checker:
    def count_batch(self, rows):
        return [len(row['conversations'][-1]['content']) for row in rows]


class LargeMixTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.executescript('''
            CREATE TABLE candidates (source TEXT,priority INTEGER,fp BLOB,body TEXT,meta TEXT,
                                      question TEXT,language TEXT,tokens INTEGER);
            CREATE TABLE heldout (question TEXT PRIMARY KEY);
            CREATE TABLE selected (candidate_id INTEGER PRIMARY KEY,shuffle INTEGER,tokens INTEGER,bytes INTEGER);
        ''')

    def tearDown(self):
        self.db.close()

    def insert(self, source, index, thought=False, question=None, content='answer'):
        row = {'conversations': [{'role': 'user', 'content': f'{source}-{index}'},
                                  {'role': 'assistant', 'content': content}]}
        if thought:
            row['conversations'][-1]['reasoning_content'] = 'A complete derivation.'
        fp, _ = fingerprints(row)
        self.db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,NULL)',
                        (source, index, fp, json.dumps(row), '{}', question, 'python',))

    def test_only_nonempty_reasoning_counts(self):
        for content, expected in [('answer', False), ('<think>\n </think>answer', False),
                                  ('<think>Step one.</think>answer', True)]:
            self.assertEqual(has_reasoning({'conversations': [{'role': 'assistant', 'content': content}]}), expected)

    def test_reasoning_surplus_and_complete_pair_byte_boundary(self):
        for i in range(100):
            self.insert('t2t_cot', i, thought=True)
            self.insert('t2t_final', i)
        stats = select_balanced_replay(self.db, 6000, 10, 2, Checker(), 16384, set(), random.Random(42))
        rows = 10 + sum(s['selected'] for s in stats.values())
        reasoning = 2 + stats['t2t_cot']['selected']
        self.assertGreater(reasoning / rows, .5)
        self.assertLess(abs(sum(s['bytes'] for s in stats.values()) - 6000), 400)

    def test_classification_and_empty_think(self):
        self.insert('t2t_replay', 1, thought=True)
        self.insert('t2t_replay', 2, content='<think>\n\n</think>answer')
        classify_replay(self.db)
        self.assertEqual(dict(self.db.execute('SELECT source,count(*) FROM candidates GROUP BY source')),
                         {'t2t_cot': 1, 't2t_final': 1})

    def test_heldout_length_and_question_cap(self):
        for i in range(5):
            self.insert('competitive_final', i, question='train')
        self.insert('competitive_final', 6, question='test')
        self.insert('competitive_final', 7, question='long', content='x' * 30)
        self.db.execute("INSERT INTO heldout VALUES ('test')")
        stats = select_source(self.db, 'competitive_final', 100000, Checker(), 10,
                              set(), Counter(), 3, random.Random(42))
        self.assertEqual(stats['selected'], 3)
        self.assertEqual(stats['over_limit'], 1)
        self.assertEqual(stats['question_cap'], 2)


if __name__ == '__main__':
    unittest.main()
