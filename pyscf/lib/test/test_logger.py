#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import unittest
import numpy
from pyscf.lib import logger


class _Rec:
    def __init__(self, verbose):
        self.stdout = io.StringIO()
        self.verbose = verbose

    def lines(self):
        return self.stdout.getvalue().splitlines()


class KnownValues(unittest.TestCase):
    def test_progress_lines(self):
        # a line at once (after=0), then at every step percent, none at 100%, and the last one
        rec = _Rec(logger.INFO)
        prog = logger.Progress(rec, 'a loop', 10, 'things', after=0., step=20)
        self.assertEqual(list(prog.items(range(10))), list(range(10)))
        lines = rec.lines()
        self.assertEqual([x.split(':')[1].split('%')[0].strip() for x in lines[:-1]],
                         ['10', '20', '40', '60', '80'])
        self.assertTrue(lines[0].startswith('    a loop: 10% of the things after '))
        self.assertTrue(lines[-1].startswith('    a loop: done in '))
        self.assertEqual(prog.count, 10)

    def test_progress_silent(self):
        # nothing below Logger.info, and nothing (not the last line either) for a short loop
        rec = _Rec(logger.NOTE)
        prog = logger.Progress(rec, 'a loop', 10, after=0.)
        list(prog.items(range(10)))
        self.assertEqual(rec.lines(), [])
        rec = _Rec(logger.INFO)
        prog = logger.Progress(rec, 'a loop', 10)
        self.assertEqual(prog.after, logger.PROGRESS_AFTER)
        self.assertEqual(prog.step, logger.PROGRESS_STEP)
        list(prog.items(range(10)))
        self.assertEqual(rec.lines(), [])
        # a Logger in place of an object with stdout and verbose
        log = logger.Logger(rec.stdout, logger.INFO)
        prog = logger.Progress(log, 'a loop', 4, after=0., step=50)
        list(prog.items('abcd'))
        self.assertEqual(len(rec.lines()), 3)   # 25%, 50%, done

    def test_progress_blocks_and_ranges(self):
        rec = _Rec(logger.INFO)
        # the items of NumInt.block_loop are counted by their weights
        blocks = [(None, None, numpy.zeros(n), None) for n in (30, 30, 40)]
        prog = logger.Progress(rec, 'grid loop', 100, after=0., step=10)
        self.assertEqual([blk[2].size for blk in prog.blocks(iter(blocks))], [30, 30, 40])
        self.assertEqual(prog.count, 100)
        self.assertEqual([x.split(':')[1].strip()[:3] for x in rec.lines()], ['30%', '60%', 'don'])
        self.assertIn('of the grid after', rec.lines()[0])
        rec = _Rec(logger.INFO)
        prog = logger.Progress(rec, 'chunks', 7, 'densities', after=0., step=10)
        self.assertEqual(list(prog.ranges([(0, 3), (3, 6), (6, 7)])), [(0, 3), (3, 6), (6, 7)])
        self.assertEqual(prog.count, 7)
        self.assertEqual(len(rec.lines()), 3)

    def test_progress_step_without_count(self):
        # with Progress(...): a step reported when it ends, if it took `after` seconds
        rec = _Rec(logger.INFO)
        with logger.Progress(rec, 'a step', 0, after=0.):
            pass
        self.assertEqual(len(rec.lines()), 1)
        self.assertTrue(rec.lines()[0].startswith('    a step: done in '))
        rec = _Rec(logger.INFO)
        with logger.Progress(rec, 'a step', 0):
            pass
        self.assertEqual(rec.lines(), [])
        # not after an exception
        rec = _Rec(logger.INFO)
        with self.assertRaises(ValueError):
            with logger.Progress(rec, 'a step', 0, after=0.):
                raise ValueError
        self.assertEqual(rec.lines(), [])


if __name__ == "__main__":
    print("Full Tests for lib.logger")
    unittest.main()
