#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''
Logging system

Log level
---------

======= ======
Level   number
------- ------
DEBUG4  9
DEBUG3  8
DEBUG2  7
DEBUG1  6
DEBUG   5
INFO    4
NOTE    3
WARN    2
ERROR   1
QUIET   0
======= ======

Large value means more noise in the output file.

.. note::
    Error and warning messages are written to stderr.

Each Logger object has its own output destination and verbose level.  So
multiple Logger objects can be created to manage the message system without
affecting each other.
The methods provided by Logger class has the direct connection to the log level.
E.g.  :func:`info` print messages if the verbose level >= 4 (INFO):

>>> import sys
>>> from pyscf import lib
>>> log = lib.logger.Logger(sys.stdout, 4)
>>> log.info('info level')
info level
>>> log.verbose = 3
>>> log.info('info level')
>>> log.note('note level')
note level


timer
-----
Logger object provides timer method for timing.  Set :attr:`TIMER_LEVEL` to
control at which level the timing information should be output.  It is 5
(DEBUG) by default.

>>> import sys, time
>>> from pyscf import lib
>>> log = lib.logger.Logger(sys.stdout, 4)
>>> t0 = logger.process_clock()
>>> log.timer('test', t0)
>>> lib.logger.TIMER_LEVEL = 4
>>> log.timer('test', t0)
    CPU time for test      0.00 sec

'''

import sys
import time

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter

from pyscf.lib import parameters as param
import pyscf.__config__

DEBUG4 = param.VERBOSE_DEBUG + 4
DEBUG3 = param.VERBOSE_DEBUG + 3
DEBUG2 = param.VERBOSE_DEBUG + 2
DEBUG1 = param.VERBOSE_DEBUG + 1
DEBUG  = param.VERBOSE_DEBUG
INFO   = param.VERBOSE_INFO
NOTE   = param.VERBOSE_NOTICE
NOTICE = NOTE
WARN   = param.VERBOSE_WARN
WARNING = WARN
ERR    = param.VERBOSE_ERR
ERROR  = ERR
QUIET  = param.VERBOSE_QUIET
CRIT   = param.VERBOSE_CRIT
ALERT  = param.VERBOSE_ALERT
PANIC  = param.VERBOSE_PANIC

TIMER_LEVEL  = getattr(pyscf.__config__, 'TIMER_LEVEL', DEBUG)

sys.verbose = NOTE

def flush(rec, msg, *args):
    rec.stdout.write(msg%args)
    rec.stdout.write('\n')
    rec.stdout.flush()

def log(rec, msg, *args):
    if rec.verbose > QUIET:
        flush(rec, msg, *args)

def error(rec, msg, *args):
    if rec.verbose >= ERROR:
        flush(rec, '\nERROR: '+msg+'\n', *args)
    sys.stderr.write('ERROR: ' + (msg%args) + '\n')

def warn(rec, msg, *args):
    if rec.verbose >= WARN:
        flush(rec, '\nWARN: '+msg+'\n', *args)
        if rec.stdout is not sys.stdout:
            sys.stderr.write('WARN: ' + (msg%args) + '\n')

def info(rec, msg, *args):
    if rec.verbose >= INFO:
        flush(rec, msg, *args)

def note(rec, msg, *args):
    if rec.verbose >= NOTICE:
        flush(rec, msg, *args)

def debug(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)

def debug1(rec, msg, *args):
    if rec.verbose >= DEBUG1:
        flush(rec, msg, *args)

def debug2(rec, msg, *args):
    if rec.verbose >= DEBUG2:
        flush(rec, msg, *args)

def debug3(rec, msg, *args):
    if rec.verbose >= DEBUG3:
        flush(rec, msg, *args)

def debug4(rec, msg, *args):
    if rec.verbose >= DEBUG4:
        flush(rec, msg, *args)

def stdout(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)
    sys.stdout.write('>>> %s\n' % msg)

def timer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec' % (msg, rec._t0-cpu0))
        return rec._t0

def timer_debug1(rec, msg, cpu0=None, wall0=None):
    if rec.verbose >= DEBUG1:
        return timer(rec, msg, cpu0, wall0)
    elif wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        return rec._t0

def init_timer():
    return process_clock(), perf_counter()

class Logger:
    '''
    Attributes:
        stdout : file object or sys.stdout
            The file to dump output message.
        verbose : int
            Large value means more noise in the output file.
    '''
    def __init__(self, stdout=sys.stdout, verbose=NOTE):
        self.stdout = stdout
        self.verbose = verbose
        self._t0 = process_clock()
        self._w0 = perf_counter()

    def init_timer(self):
        return process_clock(), perf_counter()

    log = log
    error = error
    warn = warn
    note = note
    info = info
    debug  = debug
    debug1 = debug1
    debug2 = debug2
    debug3 = debug3
    debug4 = debug4
    timer = timer
    timer_debug1 = timer_debug1

def new_logger(rec=None, verbose=None):
    '''Create and return a :class:`Logger` object

    Args:
        rec : An object which carries the attributes stdout and verbose

        verbose : a Logger object, or integer or None
            The verbose level. If verbose is a Logger object, the Logger
            object is returned. If verbose is not specified (None),
            rec.verbose will be used in the new Logger object.
    '''
    if isinstance(verbose, Logger):
        log = verbose
    elif isinstance(verbose, int):
        from pyscf.lib.misc import StreamObject
        if getattr(rec, 'stdout', None):
            log = Logger(rec.stdout, verbose)
        else:
            log = Logger(StreamObject.stdout, verbose)
    else:
        log = Logger(rec.stdout, rec.verbose)
    return log

# Progress lines of long loops (Progress, and the integral drivers' scf._vhf.oe_progress): the first once a
# loop has run for PROGRESS_AFTER seconds, then one at every PROGRESS_STEP percent, all with Logger.info
PROGRESS_AFTER = getattr(pyscf.__config__, 'PROGRESS_AFTER', 30.)
PROGRESS_STEP = getattr(pyscf.__config__, 'PROGRESS_STEP', 10)

class Progress:
    '''Progress lines of a loop that may take long, in the form of the integral drivers' lines:

        XC second derivatives, pair terms: 40% of the grid after 302 s
        XC second derivatives, pair terms: done in 755 s

    add(n) counts n more of the `total` units (`unit`: what they are, for the line) as done. Nothing is
    written before the loop has run for PROGRESS_AFTER seconds; then a line at once, one at every
    PROGRESS_STEP percent, and done() writes the last. All with Logger.info: nothing below that level.

    >>> prog = logger.Progress(mf, 'XC second derivatives, pair terms', grids.weights.size)
    >>> for ao, mask, weight, coords in prog.blocks(ni.block_loop(mol, grids, nao, 2, max_memory)):
    ...     ...

    Args:
        rec : a Logger, or an object with the attributes stdout and verbose
        label : what the loop computes
        total : the units of the whole loop
    '''
    def __init__(self, rec, label, total, unit='grid', after=None, step=None):
        self.log = new_logger(rec, rec if isinstance(rec, Logger) else None)
        self.label = label
        self.total = total
        self.unit = unit
        self.after = PROGRESS_AFTER if after is None else after
        self.step = PROGRESS_STEP if step is None else step
        self.count = 0
        self.printed = False
        self._next = self.step
        self._t0 = perf_counter()

    def add(self, n=1):
        '''n more units are done'''
        self.count += n
        if self.log.verbose < INFO or self.total <= 0:
            return
        dt = perf_counter() - self._t0
        if dt < self.after:
            return
        pct = 100. * self.count / self.total
        if pct < 100 and (pct >= self._next or not self.printed):
            self.log.info('    %s: %d%% of the %s after %d s', self.label, pct, self.unit, dt)
            self.printed = True
            self._next = (int(pct) // self.step + 1) * self.step

    def done(self, always=False):
        '''The loop has ended: the last line, if it wrote any (always: or if it ran for `after` seconds)'''
        dt = perf_counter() - self._t0
        if self.printed or (always and dt >= self.after):
            self.log.info('    %s: done in %d s', self.label, dt)
            self.printed = False

    # with Progress(log, label, 0): a step without a count, reported when it ends if it took `after` seconds
    def __enter__(self):
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.done(always=True)

    def blocks(self, block_loop):
        '''The items (ao, mask, weight, coords) of NumInt.block_loop, each counted with its grid points when
        the loop asks for the next one'''
        for blk in block_loop:
            n = blk[2].size
            yield blk
            blk = None
            self.add(n)
        self.done()

    def ranges(self, ranges):
        '''The pairs (p0, p1) of an iterable of ranges, each counted with its p1 - p0 units when the loop
        asks for the next one'''
        for p0, p1 in ranges:
            yield p0, p1
            self.add(p1 - p0)
        self.done()

    def items(self, iterable):
        '''The items of an iterable, each counted as one unit when the loop asks for the next one'''
        for x in iterable:
            yield x
            self.add(1)
        self.done()
