# -*- coding: utf-8 -*-
"""Script for memory and execution time profiling."""
import tracemalloc
from cProfile import Profile
from pstats import Stats

from hku_diabetes.analytics import Analyser
from hku_diabetes.config import TestConfig
from hku_diabetes.importer import import_all
from hku_diabetes.plot import plot_all

TOP_STATS = 20


class ProfilingConfig(TestConfig):
    test_samples = 100
    plot_samples = 100


def test():
    """Testing sequence for profiling"""
    analyser = Analyser(config=ProfilingConfig)
    data = import_all(config=ProfilingConfig)
    analyser.load()
    plot_all(analyser)


def main():
    """Main sequence"""
    analyser = Analyser(config=ProfilingConfig)
    data = import_all(config=ProfilingConfig)
    analyser.regression(data)
    del analyser
    del data    
    profiler = Profile()
    tracemalloc.start(10)
    time1 = tracemalloc.take_snapshot()
    profiler.runcall(test)
    time2 = tracemalloc.take_snapshot()
    time_stats = Stats(profiler)
    time_stats.strip_dirs()
    time_stats.sort_stats('cumulative')
    print("\n===Time Profiler Stats===\n")
    time_stats.print_stats(TOP_STATS)
    print("\n===Time Profiler Callers===\n")
    time_stats.print_callers(TOP_STATS)
    memory_stats = time2.compare_to(time1, 'lineno')
    print("\n===Memory Profiler Callers===\n")    
    for stat in memory_stats[:3]:
        print(stat)  
    print("\n===Top Memory Consumer===\n")              
    top = memory_stats[0]
    print('\n'.join(top.traceback.format()))


if __name__ == '__main__':
    main()
