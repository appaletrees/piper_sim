from types import SimpleNamespace
import pytest

from piper_sim.profiling import PipelineProfiler


def test_nested_timings_are_not_double_counted(monkeypatch):
    profiler = PipelineProfiler(sample_interval_s=100, sample_gpu=False)
    with monkeypatch.context() as patch:
        ticks = iter([10., 12., 15., 20.])
        patch.setattr('piper_sim.profiling.time.perf_counter', lambda: next(ticks))
        with profiler.stage('parent'):
            with profiler.stage('child'): pass
    assert profiler.stages['parent']['inclusive_s'] == 10
    assert profiler.stages['parent']['exclusive_s'] == 7
    assert profiler.stages['child']['exclusive_s'] == 3
    profiler.close()


def test_wrapped_failure_preserves_error_and_is_restored():
    profiler = PipelineProfiler(sample_interval_s=100, sample_gpu=False)
    def failure(): raise ValueError('original failure')
    owner = SimpleNamespace(run=failure)
    profiler.wrap(owner, 'run', 'failed_stage')
    with pytest.raises(ValueError, match='original failure'): owner.run()
    result = profiler.close()
    assert owner.run is failure
    assert result['stages']['failed_stage']['calls'] == 1
    assert not profiler._thread.is_alive()
def test_process_vram_excludes_other_gpu_jobs_and_missing_values():
    from piper_sim.profiling import process_gpu_memory
    output = '''# gpu pid type fb ccpm command
      0 123 C 17000 0 other
      0 456 G 1009 0 python
      1 456 C - - python
      2 - - - - -
    '''
    assert process_gpu_memory(output, 456) == {'0': {'process_type': 'G', 'fb_reported_MB': 1009}}
    assert process_gpu_memory(output, 999) == {}
