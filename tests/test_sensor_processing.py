"""Exact sensor semantics and bounded worker/cache failure recovery."""
import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from PIL import Image, ImageFilter
import pytest
import yaml

from piper_sim import randomization
from piper_sim.config import ROOT, load
from piper_sim.sensor_processing import SensorImageProcessor


def images():
    levels = np.arange(256, dtype=np.uint8).reshape(16,16)
    rgb = np.stack([levels, levels[::-1], np.roll(levels,3,axis=1)], axis=-1)
    result = {'overhead': rgb, 'left_wrist': rgb[::-1], 'right_wrist': rgb[:,::-1]}
    for value in result.values(): value.flags.writeable = False
    return result


def parameters(gamma=1.17, noise=1.4, blur=0., seed=53):
    return {'seed':seed, 'cameras':{name: {'gamma':gamma, 'gain':1.09,
        'white_balance':[.92,1.03,1.14], 'noise_std_uint8':noise, 'blur_sigma_px':blur}
        for name in ('overhead','left_wrist','right_wrist')}}


def original_expression(pixels, params, tick):
    result = {}
    for name, raw in pixels.items():
        p = params['cameras'][name]
        if p['blur_sigma_px']:
            raw = np.asarray(Image.fromarray(raw).filter(ImageFilter.GaussianBlur(p['blur_sigma_px'])))
        rgb = 255.*(raw.astype(np.float32)/255.)**p['gamma']
        rgb *= p['gain']*np.asarray(p['white_balance'], np.float32)
        if p['noise_std_uint8']:
            rng = np.random.default_rng(randomization.stream_seed(params['seed'], f'sensor/{name}/{tick}'))
            rgb += rng.normal(0,p['noise_std_uint8'],rgb.shape).astype(np.float32)
        result[name] = np.clip(np.rint(rgb),0,255).astype(np.uint8)
    return result


@pytest.mark.parametrize('workers',[2,3])
@pytest.mark.parametrize('gamma',[.5,1.17,2.2])
@pytest.mark.parametrize('noise,blur',[(0.,0.),(1.7,.6)])
def test_full_uint8_range_and_episode_tick_streams_match_original(workers,gamma,noise,blur):
    processor = SensorImageProcessor(workers)
    pixels = images(); unchanged = {k:v.copy() for k,v in pixels.items()}
    try:
        for seed,tick in ((53,19),(53,20),(97,19),(53,19)):
            params = parameters(gamma,noise,blur,seed); before = copy.deepcopy(params)
            actual = processor(pixels,params,tick)
            expected = original_expression(pixels,params,tick)
            assert list(actual) == list(pixels) and params == before
            for name in pixels:
                np.testing.assert_array_equal(actual[name],expected[name])
                np.testing.assert_array_equal(pixels[name],unchanged[name])
    finally: processor.close()


def test_float_fallback_and_disabled_processing_preserve_existing_contract():
    processor = SensorImageProcessor(3); pixels = images()
    try:
        assert processor(pixels,None,0) is pixels and processor._pool is None
        assert processor({},parameters(),0) == {} and processor._pool is None
        floating = {k:v.astype(np.float32)+.25 for k,v in pixels.items()}
        expected = original_expression(floating,parameters(),17)
        actual = processor(floating,parameters(),17)
        for name in pixels: np.testing.assert_array_equal(actual[name],expected[name])
    finally: processor.close()


@pytest.mark.parametrize('workers',[True,0,4,1.5,'3',None])
def test_invalid_worker_budget_rejected(workers):
    with pytest.raises(ValueError,match='integer from 1 to 3'): SensorImageProcessor(workers)


def test_failed_camera_drains_other_workers_before_raising_and_can_retry(monkeypatch):
    processor = SensorImageProcessor(3)
    original = randomization._sensor_image
    entered = {name:Event() for name in ('left_wrist','right_wrist')}
    release = Event(); finished = {name:Event() for name in entered}
    def broken(name,*args):
        if name == 'overhead': raise ValueError('camera failure')
        entered[name].set()
        assert release.wait(10)
        try: return original(name,*args)
        finally: finished[name].set()
    monkeypatch.setattr(randomization,'_sensor_image',broken)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(processor,images(),parameters(),1)
        try:
            assert all(event.wait(10) for event in entered.values())
            assert not future.done()
        finally: release.set()
        with pytest.raises(ValueError,match='camera failure'): future.result(timeout=10)
    assert all(event.is_set() for event in finished.values())
    monkeypatch.setattr(randomization,'_sensor_image',original)
    expected = original_expression(images(),parameters(),2)
    actual = processor(images(),parameters(),2)
    for name in actual: np.testing.assert_array_equal(actual[name],expected[name])
    pool = processor._pool
    processor.close(); processor.close()
    assert not any(thread.is_alive() for thread in pool._threads)
    with pytest.raises(RuntimeError,match='closed'): processor(images(),parameters(),3)


def test_partial_submission_failure_also_waits_for_borrowed_pixels(monkeypatch):
    entered, release, finished = Event(), Event(), Event()
    def held(name,pixels,params,tick):
        entered.set()
        assert release.wait(10)
        finished.set()
        return pixels.copy()
    monkeypatch.setattr(randomization,'_sensor_image',held)
    with ThreadPoolExecutor(max_workers=1) as worker, ThreadPoolExecutor(max_workers=1) as caller:
        class RejectSecond:
            calls = 0
            def submit(self,*args):
                self.calls += 1
                if self.calls == 2: raise RuntimeError('submit failed')
                return worker.submit(*args)
        future = caller.submit(randomization.sensor_images, images(),parameters(),0,executor=RejectSecond())
        try:
            assert entered.wait(10) and not future.done()
        finally: release.set()
        with pytest.raises(RuntimeError,match='submit failed'): future.result(timeout=10)
        assert finished.is_set()


@pytest.mark.parametrize('workers',[1,3])
def test_mujoco_failed_processing_does_not_cache_raw_pixels_and_reset_refreshes(monkeypatch,workers):
    pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    cfg = load(ROOT/'configs/clean_table_tcp.yaml')
    cfg['runtime']['sensor_postprocess_workers'] = workers
    cfg['task_scene'] = yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    cfg['task_scene']['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
    backend = MujocoBackend(cfg,cameras=True)
    backend.reset(seed=53,randomize=True)
    assert backend.task.domain_parameters
    # Real MuJoCo state/reset/metadata; small known camera buffers avoid GL here.
    raw = images(); captures = []
    def render(name,*size): captures.append(name); return raw[name].copy()
    monkeypatch.setattr(backend,'_render',render)
    original = randomization._sensor_image
    def broken(name,*args):
        if name == 'overhead': raise RuntimeError('sensor failed')
        return original(name,*args)
    try:
        monkeypatch.setattr(randomization,'_sensor_image',broken)
        with pytest.raises(RuntimeError,match='sensor failed'): backend.read()
        assert backend._images is None
        monkeypatch.setattr(randomization,'_sensor_image',original)
        first = backend.read(); before = len(captures)
        first.images['overhead'][:] = 0
        again = backend.read(); assert len(captures) == before
        expected = original_expression(raw,backend.task.domain_parameters,0)
        for name in raw: np.testing.assert_array_equal(again.images[name],expected[name])
        backend.step()
        stepped = backend.read(); assert len(captures) == before+3
        expected = original_expression(raw,backend.task.domain_parameters,1)
        for name in raw:
            np.testing.assert_array_equal(stepped.images[name],expected[name])
            assert stepped.image_timestamps[name] == stepped.sim_time
        backend.reset(seed=97,randomize=True)
        reset = backend.read(); assert reset.sim_time == 0
        expected = original_expression(raw,backend.task.domain_parameters,0)
        for name in raw: np.testing.assert_array_equal(reset.images[name],expected[name])
    finally: backend.close()


def test_worker_candidate_changes_no_physical_or_optical_configuration():
    from piper_sim.mujoco_asset import configuration_fingerprint
    serial = load(ROOT/'configs/clean_table_tcp.yaml')
    parallel = load(ROOT/'configs/clean_table_tcp_sensor_workers.yaml')
    assert configuration_fingerprint(serial) == configuration_fingerprint(parallel)
    assert parallel['runtime'].pop('sensor_postprocess_workers') == 3
    serial.pop('_config_path'); parallel.pop('_config_path')
    assert serial == parallel
