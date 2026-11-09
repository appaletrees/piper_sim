import pytest
import yaml

from piper_sim.config import ROOT
from piper_sim.experts.rigid_table import DEFAULT_RIGID_ORDER, validate_sort_order


def scene_specs():
    return yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())['objects']


def test_default_order_remains_unchanged_and_candidate_respects_supports():
    specs = scene_specs()
    assert validate_sort_order(None, specs, list(specs)) == DEFAULT_RIGID_ORDER
    order = ['chopstick_a','chopstick_b','pen','charger','spoon','bowl','plate','side_dish']
    assert validate_sort_order(order, specs, list(specs)) == tuple(order)


def test_order_rejects_missing_duplicate_unknown_and_reversed_support():
    specs = scene_specs();order = list(DEFAULT_RIGID_ORDER)
    bad_orders = [order[:-1], order+[order[-1]], order+['cup_00'], order+['unknown'], 'spoon,bowl,plate']
    reversed_support = order.copy();reversed_support[2:5] = ['plate','bowl','spoon'];bad_orders.append(reversed_support)
    for value in bad_orders:
        with pytest.raises(ValueError, match='rigid_sort_order'): validate_sort_order(value, specs, list(specs))


def test_order_filters_inactive_targets_without_adding_cups_or_containers():
    specs = scene_specs()
    active = [n for n in specs if n != 'chopstick_b']
    result = validate_sort_order(list(DEFAULT_RIGID_ORDER), specs, active)
    assert result == tuple(n for n in DEFAULT_RIGID_ORDER if n != 'chopstick_b')
    assert 'basket' not in result and 'cup_00' not in result
