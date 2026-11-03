"""The optional solver warning flag must not gate environment startup."""

from enum import IntFlag
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from supermimic_lab.backends.mjlab.compat import quiet_solver_warnings


@pytest.mark.parametrize("has_enum,has_flag", [(False, False), (False, True), (True, False)])
def test_warning_suppression_is_optional(monkeypatch, has_enum, has_flag):
  warp = SimpleNamespace()
  if has_enum:
    warp.OverflowType = SimpleNamespace(ITERATIONS=1, LS_ITERATIONS=2)
  monkeypatch.setitem(sys.modules, "mujoco_warp", warp)
  opt = SimpleNamespace(warn_overflow=7) if has_flag else SimpleNamespace()
  sim = SimpleNamespace(wp_model=SimpleNamespace(opt=opt), create_graph=Mock())
  quiet_solver_warnings(SimpleNamespace(sim=sim), None)
  sim.create_graph.assert_not_called()
  if has_flag:
    assert opt.warn_overflow == 7


def test_only_iteration_warning_flags_are_cleared(monkeypatch):
  class Overflow(IntFlag):
    ITERATIONS = 1
    LS_ITERATIONS = 2
    CONTACTS = 4

  monkeypatch.setitem(sys.modules, "mujoco_warp", SimpleNamespace(OverflowType=Overflow))
  opt = SimpleNamespace(warn_overflow=7)
  sim = SimpleNamespace(wp_model=SimpleNamespace(opt=opt), create_graph=Mock())
  quiet_solver_warnings(SimpleNamespace(sim=sim), None)
  assert opt.warn_overflow == Overflow.CONTACTS
  sim.create_graph.assert_called_once_with()
