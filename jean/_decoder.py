from janne.interfaces import IDecoder
from PyEDMReader import (
        EventMode,
        JaEDMReader,
        JaEDMReaderConfig,
        )

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type
from os import path
import csv
import warnings

import numpy as np
import numpy.typing as npt

def gaussian(x, mu, sig):
  return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

Vector = Tuple[int, int, int]

JUNO_RADIUS=19365
N_LPMT=17612
N_SPMT=25600
SPMT_OFFSET=300000

# NEVT / seconds = 0.0127404


@dataclass
class JeanDecoderConfig:
  filepath: str
  mode: EventMode
  lpmt_data_path: str
  spmt_data_path: str

def _read_lpmt_csv(lpmt_data_path: str) -> npt.NDArray[np.float64]:
  if not (path.exists(lpmt_data_path) and path.isfile(lpmt_data_path)):
    raise FileNotFoundError("Lpmt data path is not an existing file")

  lpmts = []
  with open(lpmt_data_path, newline="", encoding="utf-8") as lpmt_file:
    lpmts = [[lpmt[1], lpmt[2], lpmt[3]] for lpmt in  csv.reader(lpmt_file, delimiter=" ")]

  lpmts = np.array(lpmts, dtype=np.float64)

  assert lpmts.shape[0] == N_LPMT

  return lpmts

def _read_spmt_csv(spmt_data_path: str) -> npt.NDArray[np.float64]:
  if not (path.exists(spmt_data_path) and path.isfile(spmt_data_path)):
    raise FileNotFoundError("Lpmt data path is not an existing file")

  spmts = []
  with open(spmt_data_path, newline="", encoding="utf-8") as spmt_file:
    spmts = [[np.sin(float(spmt[1])) * np.cos(float(spmt[2])),
              np.sin(float(spmt[1])) * np.sin(float(spmt[2])),
              np.cos(float(spmt[1]))] for spmt in  csv.reader(spmt_file, delimiter=" ")]

  spmts = np.array(spmts, dtype=np.float64) * JUNO_RADIUS

  assert spmts.shape[0] == N_SPMT

  return spmts

class JeanDecoder(IDecoder):
  """The JEAN decoder. Takes data from a PyEDMReader and replace the pmtID
  by their X, Y, Z position.

  Use the values JUNO_RADIUS, N_LPMT, N_SPMT, SPMT_OFFSET defined in jean/_decoder.py
  to compute X,Y,Z
  """

  def __init__(self, config: Optional[JeanDecoderConfig] = None):
    self._config: Optional[JeanDecoderConfig] = config
    self.lpmt_data: Optional[npt.NDArray[np.float64]] = None
    self.spmt_data: Optional[npt.NDArray[np.float64]] = None

    self._decoder = JaEDMReader()

    if self._config:
      self.initialize(self._config)


  def initialize(self, config: JeanDecoderConfig):
    self._config = config
    self._decoder.initialize(JaEDMReaderConfig(
      filepath = self._config.filepath,
      mode = self._config.mode
      ))

    if self.lpmt_data is None:
      self.lpmt_data = _read_lpmt_csv(self._config.lpmt_data_path)

    if self.spmt_data is None:
      self.spmt_data = _read_spmt_csv(self._config.spmt_data_path)

    self._all_data = np.concatenate((self.lpmt_data, self.spmt_data), axis = 0)

  def __next__(self):
    if self._decoder is None or self._config is None:
      raise RuntimeError("JeanDecoder not initialized. Please initialize first")

    signal, truth = next(self._decoder)

    if self._config.mode == EventMode.DETSIM:
      # Grouping PMT
      acc_l = np.zeros((N_LPMT, 2))
      acc_s = np.zeros((N_SPMT, 2))

      for hit in signal:
        if hit[0] < SPMT_OFFSET:
          if hit[2] < acc_l[hit[0], 1] or acc_l[hit[0], 0] == 0:
            acc_l[hit[0], 1] = hit[2]

          acc_l[hit[0], 0] += hit[1]
        else:
          s_id = hit[0] - SPMT_OFFSET
          if hit[2] < acc_s[s_id, 1] or acc_s[s_id, 0] == 0:
            acc_s[hit[0], 1] = hit[2]

          acc_s[hit[0], 0] += hit[1]


      s_l = np.concatenate((np.arange(0, N_LPMT).reshape((N_LPMT, 1)), acc_l), axis=-1)
      s_l = s_l[s_l[:, 1] > 0]

      s_s = np.concatenate((np.arange(0, N_SPMT).reshape((N_SPMT, 1)), acc_s), axis=-1)
      s_s = s_s[s_s[:, 1] > 0]

      s = np.concatenate((s_l, s_s), axis=0)
    else:
      s = signal


    return (
        np.concatenate((
          self._all_data[s[:, 0].astype(int) % (SPMT_OFFSET - N_LPMT)],
          s[:,0:-1]), axis=-1),
        truth)

  def config(self):
    return self._config

@dataclass
class GaussianSelectorConfig:
  sub_config: Any
  mean: float
  variance: float
  seed: int

class GaussianSelector(IDecoder):
  """A decoder that will filter based on their true energy. The criterion is a gaussian
  centered on mean (MeV) with a scale of variance (MeV)
  """
  def __init__(self, config: Optional[GaussianSelectorConfig] = None, source_decoder: Optional[Type[IDecoder]] = None):
    if source_decoder is None:
      raise ValueError("This special decoder need a supplementary argument, the class of the target decoder")
    self._decoder = source_decoder()

    self._mean = -1
    self._variance = -1

    self._config = config

    self._random = np.random.default_rng(0)

    self._has_warn = False

    if config is not None:
      self.initialize(config)

  def initialize(self, config: GaussianSelectorConfig):
    self._decoder.initialize(config.sub_config)

    self._mean = config.mean
    self._variance = config.variance

    self._random = np.random.default_rng(config.seed)

  def __next__(self):
    data, truth = next(self._decoder)

    if truth is None:
      if not self._has_warn:
        warnings.warn("Gaussian selector is only configured to work with truth value, which is missing."
                      "Ignoring the selection. This warning will only trigger once")
        self._has_warn = True
      return (data, truth)

    while self._random.uniform(0.0, 1.0) > gaussian(truth[0], self._mean, self._variance):
      data, truth = next(self._decoder)

      if truth is None:
        if not self._has_warn:
          warnings.warn("Gaussian selector is only configured to work with truth value, which is missing."
                        "Ignoring the selection. This warning will only trigger once")
          self._has_warn = True
        return (data, truth)


    return (data, truth)

  def config(self):
    return self._config
