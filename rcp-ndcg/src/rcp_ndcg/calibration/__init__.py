"""Calibration: judgements in, calibrated abilities and 2PL item parameters out.

* :func:`read_judgements` -- the judgements of one or more judgement stores;
* :func:`count_gains` -- the Count-nDCG gains of the rubric windows (``evaluate(count_gains=...)``);
* :func:`calibrate` -- the 2PL fit, with the tournament or without it
  (``mode``), for one judge or several pooled (``judges``); returns an immutable
  :class:`Calibration` that saves to and loads from one directory layout;
* :func:`score_documents`, :func:`insert_documents` -- add documents to a calibration
  without refitting it (an :class:`Extension`, applied with
  :meth:`Calibration.extended`); :func:`select_opponents` picks the documents a new
  one is judged against.
"""

from rcp_ndcg_core.irt import Priors

from rcp_ndcg.calibration._projection import QUERY_ID_SEP, count_gains, judged_bt_l2, read_judgements
from rcp_ndcg.calibration.extend import (
    AnchorReport,
    Extension,
    ExtensionRecord,
    insert_documents,
    score_documents,
    select_opponents,
)
from rcp_ndcg.calibration.fit import Calibration, ThetaRow, calibrate

__all__ = [
    "QUERY_ID_SEP",
    "AnchorReport",
    "Calibration",
    "Extension",
    "ExtensionRecord",
    "Priors",
    "ThetaRow",
    "calibrate",
    "count_gains",
    "insert_documents",
    "judged_bt_l2",
    "read_judgements",
    "score_documents",
    "select_opponents",
]
