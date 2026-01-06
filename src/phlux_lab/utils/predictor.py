from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from phlux_lab.ml.vfm_model import VFMModel
from phlux_lab.utils.preprocessor import Preprocessor


class VFMPredictor:
    """
    Runtime predictor for stacked Phlux VFM models.

    Responsibilities:
      1) Run upstream models (flow, wear)
      2) Inject stacked ML outputs into dataframe
      3) Run final correction model (optional)
    """

    def __init__(
        self,
        *,
        flow_model: VFMModel,
        flow_pp: Preprocessor,
        wear_model: VFMModel | None = None,
        wear_pp: Preprocessor | None = None,
        corr_model: VFMModel | None = None,
        corr_pp: Preprocessor | None = None,
    ):
        self.flow_model = flow_model
        self.flow_pp = flow_pp

        self.wear_model = wear_model
        self.wear_pp = wear_pp

        self.corr_model = corr_model
        self.corr_pp = corr_pp

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _predict_stage(
        self,
        *,
        model: VFMModel,
        pp: Preprocessor,
        df: pd.DataFrame,
    ) -> np.ndarray:
        X = pp.transform_X(df)
        y_scaled = model.predict(X)
        y = pp.inverse_transform_y(y_scaled)
        return y[:, 0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        df: pd.DataFrame,
        apply_correction: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run stacked VFM inference.

        Returns:
        {
            "q_liquid": base flow prediction,
            "hydraulic_wear": wear prediction,
            "q_liquid_corr": corrected flow prediction (if apply_correction=True)
        }
        """

        # --------------------------------------------------
        # 1) Base FLOW prediction
        # --------------------------------------------------
        q_flow_base = self._predict_stage(
            model=self.flow_model,
            pp=self.flow_pp,
            df=df,
        )

        # --------------------------------------------------
        # 2) WEAR prediction
        # --------------------------------------------------
        wear_pred = self._predict_stage(
            model=self.wear_model,
            pp=self.wear_pp,
            df=df,
        )

        results: Dict[str, np.ndarray] = {
            "q_liquid": q_flow_base,
            "hydraulic_wear": wear_pred,
        }

        # --------------------------------------------------
        # 3) FLOW CORRECTION (stacked)
        # --------------------------------------------------
        if apply_correction and self.corr_model is not None:

            # Prepare dataframe for correction stage
            df_corr = df.copy()

            # Inject stacked predictions as features
            df_corr["q_liquid_pred"] = q_flow_base
            df_corr["hydraulic_wear_pred"] = wear_pred

            q_flow_corr = self._predict_stage(
                model=self.corr_model,
                pp=self.corr_pp,
                df=df_corr,
            )

            results["q_liquid_corr"] = q_flow_corr

        return results

