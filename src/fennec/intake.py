import os
import json
import logging

from typing import Dict, Any

from Bio.Seq import reverse_complement

from fennec.featurizer import (
    categorize_component,
    seq_list_to_seq,
    FEATURE_MAP,
)

logger = logging.getLogger()


class Intake:
    def __init__(self) -> None:
        """Initialize an Intake instance with an empty data dictionary."""
        self.data: Dict[int, Dict[str, Any]] = {}

    def load_from_json_path(self, path: str) -> None:
        """Load data from a JSON file given its file path."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self.data = json.load(f)
        else:
            raise FileNotFoundError(f"File not found at {path}")

    def load_data(self, data: Dict[int, Dict[str, Any]]) -> None:
        """Load data from a dictionary."""
        self.data = data

    def validate_sirna(self, index: int) -> None:
        """Validate siRNA sequence for a given index."""
        sirna = self.data[index]["siRNA_list"]
        si_list = sirna.split()

        # Ensure siRNA has a minimum length requirement
        if len(si_list) < 8:
            logger.warning(
                f"Invalid input: {sirna} for index {index}, siRNA must be at least 8 nt long. Skipping inference."
            )
            self.data.pop(index)
            return

        # Validate each siRNA component
        for s in si_list:
            try:
                categorize_component(s)
            except Exception:
                logger.warning(
                    f"Invalid character in siRNA: {s} at index {index}. Skipping inference."
                )
                self.data.pop(index)
                return

    def validate_meta_data(self, index: int) -> None:
        """Validate metadata fields for a given index."""
        feature_keys = FEATURE_MAP.keys()
        datapoint = self.data[index]

        for key in feature_keys:
            if key in datapoint:
                value = datapoint[key]
                expected_values = FEATURE_MAP[key]
                value_type = type(value)

                if isinstance(expected_values, float):
                    if not isinstance(value, (float, int)):
                        logger.warning(
                            f"Invalid metafeature {key} at index {index}, expected float/int, got {value_type}." +
                            " Setting to 10.0 nM."
                        )
                        datapoint[key] = 10.0
                elif value not in expected_values:
                    logger.warning(
                        f"Unexpected value for {key} at index {index}. Expected: {expected_values}, got: {value}." +
                        " Setting to null."
                    )
            else:
                # Handle missing metadata
                expected_values = FEATURE_MAP[key]
                if isinstance(expected_values, list):
                    datapoint[key] = "Unknown"
                    logger.warning(f'Missing key {key}. Setting placeholder "Unknown".')
                else:
                    datapoint[key] = expected_values
                    logger.warning(
                        f"Missing key {key}. Setting placeholder {expected_values}."
                    )

    def validate_mrna(self, index: int) -> None:
        """Validate mRNA sequence consistency with siRNA."""
        datapoint = self.data[index]
        sirna_list = datapoint["siRNA_list"]
        sirna = seq_list_to_seq(sirna_list)
        datapoint["siRNA"] = sirna

        if "mRNA" not in datapoint:
            logger.warning(
                f"mRNA sequence missing for index {index}. Assigning unmodified siRNA to mRNA."
            )
            datapoint["mRNA"] = sirna
        else:
            # No longer needed, better handling further downstream
            pass
            # mrna = str(datapoint["mRNA"]).replace("U", "T")
            # if reverse_complement(sirna[2:18].replace("U", "T")) not in mrna:
            #     logger.warning(
            #         f"Reverse complement of siRNA not found in mRNA for index {index}. Potential siRNA/mRNA flip." +
            #         " Assigning default sequence."
            #     )
            #     datapoint["mRNA"] = datapoint["siRNA"]

    def validate_input(self) -> None:
        """Perform full validation on all input data."""
        indices = list(self.data.keys())
        for index in indices:
            self.validate_sirna(index)
            if index not in self.data:
                continue
            self.validate_meta_data(index)
            self.validate_mrna(index)
