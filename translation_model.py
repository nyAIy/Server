import os
import numpy as np
from indicTrans2.inference.engine import Model, iso_to_flores
import nltk

# Define constants and configurations
CHECKPOINTS_ROOT_DIR = "indicTrans2/checkpoints"
INDIC_LANGUAGES = set(iso_to_flores)
ALLOWED_DIRECTION_STRINGS = {"en-indic", "indic-en", "indic-indic"}
DEFAULT_PIVOT_LANG = "en"
FORCE_PIVOTING = False

class LocalTranslationModel:
    def __init__(self):
        """
        Initialize the Local Translation Model with available checkpoints.
        """
        checkpoint_folders = [f.path for f in os.scandir(CHECKPOINTS_ROOT_DIR) if f.is_dir()]
        if not checkpoint_folders:
            raise RuntimeError(f"No checkpoint folders in: {CHECKPOINTS_ROOT_DIR}")

        self.models = {}
        for checkpoint_folder in checkpoint_folders:
            direction_string = os.path.basename(checkpoint_folder)
            if direction_string not in ALLOWED_DIRECTION_STRINGS:
                raise ValueError(f"Invalid checkpoint folder name: {direction_string}")
            self.models[direction_string] = Model(
                os.path.join(checkpoint_folder, "ct2_int8_model"),
                device="cpu",  # Change to "cuda" if GPU is available
                input_lang_code_format="iso",
                model_type="ctranslate2",
            )

        # Handle pivoting logic
        self.pivot_lang = None
        if "en-indic" in self.models and "indic-en" in self.models:
            if "indic-indic" not in self.models:
                self.pivot_lang = DEFAULT_PIVOT_LANG
            elif FORCE_PIVOTING:
                del self.models["indic-indic"]
                self.pivot_lang = DEFAULT_PIVOT_LANG

    def get_direction_string(self, input_language_id, output_language_id):
        """
        Determine the direction string based on input and output languages.
        """
        if input_language_id == DEFAULT_PIVOT_LANG and output_language_id in INDIC_LANGUAGES:
            return "en-indic"
        elif input_language_id in INDIC_LANGUAGES:
            if output_language_id == DEFAULT_PIVOT_LANG:
                return "indic-en"
            elif output_language_id in INDIC_LANGUAGES:
                return "indic-indic"
        return None

    def translate(self, input_texts, input_language_id, output_language_id):
        """
        Translate input texts using the appropriate model.
        """
        direction_string = self.get_direction_string(input_language_id, output_language_id)
        if not direction_string or direction_string not in self.models:
            raise RuntimeError(f"Language pair not supported: {input_language_id}-{output_language_id}")

        model = self.models[direction_string]
        return model.paragraphs_batch_translate__multilingual(
            [[text, input_language_id, output_language_id] for text in input_texts]
        )


# Initialize the translation model globally
nltk.download("punkt_tab")
translation_model = LocalTranslationModel()
