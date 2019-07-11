from tpot.config.classifier import classifier_config_dict

from tpot.base import TPOTBase


from BayOptPy.tpot.extended_tpot import ExtendedTPOTBase

class ExtendedTPOTClassifier(TPOTBase):
    """TPOT estimator for classification problems."""

    scoring_function = 'accuracy'  # Classification scoring
    default_config_dict = classifier_config_dict  # Classification dictionary
    classification = True
    regression = False
